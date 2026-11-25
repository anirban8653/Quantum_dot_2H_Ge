import os
# import time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import kwant
import numpy as np
from scipy.sparse import csr_matrix, bmat, block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
from pymablock import block_diagonalize
from multiprocessing import Pool
from tqdm import tqdm
# import time

from Hamiltonian_mathematica_v2 import gso, pseudospin_basis_spin_up_first, g_vector
from numerical_gfactor import numerical_g_sweep, save_numerical
np.set_printoptions(linewidth=150)


_H00 = None
_H10 = None
_H12 = None
_H14 = None


# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N= 50
Nband = 10
kx = 0.0
hbar = 1.05457e-34
q = 1.602e-19
muB = 5.788e-5
gspin = 2.0
Bf = 0.01 #applied magnetic field in T
Ny = Nz = N
L = 300
flag = [1, 0]
nef = 20


# ---------------------------------------------
#               SPIN MATRIX
# ---------------------------------------------
def build_spin_matrix():
    sminus = 1

    sud = np.zeros((5, 5), dtype=complex)
    sud[0, 0] = sminus
    sud[1, 1] = sminus
    sud[4, 4] = sminus
    sud[2, 3] = sminus
    sud[3, 2] = sminus

    sud_sparse = csr_matrix(sud)
    sdu_sparse = sud_sparse.conjugate()
    zero = csr_matrix((5, 5))

    smatrix = 0.5 * bmat([
        [zero, sud_sparse],
        [sdu_sparse, zero]
    ], format='csr')

    return smatrix


lat = kwant.lattice.square(norbs=Nband)

def make_system_bx(L, Bx):
    """
    Orbital Hamiltonian for B = (Bx, 0, 0), gauge A = (0, 0, Bx y).

    Conventions identical to gfactor_By.py / gfactor_Bz.py and to the signed
    scripts in large_electric_field/:
      * Kwant element H[site1, site2] uses the block gso(y1 - y2, z1 - z2)
      * Peierls phase exp(-i e/hbar A.(r1 - r2)) = exp(-i e Bx a^2 y_mid (z1 - z2) / hbar)
        (same electron-charge sign as kx -> kx + e A_x / hbar used for By, Bz)
      * y_mid is measured from the wire centre.
    """
    syst = kwant.Builder()

    a = L / (N + 1)
    a_m = a * 1e-10
    y_center = (Ny - 1) / 2.0

    phase_per_tesla = q * a_m**2 / hbar

    gso_cache = {}

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(
                a, kx, dy, dz
            )

    # Onsite terms
    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache[(0, 0)]

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag

        ry = int(y1 - y2)
        rz = int(z1 - z2)

        val = gso_cache[(ry, rz)]

        y_mid_centered = 0.5 * (y1 + y2) - y_center

        phi = phase_per_tesla * y_mid_centered * rz

        phase = np.exp(-1j * Bx * phi)

        return val * phase

    # One representative per Hermitian pair (the reverse is added by Kwant).
    for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping

    return syst.finalized()


lat_E = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system H_E
# ---------------------------------
def make_system_E():

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                V = - ((y + 1) * a - L / 2)
            else:
                V = - ((z + 1) * a - L / 2)
            syst[lat_E(y, z)] = V * np.eye(Nband) 

    return syst.finalized()



def build_effective_hamiltonian_coefficients():
    """Compute the Ef-independent effective-Hamiltonian coefficients once."""
    dB = Bf  # Tesla

    syst_0 = make_system_bx(L, 0.0)
    syst_p = make_system_bx(L, +dB)
    syst_m = make_system_bx(L, -dB)

    H0_sparse = syst_0.hamiltonian_submatrix(sparse=True)
    Horb_p = syst_p.hamiltonian_submatrix(sparse=True)
    Horb_m = syst_m.hamiltonian_submatrix(sparse=True)

    # Orbital coefficient in eV/T
    V_orbital = (Horb_p - Horb_m) / (2.0 * dB)

    # Spin coefficient in eV/T
    smatrix_sparse = build_spin_matrix()
    smatrixfull = block_diag([smatrix_sparse] * (N**2), format='csr')
    V_spin = gspin * muB * smatrixfull

    # Total first-order magnetic perturbation
    V_Bx = V_orbital + V_spin

    syst2 = make_system_E()
    V_E = syst2.hamiltonian_submatrix(sparse=True)


    H_list_sparse = [H0_sparse, V_Bx, V_E]

    num_eigenvectors = 2
    evals, evecs = eigsh(H0_sparse, k=num_eigenvectors, sigma=0.00)

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    # columns (|up>, |down>): needed for the sign of g
    psi_new = pseudospin_basis_spin_up_first(psi1, psi2, N)

    Heff_coeffs, *_ = block_diagonalize(H_list_sparse, subspace_eigenvectors=[psi_new])



    H00 = np.asarray(Heff_coeffs[(0,0,0,0)])
    H10 = np.asarray(Heff_coeffs[(0,0,1,0)])
    H12 = np.asarray(Heff_coeffs[(0,0,1,2)])
    H14 = np.asarray(Heff_coeffs[(0,0,1,4)])
    H16 = np.asarray(Heff_coeffs[(0,0,1,6)])

    # inputs for the exact-diagonalization (numerical) comparison
    numerical_inputs = (H0_sparse, V_Bx, V_E, psi_new)

    return H00, H10, H12, H14, H16, numerical_inputs


def init_worker(H00, H10, H12, H14, H16):
    """Store the small coefficient matrices once in each worker process."""
    global _H00, _H10, _H12, _H14, _H16
    _H00 = H00
    _H10 = H10
    _H12 = H12
    _H14 = H14
    _H16 = H16



def calculate_efield(Ef):
    """Calculate the gap and signed g_xx for one electric-field value."""
    # dH_eff/dBx (eV/T) in the (|up>, |down>) pseudospin basis
    H_B = _H10 + _H12 * Ef**2 + _H14 * Ef**4 + _H16 * Ef**6
    H_eff = _H00 + H_B * Bf
    w = eigh(H_eff, eigvals_only=True)
    gap = np.abs(w[1] - w[0])
    gvec = g_vector(H_B, muB)              # (g_xx, g_xy, g_xz)
    # |gvec| = gap / (muB Bf); sign taken from the diagonal component g_xx
    g = np.sign(gvec[0]) * np.linalg.norm(gvec)
    return [L, gap, Ef, g], w, gvec


def main():
    H00, H10, H12, H14, H16, numerical_inputs = build_effective_hamiltonian_coefficients()

    # imap preserves efield_list ordering while yielding completed results so
    # tqdm can update continuously.  Limit workers to the number of tasks.
    # nworkers = min(len(efield_list), os.cpu_count() or 1)
    nworkers = 8
    with Pool(
        processes=nworkers,
        initializer=init_worker,
        initargs=(H00, H10, H12, H14, H16),
    ) as pool:
        calculated = list(tqdm(
            pool.imap(calculate_efield, efield_list),
            total=len(efield_list),
            desc="Electric fields",
            unit="field",
        ))

    results = [result for result, _, _ in calculated]
    for result, w, gvec in calculated:
        _, _, Ef, g = result
        print(
            f"L = {L}, E = {Ef*1e4:.3f}, ev1 = {w[0]:.6f}, "
            f"ev2 = {w[1]:.6f}, g = {g:.4f}, "
            f"(g_xx, g_xy, g_xz) = ({gvec[0]:.4f}, {gvec[1]:.4f}, {gvec[2]:.4f})"
        )

    output_filename = "gx_moulation_Ey.dat" if flagy == 1 else "gx_moulation_Ez.dat"
    np.savetxt(
        output_filename,
        np.asarray(results),
        header="L gap E g_xx(signed)",
        fmt="%.12e",
    )
    print(f"Saved results to {output_filename}")

    # ---------- numerical (exact diagonalization) g_xx ----------
    H0_sparse, V_Bx, V_E, psi_new = numerical_inputs
    rows = numerical_g_sweep(
        H0_sparse, V_E, V_Bx, psi_new, efield_list,
        component="xx", muB=muB, B=Bf, n_workers=nworkers,
    )
    save_numerical(rows, "xx", "y" if flagy == 1 else "z", N, L)


if __name__ == "__main__":
    for flagy in flag:
        if flagy == 1:
            emax_V_um = 1.0
        if flagy == 0:
            emax_V_um = 0.075
        emax_V_angstrom = emax_V_um * 1e-4
        efield_list = np.linspace(0, emax_V_angstrom, nef) 
        main()

    
