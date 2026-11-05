import kwant
import numpy as np
from scipy.sparse import csr_matrix, bmat, block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
import time
from pymablock import block_diagonalize
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

from Hamiltonian_mathematica_v2 import psi_new_basis, gs_shifted_by

# ---------------------------------
# Parameters
# ---------------------------------
N           = 50 
Nband       = 10
L           = 300
hbar        = 1.05457e-34  # J*s
e_mass      = 9.11e-31  # kg
e0          = 1.602e-19
dim         = 1.602e-39    # dimensional scaling factor (from energy unit)
B           = 0.01
gspin       = 2
muB         = 5.788e-5
flag = [1,0]
nef = 20
# emax_V_um = 0.06
# emax_V_angstrom = emax_V_um * 1e-4
# efield_list = np.linspace(0, emax_V_angstrom, nef) 

# ==================================================
# Effective-Hamiltonian configuration
# ==================================================
# Each tuple is (power of By, power of E):
#   (0, 0) -> H00
#   (1, 0) -> H10 * By
#   (1, 1) -> H11 * By * E
#   (1, 2) -> H12 * By * E**2
#   (1, 3) -> H13 * By * E**3
#
# This is the ONLY place that needs to change when adding/removing terms.
ACTIVE_HEFF_TERMS = [
    (0, 0),  # H00
    (1, 0),  # H10 * By
    # (1, 1),  # H11 * By * E
    (1, 2),  # H12 * By * E**2
    (1, 4),  # H14 * By * E**4
    (1, 6),  # H14 * By * E**6
]

# Set to None to use all available CPUs, capped by the number of fields.
N_PROCESSES = 8


Ny = Nz = N

# ==================================================
# General Kwant system builder
# ==================================================
lat = kwant.lattice.square(norbs=Nband)
def make_system_by( kx, By):

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    a = L / (Ny + 1)
    z_center = (Nz - 1) / 2.0

    # -----------------------------
    # Onsite function
    # -----------------------------
    def onsite(site):
        y, z = site.tag

        z_mid_centered = z - z_center

        return gs_shifted_by(
            a=a,
            kx_base=kx,
            ry=0,
            rz=0,
            By=By,
            z_mid_centered=z_mid_centered
        )

    # -----------------------------
    # Hopping function
    # -----------------------------
    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag

        # Kwant matrix element H[site1, site2]
        ry = int(y1 - y2)
        rz = int(z1 - z2)

        # midpoint coordinate
        z_mid_centered = 0.5 * (z1 + z2) - z_center

        return gs_shifted_by(
            a=a,
            kx_base=kx,
            ry=ry,
            rz=rz,
            By=By,
            z_mid_centered=z_mid_centered
        )

    # -----------------------------
    # Add sites
    # -----------------------------
    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite

    # -----------------------------
    # Add hoppings
    # -----------------------------
    # Use only one representative from each Hermitian pair.
    hop_directions = [
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
    ]

    for d in hop_directions:
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


# Worker-local data are assigned once by the Pool initializer. A dictionary is
# used so the Pool code does not change when H11, H12, H13, ... are selected.
_worker_heff_terms = None
_worker_B = None
_worker_muB = None


def init_gfactor_worker(heff_terms, magnetic_field, bohr_magneton):
    """Initialize constant data once in each worker process."""
    global _worker_heff_terms, _worker_B, _worker_muB
    _worker_heff_terms = heff_terms
    _worker_B = magnetic_field
    _worker_muB = bohr_magneton


def calculate_gfactor(Ef):
    """Calculate the effective spectrum and g-factor for one field value."""
    first_matrix = next(iter(_worker_heff_terms.values()))
    H_eff = np.zeros_like(first_matrix, dtype=complex)

    for (by_power, e_power), coefficient in _worker_heff_terms.items():
        H_eff += coefficient * _worker_B**by_power * Ef**e_power

    eigenvalues = eigh(H_eff, eigvals_only=True)
    gap = np.abs(eigenvalues[1] - eigenvalues[0])
    gfactor = gap / (_worker_muB * _worker_B)
    return Ef, H_eff, eigenvalues, gap, gfactor


def main():
    t0 = time.time()
    print(f"\nBuilding system for N = {N}")

    # ---------------------------------------------
    # Create the S matrix
    # ---------------------------------------------
    sminus = -1j
    sud = np.zeros((5, 5), dtype=complex)
    sud[0, 0] = sminus
    sud[1, 1] = sminus
    sud[4, 4] = sminus
    sud[2, 3] = sminus
    sud[3, 2] = sminus

    sud_sparse = csr_matrix(sud)
    sdu_sparse = sud_sparse.conjugate()
    zero = csr_matrix((5, 5))

    # 10x10 spin matrix (ORDER PRESERVED)
    smatrix_sparse = 0.5 * bmat([
        [zero, sud_sparse],
        [sdu_sparse, zero]
    ], format="csr")
    smatrixfull = block_diag([smatrix_sparse] * N**2, format="csr")

    # ==================================================
    # Build linear-in-By magnetic perturbation,
    # including orbital + spin Zeeman terms
    # ==================================================
    print("\nBuilding H(+By) and H(-By)...")
    syst_Bp = make_system_by(0.0, B)
    syst_Bm = make_system_by(0.0, -B)
    H_Bp = syst_Bp.hamiltonian_submatrix(sparse=True)
    H_Bm = syst_Bm.hamiltonian_submatrix(sparse=True)

    # Linear magnetic perturbation, units: eV/T
    V_orbital = (H_Bp - H_Bm) / (2.0 * B)
    V_spin = gspin * muB * smatrixfull
    V_By = V_orbital + V_spin

    syst_B0 = make_system_by(0.0, 0.0)
    H0_sparse = syst_B0.hamiltonian_submatrix(sparse=True)
    syst2 = make_system_E()
    V_Ey = syst2.hamiltonian_submatrix(sparse=True)
    H_list_sparse = [H0_sparse, V_By, V_Ey]

    _, evecs = eigsh(H0_sparse, k=2, sigma=0.00)
    evecs_ortho, _ = qr(evecs, mode="economic")
    psi_new = np.asarray(
        psi_new_basis(evecs_ortho[:, 0], evecs_ortho[:, 1], N)
    ).T

    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse,
        subspace_eigenvectors=[psi_new],
    )

    # Materialize only the terms selected in ACTIVE_HEFF_TERMS. The first two
    # indices belong to Pymablock's internal bookkeeping; the final two are
    # the powers of By and E, respectively.
    heff_terms = {
        (by_power, e_power): np.asarray(
            Heff_coeffs[(0, 0, by_power, e_power)]
        )
        for by_power, e_power in ACTIVE_HEFF_TERMS
    }

    selected_terms = " + ".join(
        f"H{by_power}{e_power}" for by_power, e_power in ACTIVE_HEFF_TERMS
    )
    print(f"\nUsing effective-Hamiltonian terms: {selected_terms}")

    # available_workers = cpu_count() if N_PROCESSES is None else N_PROCESSES
    n_workers = 8 #max(1, min(int(available_workers), len(efield_list)))
    print(f"Calculating {len(efield_list)} field points with {n_workers} workers...")

    with Pool(
        processes=n_workers,
        initializer=init_gfactor_worker,
        initargs=(heff_terms, B, muB),
    ) as pool:
        results = list(tqdm(
            pool.imap(calculate_gfactor, efield_list, chunksize=1),
            total=len(efield_list),
            desc="Electric-field sweep",
            unit="field",
        ))

    # imap preserves the order of efield_list, so the output remains ordered.
    for Ef, H_eff, eigenvalues, gap, gfactor in results:
        # print("\nEffective Hamiltonian")
        # print(H_eff.round(8))
        print(
            f"L = {L}, E = {Ef * 1e4:.3f}, "
            # f"ev1 = {eigenvalues[0]:.6f}, ev2 = {eigenvalues[1]:.6f}, "
            f"gap = {gap * 1e6:.4f}, g = {gfactor:.3f}"
        )

    # Save columns as: L [Angstrom], gap [micro-eV], E [V/micrometre], g.
    # flagy = 1 applies Ey, while flagy = 0 applies Ez.
    output_filename = (
        "gy_moulation_Ey.dat" if flagy == 1 else "gy_moulation_Ez.dat"
    )
    output_data = np.asarray([
        [L, gap * 1e6, Ef * 1e4, gfactor]
        for Ef, _, _, gap, gfactor in results
    ])
    np.savetxt(
        output_filename,
        output_data,
        fmt=["%.8f", "%.10e", "%.10e", "%.10e"],
        header="L_Angstrom gap_microeV E_V_per_um g",
    )
    print(f"Saved data to {output_filename}")

    print(f"\nTotal runtime: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    for flagy in flag:
        if flagy == 1:
            emax_V_um = 1.0
        if flagy == 0:
            emax_V_um = 0.075
        emax_V_angstrom = emax_V_um * 1e-4
        efield_list = np.linspace(0, emax_V_angstrom, nef)         
        print(f"\nRunning for flagy = {flagy}")
        main()
    
