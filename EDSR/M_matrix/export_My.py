"""
Export the 2x2 Zeeman matrix M_y (eV/T) of the top valence Kramers pair of a
2H-Ge nanowire for several electric-field directions in the y-z plane (B || y).

    H_Z = B_x M_x + B_y M_y + B_z M_z        (B in tesla, H_Z in eV)

M_y = P^dag (dH/dB_y) P, with dH/dB_y = orbital part (central difference
(H(+B) - H(-B)) / 2B, kx shift for A = (B z, 0, 0)) + spin part gspin * muB * S_y.

Basis convention (identical in export_Mx/My/Mz.py, so the three matrices can be
added): psi_new_basis -- S_z diagonal in the pair (column 0 = lower <S_z>),
<1|S_x|2> real and positive. Only a global phase is left, which drops out of M.
"""
import os

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import time
import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
from Hamiltonian_mathematica_v2 import (gs_shifted_by, spin_matrix_small,
                                        apply_spin_operator, psi_new_basis)

# ---------------------------------------------
#               PARAMETERS
# ---------------------------------------------
N = 100
Ny = Nz = N
L = 300.0            # Angstrom
Nband = 10
kx = 0.0
Bf = 0.01            # T, finite-difference step for the orbital term
muB = 5.788e-5       # eV/T
gspin = 2.0
hbar = 1.05457e-34
q = 1.602e-19
sigma_val = 0.12     # eV, shift-invert target
N_EIG = 4            # the pair + next pair (gap check)

E_V_um = 6.0         # electric field strength (V/um)
# Field direction: angle theta (deg) from +y toward +z  (0 -> E_y, 90 -> E_z)
THETAS_DEG = [0.0, 90.0, 83.2]

SX, SY, SZ = spin_matrix_small()


# ---------------------------------------------
#               SYSTEM
# ---------------------------------------------
def make_system(B):
    """B || y: A = (B z, 0, 0), kx shift (gs_shifted_by)."""
    a = L / (Ny + 1)
    z_center = (Nz - 1) / 2.0

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    def onsite(site):
        y, z = site.tag
        return np.asarray(gs_shifted_by(a, kx, 0, 0, B, z - z_center), dtype=complex)

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        zmid = 0.5 * (z1 + z2) - z_center
        return np.asarray(gs_shifted_by(a, kx, int(y1 - y2), int(z1 - z2), B, zmid),
                          dtype=complex)

    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite
    for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping
    return syst.finalized()


def site_tags(fsyst):
    return np.array([s.tag for s in fsyst.sites], dtype=float)


def potential_diagonal(Ef, theta, tags):
    """V = -E.(r - L/2) with E = Ef (cos theta, sin theta) in the (y, z) plane,
    on the diagonal in the finalized site order (Ef in V/Angstrom, theta in rad)."""
    a = L / (Ny + 1)
    ypos = (tags[:, 0] + 1) * a - L / 2
    zpos = (tags[:, 1] + 1) * a - L / 2
    V = -Ef * (np.sin(theta) * ypos + np.cos(theta) * zpos)
    return np.repeat(V, Nband)


# ---------------------------------------------
#               MAIN
# ---------------------------------------------
def main():
    t0 = time.time()

    f0, fp, fm = make_system(0.0), make_system(+Bf), make_system(-Bf)
    tags = site_tags(f0)
    assert np.array_equal(site_tags(fp), tags) and np.array_equal(site_tags(fm), tags)
    H = f0.hamiltonian_submatrix(sparse=True).tocsr()
    dH_orb = ((fp.hamiltonian_submatrix(sparse=True)
               - fm.hamiltonian_submatrix(sparse=True)) / (2 * Bf)).tocsr()

    H0 = H

    for theta_deg in THETAS_DEG:
        Ef = E_V_um * 1e-4                                   # V/Angstrom
        H = H0.copy()
        H.setdiag(H.diagonal() + potential_diagonal(Ef, np.deg2rad(theta_deg), tags))

        # Kramers pair closest to sigma_val (B = 0)
        w, v = eigsh(H, k=N_EIG, sigma=sigma_val, which="LM")
        order = np.argsort(np.abs(w - sigma_val))
        pair, rest = order[:2], order[2:]
        gap_next = np.min(np.abs(w[rest][:, None] - w[pair][None, :]))

        # eigsh does not return an orthonormal basis inside a degenerate pair
        P, _ = np.linalg.qr(v[:, pair])
        p1, p2 = psi_new_basis(P[:, 0], P[:, 1], None)       # fixed basis convention
        P = np.column_stack((p1, p2))

        # projected Zeeman matrix
        DP = np.column_stack([dH_orb @ P[:, j] + gspin * muB * apply_spin_operator(P[:, j], SY)
                              for j in range(2)])
        M = P.conj().T @ DP
        M = 0.5 * (M + M.conj().T)                           # 2x2, eV/T

        out = f"My_Eth{theta_deg:.2f}_{E_V_um:.3f}Vum_N{N}_L{int(L)}.npy"
        np.save(out, M)

        # checks
        sz = [np.vdot(p, apply_spin_operator(p, SZ)).real for p in (p1, p2)]
        pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
        g_vec = np.array([np.trace(M @ s).real / muB for s in pauli])

        print(f"E = {E_V_um} V/um at theta = {theta_deg:.2f} deg,  pair energy = {w[pair].mean():.6f} eV,  "
              f"gap to next level = {gap_next*1e3:.3f} meV")
        print(f"M_y (eV/T) =\n{np.array2string(M, precision=8)}")
        print(f"trace = {np.trace(M).real:.2e} eV/T (should be ~0)")
        print(f"|g_y| = {np.linalg.norm(g_vec):.5f},  pseudospin vector = {np.round(g_vec, 5)}")
        print(f"<S_z> in pair = {sz[0]:.4f}, {sz[1]:.4f}")
        if abs(sz[0] - sz[1]) < 1e-3:
            print("WARNING: <S_z> nearly degenerate in the pair; basis convention ill-defined.")
        print(f"Saved {out}   ({time.time() - t0:.0f} s)\n")


if __name__ == "__main__":
    main()
