"""
Export the 2x2 Zeeman matrix M_x (eV/T) of the top valence Kramers pair of a
2H-Ge nanowire at ONE electric field (B || x).

    H_Z = B_x M_x + B_y M_y + B_z M_z        (B in tesla, H_Z in eV)

M_x = P^dag (dH/dB_x) P, with dH/dB_x = orbital part (central difference
(H(+B) - H(-B)) / 2B, Peierls phases for A = (0, 0, B y)) + spin part gspin * muB * S_x.

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
from Hamiltonian_mathematica_v2 import (gso, spin_matrix_small,
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
E_DIRECTION = "z"    # "y" -> E_y, "z" -> E_z

SX, SY, SZ = spin_matrix_small()


# ---------------------------------------------
#               SYSTEM
# ---------------------------------------------
def make_system(B):
    """B || x: A = (0, 0, B y), Peierls phases on the y/z hoppings."""
    a = L / (Ny + 1)
    y_center = (Ny - 1) / 2.0
    phase_prefactor = q * B * (a * 1e-10) ** 2 / hbar   # q B a^2 / hbar
    blocks = {(dy, dz): np.asarray(gso(a, kx, dy, dz), dtype=complex)
              for dy in (-1, 0, 1) for dz in (-1, 0, 1)}

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    def onsite(site):
        return blocks[(0, 0)]

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        ymid = 0.5 * (y1 + y2) - y_center
        phase = np.exp(1j * phase_prefactor * ymid * (z1 - z2))
        return blocks[(int(y1 - y2), int(z1 - z2))] * phase

    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite
    for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping
    return syst.finalized()


def site_tags(fsyst):
    return np.array([s.tag for s in fsyst.sites], dtype=float)


def potential_diagonal(Ef, direction, tags):
    """V = -E (r - L/2) on the diagonal, in the finalized site order (Ef in V/Angstrom)."""
    a = L / (Ny + 1)
    coord = tags[:, 0] if direction == "y" else tags[:, 1]
    V = -Ef * ((coord + 1) * a - L / 2)
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

    Ef = E_V_um * 1e-4                                   # V/Angstrom
    H.setdiag(H.diagonal() + potential_diagonal(Ef, E_DIRECTION, tags))

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
    DP = np.column_stack([dH_orb @ P[:, j] + gspin * muB * apply_spin_operator(P[:, j], SX)
                          for j in range(2)])
    M = P.conj().T @ DP
    M = 0.5 * (M + M.conj().T)                           # 2x2, eV/T

    out = f"Mx_E{E_DIRECTION}{E_V_um:.3f}Vum_N{N}_L{int(L)}.npy"
    np.save(out, M)

    # checks
    sz = [np.vdot(p, apply_spin_operator(p, SZ)).real for p in (p1, p2)]
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    g_vec = np.array([np.trace(M @ s).real / muB for s in pauli])

    print(f"E_{E_DIRECTION} = {E_V_um} V/um,  pair energy = {w[pair].mean():.6f} eV,  "
          f"gap to next level = {gap_next*1e3:.3f} meV")
    print(f"M_x (eV/T) =\n{np.array2string(M, precision=8)}")
    print(f"trace = {np.trace(M).real:.2e} eV/T (should be ~0)")
    print(f"|g_x| = {np.linalg.norm(g_vec):.5f},  pseudospin vector = {np.round(g_vec, 5)}")
    print(f"<S_z> in pair = {sz[0]:.4f}, {sz[1]:.4f}")
    if abs(sz[0] - sz[1]) < 1e-3:
        print("WARNING: <S_z> nearly degenerate in the pair; basis convention ill-defined.")
    print(f"Saved {out}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
