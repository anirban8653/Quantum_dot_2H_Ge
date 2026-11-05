import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import kwant
import time
from scipy.sparse.linalg import eigsh
from scipy.linalg import qr
from pymablock import block_diagonalize
from Hamiltonian_mathematica import gso, dgso, psi_new_basis
np.set_printoptions(linewidth=200, suppress=True, precision=5)

# =========================
# TIMER START
# =========================
t0 = time.time()

# ---------------------------------
# Parameters
# ---------------------------------
N = 100
Nband = 10
L = 300
Ny = Nz = N
E_V_um = 6.0              # electric-field magnitude in V/um
Ef = E_V_um * 1e-4        # V/A
sigma_val = 0.1

# Field direction in the y-z plane: angle theta measured from +y toward +z.
#   theta = 0   -> E along z
#   theta = 90  -> E along y
# Add any extra angles you want to explore (degrees).
THETAS_DEG = [0.0, 90.0, 83.2]

lat = kwant.lattice.square(norbs=Nband)


# ---------------------------------
# Build system main Hamiltonian
# ---------------------------------
def make_system(kx, Ef, theta):
    """theta in radians; E = Ef * (cos(theta) y_hat + sin(theta) z_hat)."""

    syst = kwant.Builder()
    a = L / (Ny + 1)
    Ey, Ez = Ef * np.sin(theta), Ef * np.cos(theta)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            ypos = (y + 1) * a - L / 2
            zpos = (z + 1) * a - L / 2
            V = -(Ey * ypos + Ez * zpos)
            syst[lat(y, z)] = V * np.eye(Nband) + gso_cache[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache[(dy, dz)]

    return syst.finalized()


# ---------------------------------
# Build system dH/dkx
# ---------------------------------
def make_system_kx(kx):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    gso_cache_kx = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache_kx[(dy, dz)] = dgso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache_kx[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache_kx[(dy, dz)]

    return syst.finalized()


def compute_H11(Ef, theta, V_kx):
    """Pymablock first-order-in-kx coefficient H11 in the 2-dim ground-state subspace."""
    syst0 = make_system(0.0, Ef, theta)
    H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

    evals, evecs = eigsh(H0_sparse, k=2, sigma=sigma_val)
    evecs_ortho, _ = qr(evecs, mode="economic")

    psi_new = psi_new_basis(evecs_ortho[:, 0], evecs_ortho[:, 1], N)
    psi_new = np.array(psi_new).T

    Heff_coeffs, *_ = block_diagonalize([H0_sparse, V_kx], subspace_eigenvectors=[psi_new])
    return np.asarray(Heff_coeffs[(0, 0, 1)])


if __name__ == "__main__":

    V_kx = make_system_kx(0.0).hamiltonian_submatrix(sparse=True)

    for theta_deg in THETAS_DEG:
        theta = np.deg2rad(theta_deg)
        print(f"\nPymablock, E at theta = {theta_deg:.2f} deg from y, "
              f"E = {E_V_um} V/um, N = {N}, L = {L}")

        H11 = compute_H11(Ef, theta, V_kx)
        print(H11)

        fname = f"R_mat_th{theta_deg:.2f}_E{E_V_um:.2f}_N{N}_L{L}.npy"
        np.save(fname, H11)
        print(f"Exported: {fname}")

    # =========================
    # Timer
    # =========================
    mins, secs = divmod(time.time() - t0, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")