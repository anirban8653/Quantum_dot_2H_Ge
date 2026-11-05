import time
from multiprocessing import Pool, cpu_count

import kwant
import numpy as np
from pymablock import block_diagonalize
from scipy.linalg import eigh, qr
from scipy.sparse import bmat, block_diag, csr_matrix, identity
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm

from Hamiltonian_mathematica_v2 import psi_new_basis, gs_shifted_bz


# ---------------------------------
# Parameters
# ---------------------------------
N = 50
Nband = 10
L = 300
B = 0.01
gspin = 2
muB = 5.788e-5

nef = 20
Ef_V_um = 0.1
Ef = Ef_V_um*1e-4

# Set this to an integer (for example, 30) to select the number of workers.
# None uses all available CPUs, capped by the number of electric-field points.
N_PROCESSES = None

Ny = Nz = N
lat = kwant.lattice.square(norbs=Nband)
lat_E = kwant.lattice.square(norbs=Nband)


# ==================================================
# General Kwant system builder
# ==================================================
def make_system_bz(kx, Bz):
    """Build the 2H-Ge nanowire Hamiltonian for B = (0, 0, Bz)."""
    syst = kwant.Builder()
    a = L / (Ny + 1)
    y_center = (Ny - 1) / 2.0

    def onsite(site):
        y, _ = site.tag
        y_mid_centered = y - y_center
        return gs_shifted_bz(
            a=a,
            kx_base=kx,
            ry=0,
            rz=0,
            Bz=Bz,
            y_mid_centered=y_mid_centered,
        )

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        ry = int(y1 - y2)
        rz = int(z1 - z2)
        y_mid_centered = 0.5 * (y1 + y2) - y_center
        return gs_shifted_bz(
            a=a,
            kx_base=kx,
            ry=ry,
            rz=rz,
            Bz=Bz,
            y_mid_centered=y_mid_centered,
        )

    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite

    # Use only one representative from each Hermitian hopping pair.
    for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(direction, lat, lat)] = hopping

    return syst.finalized()


def make_system_E(flagy):
    """Build the position operator for Ey (flagy=1) or Ez (flagy=0)."""
    syst = kwant.Builder()
    a = L / (Ny + 1)

    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                potential = -((y + 1) * a - L / 2)
            else:
                potential = -((z + 1) * a - L / 2)
            syst[lat_E(y, z)] = potential * np.eye(Nband)

    return syst.finalized()





# ---------------------------------------------
# Create the S matrix
# ---------------------------------------------


zero = csr_matrix((5, 5))
one = np.eye(5,5)
# 10x10 spin matrix (ORDER PRESERVED)
smatrix_sparse = 0.5 * bmat([
    [one, zero],
    [zero, -one]
], format="csr")
smatrixfull = block_diag([smatrix_sparse] * N**2, format="csr")

# ==================================================
# Build linear-in-By magnetic perturbation,
# including orbital + spin Zeeman terms
# ==================================================
print("\nBuilding H(+By) and H(-By)...")
syst_Bp = make_system_bz(0.0, B)
syst_Bm = make_system_bz(0.0, -B)
H_Bp = syst_Bp.hamiltonian_submatrix(sparse=True)
H_Bm = syst_Bm.hamiltonian_submatrix(sparse=True)

# Linear magnetic perturbation, units: eV/T
V_orbital = (H_Bp - H_Bm) / (2.0 * B)
V_spin = gspin * muB * smatrixfull
V_By = V_orbital + V_spin

syst_B0 = make_system_bz(0.0, 0.0)
H0_sparse = syst_B0.hamiltonian_submatrix(sparse=True)

for flagy in [1,0]:
    syst2 = make_system_E(flagy)
    V_Ey = syst2.hamiltonian_submatrix(sparse=True)
    H_list_sparse = [H0_sparse, V_By, V_Ey]



    num_eigenvectors = 2
    evals, evecs = eigsh(H0_sparse, k=num_eigenvectors, sigma=0.00)

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    psi_new = psi_new_basis(psi1, psi2, N)
    psi_new = np.array(psi_new).T

    Heff_coeffs, *_ = block_diagonalize(H_list_sparse, subspace_eigenvectors=[psi_new])



    H00 = np.asarray(Heff_coeffs[(0,0,0,0)])
    H10 = np.asarray(Heff_coeffs[(0,0,1,0)])
    H12 = np.asarray(Heff_coeffs[(0,0,1,2)])
    H14 = np.asarray(Heff_coeffs[(0,0,1,4)])
    H16 = np.asarray(Heff_coeffs[(0,0,1,6)])





    correction = (H10 + H12 * Ef**2 + H14 * Ef**4+ H16 * Ef**6)
    H_eff = H00 + correction * B

    print("correction: \n",correction.round(8))
    w = eigh(correction * B, eigvals_only=True)
    gap = np.abs(w[1] - w[0])
    g = gap / (muB * B)

    print("g : \n",g.round(6))
    if flagy == 1:
        filename = f"Mz_matrix_N{N}_Ey_{Ef_V_um:.2f}.npy"
    else:
        filename = f"Mz_matrix_N{N}_Ez_{Ef_V_um:.2f}.npy"
    np.save(filename,correction)
    print( filename ," saved.")