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

nef = 20
Ef_V_um = 0.1
Ef = Ef_V_um*1e-4



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
def make_system_E(flagy):

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
        filename = f"My_matrix_N{N}_Ey_{Ef_V_um:.2f}.npy"
    else:
        filename = f"My_matrix_N{N}_Ez_{Ef_V_um:.2f}.npy"
    np.save(filename,correction)
    print( filename ," saved.")