import os
from multiprocessing import Pool

# Prevent each worker from starting additional BLAS/OpenMP threads.
# This avoids CPU oversubscription when several eigsh calls run in parallel.
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import kwant
import numpy as np
from scipy.sparse import lil_matrix,block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigsh
from Hamiltonian_mathematica_v2 import gs_shifted_bz
from tqdm.auto import tqdm
import time


# =========================
# Precompute hopping blocks
# =========================


        
# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N = 50
Ny = Nz = N
L = 300
Nband = 10
kx = 0.0
Bf = 0.01
hbar = 1.05457e-34
q = 1.602e-19
muB = 5.788e-5
gspin = 2.0
emax_V_um = 7.0
emax_V_a = emax_V_um * 1e-4
Ef_list = np.linspace(0, emax_V_a, 21)
flag = [1,0]
sigma_val = 0.12

# Adjust this if memory is limited. Each process holds its own sparse matrices.
N_PROCESSES = 10 #min(len(Ef_list), os.cpu_count() or 1)


def make_system():
    a = L / (Ny + 1)
    y_center = (Ny - 1) / 2.0

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    def onsite(site):
        y, z = site.tag
        y_centered = y - y_center

        return np.asarray(
            gs_shifted_bz(a, kx, 0, 0, Bf, y_centered),
            dtype=complex,
        )

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag

        hop_dy = int(y1 - y2)
        hop_dz = int(z1 - z2)
        ymid_centered = 0.5 * (y1 + y2) - y_center

        return np.asarray(
            gs_shifted_bz(
                a, kx, hop_dy, hop_dz, Bf, ymid_centered
            ),
            dtype=complex,
        )

    # --- Build lattice ---
    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite

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
def make_system_E(Ef):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):


            if flagy == 1:
                V = -Ef * ((y + 1) * a - L / 2)
            else:
                V = -Ef * ((z + 1) * a - L / 2)

            syst[lat_E(y, z)] = V * np.eye(Nband)

    return syst.finalized()


#---------------------------------------------
#           Create the S matrix
#---------------------------------------------
# --- base 5x5 components ---

zero = csr_matrix((5, 5), dtype=complex)
identity_5 = csr_matrix(np.eye(5, dtype=complex))

smatrix_sparse = 0.5 * bmat([
    [identity_5, zero],
    [zero, -identity_5],
], format="csr")


#---------------------------------------------
#               Main parameters
#---------------------------------------------

H0_WORKER = None
SMATRIXFULL_WORKER = None


def initialize_worker():
    """Build field-independent matrices once in each worker process."""
    global H0_WORKER, SMATRIXFULL_WORKER

    syst = make_system()
    H0_WORKER = syst.hamiltonian_submatrix(sparse=True)

    n_sites = H0_WORKER.shape[0] // Nband
    SMATRIXFULL_WORKER = block_diag(
        [smatrix_sparse] * n_sites,
        format='csr',
    )


def calculate_gfactor(Ef):
    """Calculate the top-valence-band g-factor for one electric field."""
    time_start = time.time()

    syst_E = make_system_E(Ef)
    HE = syst_E.hamiltonian_submatrix(sparse=True)

    H = H0_WORKER + gspin * muB * Bf * SMATRIXFULL_WORKER + HE
    
    # Compute eigenvalues near zero
    vbs = eigsh(H, k=2, sigma=sigma_val, which='LM', return_eigenvectors=False)
    vbs = np.sort(vbs)

    vb1, vb2 = vbs       # valence band top (closer to zero is vb2)
    
    # only gap of vb2, vb1, vb0
    gap_vb = abs(vb1 - vb2)
    # gap_vb1 = abs(vb3 - vb4)
    # gap_vb0 = abs(vb5 - vb6)

    # g-factor of top vb
    gvb = gap_vb / (muB * Bf)

    time_end = time.time()
    elapsed_time = time_end - time_start

    result = [(L / 10), Ef * 1e4, gvb, gap_vb]
    details = (Ef, vb1, vb2, gap_vb, gvb, elapsed_time)
    return result, details


def main():
    gfactors = []

    with Pool(
        processes=N_PROCESSES,
        initializer=initialize_worker,
    ) as pool:
        results = pool.imap_unordered(calculate_gfactor, Ef_list, chunksize=1)

        for result, details in tqdm(
            results,
            total=len(Ef_list),
            desc="Calculating g-factors",
            unit="field",
        ):
            Ef, vb1, vb2, gap_vb, gvb, elapsed_time = details
            minutes, seconds = divmod(int(elapsed_time), 60)

            tqdm.write(
                f"E: {Ef * 1e4: .3f}  "
                f"VB: {vb1: .6f}, {vb2: .6f}  "
                f"gap={gap_vb: .8f}  g={gvb: .3f}  "
                f"time={minutes} min {seconds} sec"
            )
            gfactors.append(result)

    # imap_unordered keeps the progress bar responsive; restore Ef_list order
    # before writing the data file.
    gfactors = np.asarray(sorted(gfactors, key=lambda row: row[1]))

    if flagy == 1:
        output_file = f"gfactor_zz_vs_Ey_N{N}_L{L}.dat"
    else:
        output_file = f"gfactor_zz_vs_Ez_N{N}_L{L}.dat"

    np.savetxt(
        output_file,
        gfactors,
        header="size(nm)  E   g_vb   gap_vb(eV)",
    )
    print(f"Saved {output_file}")


if __name__ == "__main__":
    for flagy in flag:
        main()
