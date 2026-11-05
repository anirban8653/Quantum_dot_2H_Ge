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
N= 50
Ny = Nz = N
Nband = 10
kx = 0.0

hbar = 1.05457e-34
q = 1.602e-19
gspin = 2.0
Bf_values = np.linspace(0.01, 4.0, 10)
muB = 5.788e-5  # eV/T
Ef_V_um = 0.1
Ef = Ef_V_um * 1e-4
L = 300
flag  = [1,0]
sigma_val = 0.12
# Adjust this if memory is limited. Each process holds its own sparse matrices.
N_PROCESSES = 4 #min(len(Ef_list), os.cpu_count() or 1)


def make_system(Bf):
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

def make_system_E(field_direction):
    syst = kwant.Builder()
    a = L / (Ny + 1)

    for y in range(Ny):
        for z in range(Nz):
            if field_direction == 1:
                coordinate = (y + 1) * a - L / 2
            else:
                coordinate = (z + 1) * a - L / 2

            V = -Ef * coordinate
            syst[lat_E(y, z)] = V * np.eye(Nband)

    return syst.finalized()



#---------------------------------------------
#           Create the S matrix
#---------------------------------------------
# --- base 5x5 components ---

zero = csr_matrix((5, 5))
iden = np.eye(5, 5)
# --- 10x10 spin matrix (ORDER PRESERVED) ---
smatrix_sparse = 0.5 * bmat([
    [iden, zero],
    [zero, -iden]
], format='csr')




# ---------------------------------------------
#       Function to extract top VB splitting
# ---------------------------------------------
HSPIN_WORKER = None


def initialize_worker():
    """Construct matrices that do not depend on Bf or field direction."""
    global HSPIN_WORKER

    n_sites = Ny * Nz
    HSPIN_WORKER = block_diag(
        [smatrix_sparse] * n_sites,
        format="csr",
    )


def get_top_vb_gap(task):
    Bf, field_direction = task
    time_start = time.time()

    # H0 depends on Bf through the orbital Peierls phase.
    syst = make_system(Bf)
    H0 = syst.hamiltonian_submatrix(sparse=True)

    # Electric potential depends on field direction.
    syst_E = make_system_E(field_direction)
    HE = syst_E.hamiltonian_submatrix(sparse=True)

    H = (
        H0
        + gspin * muB * Bf * HSPIN_WORKER
        + HE
    )

    eigs = eigsh(
        H,
        k=2,
        sigma=sigma_val,
        which="LM",
        return_eigenvectors=False,
    )
    eigs = np.sort(eigs)

    gap = abs(eigs[1] - eigs[0])
    gzz = gap / (muB * abs(Bf))
    elapsed_time = time.time() - time_start

    return Ef, Bf, gap, eigs[0], eigs[1], gzz, elapsed_time


# ---------------------------------------------
#               Main calculation
# ---------------------------------------------
def main(field_direction):
    tasks = [(Bf, field_direction) for Bf in Bf_values]
    results = []

    with Pool(
        processes=min(N_PROCESSES, len(tasks)),
        initializer=initialize_worker,
    ) as pool:
        field_results = pool.imap_unordered(
            get_top_vb_gap,
            tasks,
            chunksize=1,
        )

        for Ef_out, Bf, gap, E1, E2, gzz, elapsed_time in tqdm(
            field_results,
            total=len(tasks),
            desc="Magnetic-field sweep",
            unit="field",
        ):
            minutes, seconds = divmod(int(elapsed_time), 60)

            tqdm.write(
                f"E = {Ef_out * 1e4:.4f} V/um  "
                f"B = {Bf:.4f} T  "
                f"E1 = {E1:.8f} eV  "
                f"E2 = {E2:.8f} eV  "
                f"gap = {gap:.8e} eV  "
                f"gzz = {gzz:.4f}  "
                f"time = {minutes} min {seconds} sec"
            )

            results.append([
                L / 10,
                Ef_out * 1e4,
                Bf,
                gap,
                gzz,
            ])

    # Ef is constant, so sort by Bf.
    results = np.asarray(sorted(results, key=lambda row: row[2]))

    if field_direction == 1:
        output_file = f"gfactor_zz_vs_Ey_N{N}_L{L}.dat"
        field_label = "Ey(V/um)"
    else:
        output_file = f"gfactor_zz_vs_Ez_N{N}_L{L}.dat"
        field_label = "Ez(V/um)"

    np.savetxt(
        output_file,
        results,
        header=f"size(nm) {field_label} B(T) gap(eV) g_zz",
    )

    print(f"Saved {output_file}")


if __name__ == "__main__":
    for field_direction in flag:
        main(field_direction)