import os
from multiprocessing import Pool

# Avoid nested BLAS/OpenMP parallelism inside multiprocessing workers.
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import kwant
import numpy as np
from scipy.sparse import csr_matrix, block_diag, bmat
from scipy.sparse.linalg import eigsh
import time
from tqdm.auto import tqdm

# from Hamiltonian_real_april import build_H_real
from Hamiltonian_mathematica_v2 import gso
# print matrix without warping lines in terminal
np.set_printoptions(linewidth=150)


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
Bf = 0.01
muB = 5.788e-5  # eV/T
# Ef_values = np.linspace(0, 0.6e-5, 10)
L = 300
flag  = [1,0]

# Each worker stores its own large sparse Hamiltonian. Reduce this value if
# memory is limited.
N_PROCESSES = 5 #min(8, len(Ef_values), os.cpu_count() or 1)

lat = kwant.lattice.square(norbs=Nband)
# ---------------------------------------------
#           Build finite system for Bz + Ey
# ---------------------------------------------
def make_system():
    syst = kwant.Builder()

    a = L / (N + 1)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache[(0, 0)] 


    def hopping(site1, site2):
    
        yi, zi = site1.tag
        yj, zj = site2.tag

        dy = yj - yi
        dz = zj - zi

        val = gso_cache.get((dy, dz), None)
        if val is None:
            return None

        # lattice spacing in meters
        a_m = a * 1e-10

        # Peierls prefactor: q B a^2 / hbar
        phase_prefactor = q * Bf * a_m**2 / hbar

        # midpoint y coordinate in lattice-index units
        y_mid = 0.5 * (yi + yj)

        # Peierls phase for A_z = Bx y
        phase = np.exp(-1j * phase_prefactor * y_mid * dz)

        return val * phase
    
    # ---------- Apply hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = hopping

    return syst.finalized()
#---------------------------------------------
#           Create the S matrix
#---------------------------------------------
# --- base 5x5 components ---
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

# --- 10x10 spin matrix (ORDER PRESERVED) ---
smatrix_sparse = 0.5 * bmat([
    [zero, sud_sparse],
    [sdu_sparse, zero]
], format='csr')



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




# ---------------------------------------------
#       Function to extract top VB splitting
# ---------------------------------------------
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
        format="csr",
    )


def get_top_vb_gap(Ef):
    time_start = time.time()

    syst_E = make_system_E(Ef)
    HE = syst_E.hamiltonian_submatrix(sparse=True)

    H = H0_WORKER + gspin * muB * Bf * SMATRIXFULL_WORKER + HE

    eigs = eigsh(
        H,
        k=2,
        sigma=0,
        which="LM",
        return_eigenvectors=False
    )

    eigs = np.sort(eigs)

    gap = abs(eigs[1] - eigs[0])
    gxx = gap / (muB * abs(Bf))

    elapsed_time = time.time() - time_start
    return Ef, gap, eigs[0], eigs[1], gxx, elapsed_time



# ---------------------------------------------
#               Main calculation
# ---------------------------------------------
def main():
    results = []

    with Pool(
        processes=N_PROCESSES,
        initializer=initialize_worker,
    ) as pool:
        field_results = pool.imap_unordered(
            get_top_vb_gap,
            Ef_values,
            chunksize=1,
        )

        for Ef, gap, E1, E2, gxx, elapsed_time in tqdm(
            field_results,
            total=len(Ef_values),
            desc="Electric-field sweep",
            unit="field",
        ):
            minutes, seconds = divmod(int(elapsed_time), 60)
            tqdm.write(
                f"E = {Ef * 1e4:.4f} V/um  "
                f"E1 = {E1:.8f} eV  E2 = {E2:.8f} eV  "
                f"gap = {gap:.8e} eV  gxx = {gxx:.4f}  "
                f"time = {minutes} min {seconds} sec"
            )

            results.append([
                L / 10,
                Ef * 1e4,
                gap,
                gxx,
            ])

    # imap_unordered keeps the progress bar responsive. Sort by electric field
    # so the output file has the same order as Ef_values.
    results = np.asarray(sorted(results, key=lambda row: row[1]))

    if flagy == 1:
        output_file = f"gfactor_xx_vs_Ey_N{N}_L{L}.dat"
        field_label = "Ey(V/um)"
    else:
        output_file = f"gfactor_xx_vs_Ez_N{N}_L{L}.dat"
        field_label = "Ez(V/um)"

    np.savetxt(
        output_file,
        results,
        header=f"size(nm)  {field_label}  gap(eV)  g_xx",
    )
    print(f"Saved {output_file}")


if __name__ == "__main__":
    for flagy in flag:
        if flagy == 1:
            Ef_values = np.linspace(0, 1.0*1e-4, 10)
        else:
            Ef_values = np.linspace(0, 0.075*1e-4, 10)
        main()
