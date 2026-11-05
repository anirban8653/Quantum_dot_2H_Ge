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
Bf_values = np.linspace(0.01, 4.0, 10)
muB = 5.788e-5  # eV/T
Ef_V_um = 0.1
Ef = Ef_V_um*1e-4
L = 300
flag  = [1,0]
sigma_val = 0.1

# Each worker stores its own large sparse Hamiltonian. Reduce this value if
# memory is limited.
N_PROCESSES = 4 #min(8, len(Ef_values), os.cpu_count() or 1)

lat = kwant.lattice.square(norbs=Nband)
# ---------------------------------------------
#           Build finite system for Bz + Ey
# ---------------------------------------------
def make_system(Bf):
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
    gxx = gap / (muB * abs(Bf))
    elapsed_time = time.time() - time_start

    return Ef, Bf, gap, eigs[0], eigs[1], gxx, elapsed_time


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

        for Ef_out, Bf, gap, E1, E2, gxx, elapsed_time in tqdm(
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
                f"gxx = {gxx:.4f}  "
                f"time = {minutes} min {seconds} sec"
            )

            results.append([
                L / 10,
                Ef_out * 1e4,
                Bf,
                gap,
                gxx,
            ])

    # Ef is constant, so sort by Bf.
    results = np.asarray(sorted(results, key=lambda row: row[2]))

    if field_direction == 1:
        output_file = f"gfactor_xx_vs_Ey_N{N}_L{L}.dat"
        field_label = "Ey(V/um)"
    else:
        output_file = f"gfactor_xx_vs_Ez_N{N}_L{L}.dat"
        field_label = "Ez(V/um)"

    np.savetxt(
        output_file,
        results,
        header=f"size(nm) {field_label} B(T) gap(eV) g_xx",
    )

    print(f"Saved {output_file}")


if __name__ == "__main__":
    for field_direction in flag:
        main(field_direction)