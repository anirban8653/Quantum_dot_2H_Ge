import kwant
import numpy as np
from scipy.sparse import csr_matrix, block_diag, bmat
from scipy.sparse.linalg import eigsh
import time

# from Hamiltonian_real_april import build_H_real
from Hamiltonian_mathematica_v2 import gso
# print matrix without warping lines in terminal
np.set_printoptions(linewidth=150)


# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N= 100
Ny = Nz = N
Nband = 10
kx = 0.0

hbar = 1.05457e-34
q = 1.602e-19
gspin = 2.0
Bf = 0.01
muB = 5.7883818060e-5  # eV/T


# Ef is in eV/Angstrom, equivalently V/Angstrom.
# 1 V/um = 1e-4 V/Angstrom.
# Therefore 0.6e-5 corresponds to 0.06 V/um.
Ef_values = [0, 0.6e-5]

size_list = [300]  # Angstrom, i.e. 30 nm



lat = kwant.lattice.square(norbs=Nband)


# ---------------------------------------------
#           Build finite system for Bz + Ey
# ---------------------------------------------
def make_system(Bf, L, Ey):
    syst = kwant.Builder()

    a = L / (N + 1)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            y_coord = (y + 1) * a - L / 2
            V_E = -Ey * y_coord
            syst[lat(y, z)] = gso_cache[(0, 0)] + V_E * np.eye(Nband)


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


# ---------------------------------------------
#       Function to extract top VB splitting
# ---------------------------------------------
def get_top_vb_gap(Bf, L, Ey, smatrixfull):
    syst = make_system(Bf, L, Ey)
    H0 = syst.hamiltonian_submatrix(sparse=True)

    H = H0 + gspin * muB * Bf * smatrixfull

    eigs = eigsh(
        H,
        k=8,
        sigma=0,
        which="LM",
        return_eigenvectors=False,
        tol=1e-10
    )

    eigs = np.sort(eigs)

    vb_states = eigs[eigs < 0]

    if len(vb_states) >= 2:
        E1, E2 = vb_states[-2], vb_states[-1]
    else:
        E1, E2 = eigs[-2], eigs[-1]

    gap = abs(E2 - E1)

    return gap, E1, E2, eigs



# ---------------------------------------------
#               Main calculation
# ---------------------------------------------
results = []

for size in size_list:

    L = size
    Ny = Nz = N

    print("\n==============================")
    print(f"L = {L / 10:.1f} nm")
    print("==============================")

    smatrixfull = block_diag([smatrix_sparse] * (Ny * Nz), format="csr")

    for Ey in Ef_values:

        time_start = time.time()

        # Use +B and -B average for numerical stability
        gap_plus, E1p, E2p, eigs_plus = get_top_vb_gap(
            +Bf, L, Ey, smatrixfull
        )

        gap_minus, E1m, E2m, eigs_minus = get_top_vb_gap(
           -Bf, L, Ey, smatrixfull
        )

        gap_avg = 0.5 * (abs(gap_plus) + abs(gap_minus))
        gxx = gap_avg / (muB * abs(Bf))

        Ey_Vum = Ey * 1e4

        print("\n--------------------------------")
        print(f"Ey = {Ey_Vum:.5f} V/um")
        print(f"gap(+B) = {gap_plus:.10e} eV")
        print(f"gap(-B) = {gap_minus:.10e} eV")
        print(f"gap_avg = {gap_avg:.10e} eV")
        print(f"g_xx = {gxx:.8f}")

        results.append([
            L / 10,
            Ey,
            Ey_Vum,
            Bf,
            gap_plus,
            gap_minus,
            gap_avg,
            gxx
        ])

        elapsed_time = time.time() - time_start
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time taken: {minutes} min {seconds} sec")


results = np.array(results)

np.savetxt(
    f"gfactor_xx_vs_Ey_N{N}_L{size_list[0]}.dat",
    results,
    header="size(nm)  Ey(eV/Angstrom)  Ey(V/um)  Bz(T)  gap_plus(eV)  gap_minus(eV)  gap_avg(eV)  g_zz"
)