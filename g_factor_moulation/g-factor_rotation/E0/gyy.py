import kwant
import numpy as np
from scipy.sparse import lil_matrix,block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigsh
from Hamiltonian_real_april import build_H_real_by
import time



# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N = 100
Nband = 10
kx = 0.0

muB = 5.7883818060e-5  # eV/T
gspin = 2.0

B_work = 0.01  # Tesla

# Ef is in eV/Angstrom, equivalently V/Angstrom.
# 1 V/um = 1e-4 V/Angstrom.
# Therefore 0.6e-5 corresponds to 0.06 V/um.
Ef_values = [0, 0.6e-5]

size_list = [300]  # Angstrom, i.e. 30 nm

# ---------------------------------------------
#           Build finite system for Bz + Ey
# ---------------------------------------------

def make_system(Ny, Nz, L, kx, Bf, Ey):
    a = L / (Ny + 1)
    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    # --- Onsite Function ---
    def onsite(site):
        y, z = site.tag
        y_coord = (y + 1) * a - L / 2
        V_E = -Ey * y_coord
        # We call your function for this specific y
        H_dict = build_H_real_by(kx, Ny, L, Bf, z_position=z)
        return H_dict[(0, 0)] + V_E * np.eye(Nband)


    # --- Hopping Function ---
    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        
        # Calculate displacements to find the right key in your H dict
        dy, dz = int(y1 - y2), int(z1 - z2)
        
        # For magnetic phases, it's standard to use the average position
        zmid = (z1 + z2) / 2
        H_dict = build_H_real_by(kx, Ny, L, Bf, z_position=zmid)
        
        return H_dict[(dy, dz)]
    
    # --- Build Lattice ---
    # Assign the function 'onsite' to all sites
    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite

    # Define all hopping directions present in your build_H_real_by
    # Including diagonals for the A6 terms (±1, ±1)
    hop_directions = [
        (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1), (-1, 0), (0, -1)
    ]
    
    for d in hop_directions:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping

    return syst.finalized()



#---------------------------------------------
#           Create the S matrix
#---------------------------------------------
# --- base 5x5 components ---
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

# --- 10x10 spin matrix (ORDER PRESERVED) ---
smatrix_sparse = 0.5 * bmat([
    [zero, sud_sparse],
    [sdu_sparse, zero]
], format='csr')




# ---------------------------------------------
#       Function to extract top VB splitting
# ---------------------------------------------
def get_top_vb_gap(Ny, Nz, L, kx, Bz, Ey, smatrixfull):
    syst = make_system(Ny, Nz, L, kx, Bz, Ey)
    H0 = syst.hamiltonian_submatrix(sparse=True)

    H = H0 + gspin * muB * Bz * smatrixfull

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
            Ny, Nz, L, kx, +B_work, Ey, smatrixfull
        )

        gap_minus, E1m, E2m, eigs_minus = get_top_vb_gap(
            Ny, Nz, L, kx, -B_work, Ey, smatrixfull
        )

        gap_avg = 0.5 * (abs(gap_plus) + abs(gap_minus))
        gzz = gap_avg / (muB * abs(B_work))

        Ey_Vum = Ey * 1e4

        print("\n--------------------------------")
        print(f"Ey = {Ey_Vum:.5f} V/um")
        print(f"gap(+B) = {gap_plus:.10e} eV")
        print(f"gap(-B) = {gap_minus:.10e} eV")
        print(f"gap_avg = {gap_avg:.10e} eV")
        print(f"g_yy = {gzz:.8f}")

        results.append([
            L / 10,
            Ey,
            Ey_Vum,
            B_work,
            gap_plus,
            gap_minus,
            gap_avg,
            gzz
        ])

        elapsed_time = time.time() - time_start
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time taken: {minutes} min {seconds} sec")


results = np.array(results)

np.savetxt(
    f"gfactor_yy_vs_Ey_N{N}_L{size_list[0]}.dat",
    results,
    header="size(nm)  Ey(eV/Angstrom)  Ey(V/um)  Bz(T)  gap_plus(eV)  gap_minus(eV)  gap_avg(eV)  g_zz"
)