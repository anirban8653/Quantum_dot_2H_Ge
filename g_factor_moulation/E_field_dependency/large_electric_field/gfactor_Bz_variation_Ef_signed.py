"""
Signed g_zz of the top valence Kramers pair of a 2H-Ge nanowire versus an
electric field along y or z (B || z only).

Method
------
1. For every field E, the Kramers pair P = (|1>, |2>) is computed at B = 0 and the
   first-order Zeeman operator for B || z (orbital/Peierls + spin) is projected:
       M_z = P^dag (dH/dB_z) P = (muB/2) sum_beta g_{z beta} sigma_beta
       g_{z beta} = Tr(M_z sigma_beta) / muB,      |g_z| = splitting / (muB B)
   The orbital part is a central difference (H(+B) - H(-B)) / 2B.

2. Sign of g_zz without any other field direction:
   The pseudospin basis is carried smoothly from one field to the next
   (P_E -> P_E X, with X the unitary closest to P_prev^dag P_E). In that smooth
   basis the vector g_z(E) changes continuously; if g_zz passes through zero
   it reverses direction, i.e. g_z(E_i) . g_z(E_{i-1}) < 0. The sign is
   accumulated from these flips and referenced to E = 0.
   Check: `overlap_min` (smallest singular value of P_prev^dag P_E) should stay
   close to 1. If it drops (say < 0.7) the field step is too large or another
   level crosses the pair -> use more field points.

Output
------
gzz_vs_E{y,z}_N{N}_L{L}.dat : E, g_zz (signed), Delta g_zz (%), |g_z|,
                              overlap_min, E_pair, gap_next
gzz_vs_E_N{N}_L{L}.png      : g_zz vs E and percentage change vs E
"""
import os
from multiprocessing import Pool

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import time
import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
from Hamiltonian_mathematica_v2 import gs_shifted_bz, spin_matrix_small, apply_spin_operator
from tqdm.auto import tqdm

# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N = int(os.environ.get("GF_N", 100))
Ny = Nz = N
L = float(os.environ.get("GF_L", 300))
Nband = 10
kx = 0.0
Bf = 0.01            # T, finite-difference step for the orbital term
muB = 5.788e-5       # eV/T
gspin = 2.0
sigma_val = 0.12     # eV, shift-invert target
N_EIG = 4            # 2 Kramers pairs: the pair + next level (level check)

emax_V_um = 7.00
N_E = 21                                       # uniformly spaced fields
E_LIST_V_um = np.linspace(0, emax_V_um, N_E)   # V/um
Ef_list = E_LIST_V_um * 1e-4                   # V/Angstrom

FIELD_DIRECTIONS = ["y", "z"]                  # "y" -> E_y, "z" -> E_z

N_PROCESSES = int(os.environ.get("GF_NPROC", os.environ.get("SLURM_CPUS_PER_TASK", 10)))

_, _, SZ = spin_matrix_small()
PAULI = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


# ---------------------------------------------
#               SYSTEM BUILDERS
# ---------------------------------------------
def make_system(B):
    """Orbital Hamiltonian for B || z (A = (-B y, 0, 0)), same gauge as before."""
    a = L / (Ny + 1)
    y_center = (Ny - 1) / 2.0

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    def onsite(site):
        y, z = site.tag
        return np.asarray(gs_shifted_bz(a, kx, 0, 0, B, y - y_center), dtype=complex)

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        ymid = 0.5 * (y1 + y2) - y_center
        return np.asarray(
            gs_shifted_bz(a, kx, int(y1 - y2), int(z1 - z2), B, ymid), dtype=complex
        )

    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite
    for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping
    return syst.finalized()


def site_tags(fsyst):
    return np.array([s.tag for s in fsyst.sites], dtype=float)


def potential_diagonal(Ef, direction, tags):
    """Electrostatic potential on the diagonal, in the finalized site order."""
    a = L / (Ny + 1)
    coord = tags[:, 0] if direction == "y" else tags[:, 1]
    V = -Ef * ((coord + 1) * a - L / 2)
    return np.repeat(V, Nband)


# ---------------------------------------------
#               WORKER STATE
# ---------------------------------------------
H0_WORKER = None
TAGS_WORKER = None
DHZ_WORKER = None      # dH/dB_z (orbital part), eV/T


def initialize_worker():
    global H0_WORKER, TAGS_WORKER, DHZ_WORKER
    f0, fp, fm = make_system(0.0), make_system(+Bf), make_system(-Bf)
    TAGS_WORKER = site_tags(f0)
    assert np.array_equal(site_tags(fp), TAGS_WORKER)
    assert np.array_equal(site_tags(fm), TAGS_WORKER)
    H0_WORKER = f0.hamiltonian_submatrix(sparse=True).tocsr()
    Hp = fp.hamiltonian_submatrix(sparse=True)
    Hm = fm.hamiltonian_submatrix(sparse=True)
    DHZ_WORKER = ((Hp - Hm) / (2 * Bf)).tocsr()


def calculate(args):
    """Kramers pair P and projected Zeeman matrix M_z at one field."""
    Ef, direction = args
    t0 = time.time()

    H = H0_WORKER.copy()
    H.setdiag(H.diagonal() + potential_diagonal(Ef, direction, TAGS_WORKER))

    w, v = eigsh(H, k=N_EIG, sigma=sigma_val, which="LM")
    order = np.argsort(np.abs(w - sigma_val))        # same selection as before
    pair, rest = order[:2], order[2:]
    E_pair = w[pair]
    gap_next = np.min(np.abs(w[rest][:, None] - E_pair[None, :]))

    # eigsh (ARPACK, complex Hermitian) does NOT return an orthonormal basis
    # inside a degenerate Kramers pair -> orthonormalize the pair explicitly.
    P, _ = np.linalg.qr(v[:, pair])
    DP = np.column_stack([
        DHZ_WORKER @ P[:, j] + gspin * muB * apply_spin_operator(P[:, j], SZ)
        for j in range(2)
    ])
    M = P.conj().T @ DP
    M = 0.5 * (M + M.conj().T)                       # 2x2, eV/T

    return Ef, P, M, E_pair.mean(), gap_next, time.time() - t0


def pauli_vector(M):
    return np.array([np.trace(M @ s).real / muB for s in PAULI])


# ---------------------------------------------
#               MAIN
# ---------------------------------------------
def run(direction):
    with Pool(processes=N_PROCESSES, initializer=initialize_worker) as pool:
        jobs = [(Ef, direction) for Ef in Ef_list]
        res = []
        for out in tqdm(pool.imap_unordered(calculate, jobs, chunksize=1),
                        total=len(jobs), desc=f"E_{direction}", unit="field"):
            Ef, _, M, _, gap_next, dt = out
            tqdm.write(f"E_{direction}={Ef*1e4:6.3f} V/um  |g_z|={np.linalg.norm(pauli_vector(M)):7.4f}"
                       f"  gap_next={gap_next*1e3:7.3f} meV  ({dt:.0f} s)")
            res.append(out)
    res.sort(key=lambda r: r[0])

    # smooth pseudospin gauge + sign tracking
    rows = []
    P_prev, g_prev, sign = None, None, 1.0
    for Ef, P, M, E_pair, gap_next, _ in res:
        overlap_min = 1.0
        if P_prev is not None:
            O = P_prev.conj().T @ P                  # 2x2
            W, S, Vh = np.linalg.svd(O)
            overlap_min = S.min()
            X = Vh.conj().T @ W.conj().T             # unitary closest to O^dag
            P = P @ X
            M = X.conj().T @ M @ X
        g = pauli_vector(M)
        if g_prev is not None and np.dot(g, g_prev) < 0:
            sign = -sign
        rows.append([Ef * 1e4, sign * np.linalg.norm(g), np.linalg.norm(g),
                     overlap_min, E_pair, gap_next])
        P_prev, g_prev = P, g

    data = np.array(rows)
    g0 = data[0, 1]
    dg_percent = (data[:, 1] - g0) / abs(g0) * 100
    data = np.column_stack([data[:, 0], data[:, 1], dg_percent, data[:, 2:]])

    out = f"gzz_vs_E{direction}_N{N}_L{int(L)}.dat"
    np.savetxt(out, data, header="E(V/um)  g_zz  dg_zz(%)  |g_z|  overlap_min  E_pair(eV)  gap_next(eV)")
    print(f"Saved {out}")

    for i in np.where(np.diff(np.sign(data[:, 1])) != 0)[0]:
        print(f"  g_zz changes sign between E_{direction} = {data[i,0]:.3f} and {data[i+1,0]:.3f} V/um")
    if data[1:, 4].min() < 0.7:
        print(f"  WARNING: overlap between neighbouring fields drops to {data[1:,4].min():.2f};"
              f" the sign tracking may be unreliable -> increase N_E.")
    return data


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    style = {"y": dict(marker="o", label=r"$E_y$"), "z": dict(marker="s", label=r"$E_z$")}
    for d, data in results.items():
        axs[0].plot(data[:, 0], data[:, 1], **style[d])
        axs[1].plot(data[:, 0], data[:, 2], **style[d])
    for ax in axs:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"$\mathcal{E}$ (V/$\mu$m)")
        ax.legend()
    axs[0].set_ylabel(r"$g_{zz}$")
    axs[1].set_ylabel(r"$\Delta g_{zz}/|g_{zz}(0)|$ (%)")
    fig.tight_layout()
    out = f"gzz_vs_E_N{N}_L{int(L)}.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    results = {d: run(d) for d in FIELD_DIRECTIONS}
    plot(results)