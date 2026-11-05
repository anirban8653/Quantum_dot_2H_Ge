"""
Signed g_xx of the top valence Kramers pair of a 2H-Ge nanowire versus an
electric field along y or z (B || x, along the wire, only).

Method
------
1. For every field E, the Kramers pair P = (|1>, |2>) is computed at B = 0 and the
   first-order Zeeman operator for B || x (orbital/Peierls + spin) is projected:
       M_x = P^dag (dH/dB_x) P = (muB/2) sum_beta g_{x beta} sigma_beta
       g_{x beta} = Tr(M_x sigma_beta) / muB,      |g_x| = splitting / (muB B)
   The orbital part is a central difference (H(+B) - H(-B)) / 2B, with B_x
   entering through Peierls phases on the y/z hoppings (A = (0, 0, B_x y)).
   The B = 0 Hamiltonian is built exactly as in the g_yy / g_zz scripts, so
   all three use the same Kramers pair.

2. Sign of g_xx without any other field direction:
   The pseudospin basis is carried smoothly from one field to the next
   (P_E -> P_E X, with X the unitary closest to P_prev^dag P_E). In that smooth
   basis the vector g_x(E) changes continuously; if g_xx passes through zero
   it reverses direction, i.e. g_x(E_i) . g_x(E_{i-1}) < 0. The sign is
   accumulated from these flips and referenced to E = 0.
   Check: `overlap_min` (smallest singular value of P_prev^dag P_E) should stay
   close to 1. If it drops (say < 0.7) the field step is too large or another
   level crosses the pair -> use more field points.

Output
------
gxx_vs_E{y,z}_N{N}_L{L}.dat : E, g_xx (signed), Delta g_xx (%), |g_x|,
                              overlap_min, E_pair, gap_next
gxx_vs_E_N{N}_L{L}.png      : g_xx vs E and percentage change vs E
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
from Hamiltonian_mathematica_v2 import gso, spin_matrix_small, apply_spin_operator
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

SX, _, _ = spin_matrix_small()
hbar = 1.05457e-34
q = 1.602e-19
PAULI = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


# ---------------------------------------------
#               SYSTEM BUILDERS
# ---------------------------------------------
def make_system(B):
    """Orbital Hamiltonian for B || x: A = (0, 0, B y), Peierls phases on hoppings.
    Hopping blocks gso(y1 - y2, z1 - z2) as in the g_yy / g_zz scripts."""
    a = L / (Ny + 1)
    y_center = (Ny - 1) / 2.0
    phase_prefactor = q * B * (a * 1e-10) ** 2 / hbar   # q B a^2 / hbar

    blocks = {(dy, dz): np.asarray(gso(a, kx, dy, dz), dtype=complex)
              for dy in (-1, 0, 1) for dz in (-1, 0, 1)}

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    def onsite(site):
        return blocks[(0, 0)]

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        ymid = 0.5 * (y1 + y2) - y_center
        # Peierls phase for A_z = B_x y (same form as the original Bx script)
        phase = np.exp(1j * phase_prefactor * ymid * (z1 - z2))
        return blocks[(int(y1 - y2), int(z1 - z2))] * phase

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
DHX_WORKER = None      # dH/dB_x (orbital part), eV/T


def initialize_worker():
    global H0_WORKER, TAGS_WORKER, DHX_WORKER
    f0, fp, fm = make_system(0.0), make_system(+Bf), make_system(-Bf)
    TAGS_WORKER = site_tags(f0)
    assert np.array_equal(site_tags(fp), TAGS_WORKER)
    assert np.array_equal(site_tags(fm), TAGS_WORKER)
    H0_WORKER = f0.hamiltonian_submatrix(sparse=True).tocsr()
    Hp = fp.hamiltonian_submatrix(sparse=True)
    Hm = fm.hamiltonian_submatrix(sparse=True)
    DHX_WORKER = ((Hp - Hm) / (2 * Bf)).tocsr()


def calculate(args):
    """Kramers pair P and projected Zeeman matrix M_x at one field."""
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
        DHX_WORKER @ P[:, j] + gspin * muB * apply_spin_operator(P[:, j], SX)
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
            tqdm.write(f"E_{direction}={Ef*1e4:6.3f} V/um  |g_x|={np.linalg.norm(pauli_vector(M)):7.4f}"
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

    out = f"gxx_vs_E{direction}_N{N}_L{int(L)}.dat"
    np.savetxt(out, data, header="E(V/um)  g_xx  dg_xx(%)  |g_x|  overlap_min  E_pair(eV)  gap_next(eV)")
    print(f"Saved {out}")

    for i in np.where(np.diff(np.sign(data[:, 1])) != 0)[0]:
        print(f"  g_xx changes sign between E_{direction} = {data[i,0]:.3f} and {data[i+1,0]:.3f} V/um")
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
    axs[0].set_ylabel(r"$g_{xx}$")
    axs[1].set_ylabel(r"$\Delta g_{xx}/|g_{xx}(0)|$ (%)")
    fig.tight_layout()
    out = f"gxx_vs_E_N{N}_L{int(L)}.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    results = {d: run(d) for d in FIELD_DIRECTIONS}
    plot(results)