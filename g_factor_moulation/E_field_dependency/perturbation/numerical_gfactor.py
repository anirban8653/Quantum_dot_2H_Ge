"""
Numerical (exact-diagonalization) signed g factor versus electric field.

For every field E:
  1. H(E) = H0 + E * V_E is diagonalized near the band edge (shift-invert).
  2. The Kramers pair P_E is the pair of eigenvectors with the largest overlap
     with the E = 0 pseudospin basis P0 = (|up>, |down>) used by pymablock.
  3. P_E is rotated onto P0 with the unitary closest to P_E^dag P0 (polar
     decomposition), i.e. the same minimal rotation as the Schrieffer-Wolff
     basis of the perturbative effective Hamiltonian.
  4. M = P_E^dag (dH/dB) P_E  (orbital + spin, eV/T) gives
         g_beta = Tr(M sigma_beta) / muB,   |g| = splitting / (muB B).
     The saved g is sign(g_alpha alpha) * |g|, as in the perturbative scripts.

`overlap_min` (smallest singular value of P0^dag P_E) should stay close to 1;
a warning is printed if it drops below 0.7.
"""
import multiprocessing as mp

import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import eigsh

from Hamiltonian_mathematica_v2 import g_vector

COMPONENT_INDEX = {"xx": 0, "yy": 1, "zz": 2}

_W = {}   # worker state, inherited through fork


def _point(Ef):
    H = (_W["H0"] + Ef * _W["V_E"]).tocsc()
    w, v = eigsh(H, k=_W["n_eig"], sigma=_W["sigma"], which="LM")

    # Kramers pair = the two eigenvectors with the largest weight on P0
    weight = np.sum(np.abs(_W["P0"].conj().T @ v) ** 2, axis=0)
    order = np.argsort(weight)[::-1]
    pair, rest = order[:2], order[2:]
    E_pair = w[pair].mean()
    gap_next = np.min(np.abs(w[rest] - E_pair)) if len(rest) else np.nan

    P, _ = qr(v[:, pair], mode="economic")   # orthonormalize inside the pair

    # align with the E = 0 pseudospin basis
    O = _W["P0"].conj().T @ P                # 2x2
    W, S, Vh = np.linalg.svd(O)
    X = Vh.conj().T @ W.conj().T             # unitary closest to O^dag
    P = P @ X

    M = P.conj().T @ (_W["V_B"] @ P)
    M = 0.5 * (M + M.conj().T)
    gvec = g_vector(M, _W["muB"])
    return Ef, gvec, S.min(), E_pair, gap_next


def numerical_g_sweep(H0, V_E, V_B, P0, efield_list, component, muB, B,
                      n_workers=8, sigma=0.0, n_eig=6):
    """
    H0, V_E, V_B : sparse (V_B = dH/dB incl. spin Zeeman, eV/T)
    P0           : (dim, 2) E = 0 pseudospin basis, columns (|up>, |down>)
    efield_list  : fields in V/Angstrom
    Returns rows [Ef(V/A), g_signed, gap(eV), g_a1, g_a2, g_a3, overlap_min,
                  E_pair(eV), gap_next(eV)].
    """
    _W.update(H0=H0.tocsr(), V_E=V_E.tocsr(), V_B=V_B.tocsr(), P0=P0,
              muB=muB, sigma=sigma, n_eig=n_eig)
    k = COMPONENT_INDEX[component]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(n_workers, len(efield_list))) as pool:
        out = pool.map(_point, list(efield_list), chunksize=1)

    rows = []
    for Ef, gvec, overlap, E_pair, gap_next in sorted(out, key=lambda r: r[0]):
        gnorm = np.linalg.norm(gvec)
        g_signed = np.sign(gvec[k]) * gnorm
        rows.append([Ef, g_signed, gnorm * muB * B, *gvec, overlap, E_pair, gap_next])
        print(f"  [numerical] E = {Ef*1e4:.4f} V/um, g_{component} = {g_signed:.5f}, "
              f"g_vec = ({gvec[0]:.4f}, {gvec[1]:.4f}, {gvec[2]:.4f}), "
              f"overlap = {overlap:.4f}, gap_next = {gap_next*1e3:.3f} meV")
    rows = np.asarray(rows)
    if rows[:, 6].min() < 0.7:
        print(f"  WARNING: overlap with the E = 0 pair drops to {rows[:, 6].min():.2f}; "
              f"the pair may have changed character -> sign/assignment unreliable.")
    return rows


def save_numerical(rows, component, direction, N, L):
    """
    Save in the layout read by plot_gfactor_E.ipynb:
      xx      : size(nm)  E(V/um)  gap(eV)  g_xx
      yy / zz : size(nm)  E(V/um)  g        gap(eV)
    g is signed. direction is "y" or "z".
    """
    size_nm = L / 10.0
    E_um = rows[:, 0] * 1e4
    g, gap = rows[:, 1], rows[:, 2]
    size_col = np.full_like(E_um, size_nm)
    if component == "xx":
        data = np.column_stack([size_col, E_um, gap, g])
        header = f"size(nm)  E{direction}(V/um)  gap(eV)  g_xx(signed)"
    else:
        data = np.column_stack([size_col, E_um, g, gap])
        header = f"size(nm)  E{direction}(V/um)  g_{component}(signed)  gap(eV)"
    fname = f"gfactor_{component}_vs_E{direction}_N{N}_L{int(L)}.dat"
    np.savetxt(fname, data, header=header)
    print(f"Saved numerical data to {fname}")
    return fname
