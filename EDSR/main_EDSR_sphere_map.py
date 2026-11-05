"""
EDSR Rabi frequency over the full sphere of magnetic-field directions.

    B_hat = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))

For every direction the field magnitude is adjusted so that the qubit splitting
stays at target_fZ_GHz (same fixed-Zeeman loop as
main_EDSR_fixed_Zeeman_angle_sweep.py), and the Rabi frequency
f_R = |<1| e E_ac x |0>| / h is computed.

Your three plane sweeps are cuts through this map:
    xy plane : theta = 90 deg      (angle = phi)
    xz plane : phi = 0 / 180 deg   (angle = theta)
    yz plane : phi = 90 / 270 deg  (angle = theta)

Speed-ups
---------
* Inversion symmetry: B -> -B leaves |<1|x|0>| unchanged, so
  f_R(theta, phi + 180) = f_R(180 - theta, phi). Only phi in [0, 180] is computed.
* The theta grid is refined near the equator (theta = 90 deg), where the
  strongly anisotropic g tensor squeezes all structure into a narrow band.
* Grid points run in parallel (multiprocessing.Pool).

Output: EDSR_sphere_map_fz_{fZ}_Eth{theta_E:.2f}dc_{E}.npz
"""
import os

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm


# ---------------------------------
# Parameters (same physics as main_EDSR_fixed_Zeeman_angle_sweep.py)
# ---------------------------------
N             = 100
L             = 300
muB           = 5.788e-5        # eV/T
h_eVs         = 4.135667696e-15 # eV s
target_fZ_GHz = 3               # fixed Zeeman splitting for every field direction
max_B_iterations = 8
fZ_tolerance_GHz = 1e-6
B_max_T       = 10.0            # points needing a larger field are stored as NaN
E_dc_Vum      = 6.0
# dc-field direction: angle theta_E (deg) in the y-z plane, measured from +y toward +z
#   0 -> E_y^dc, 90 -> E_z^dc; one map is computed per entry
E_dc_thetas   = (0.0, 90.0, 83.2)
E_dc_theta    = E_dc_thetas[0]  # set by the loop in __main__
DATA_DIR      = "import_data"
R_INCLUDES_FIELD = True         # see main_EDSR_fixed_Zeeman_angle_sweep.py
E_ac_Vum      = 1e-3
f_orb_GHz     = 100.0
box_factor    = 8.0
Nx_spin       = 401
num_eigs      = 6               # only the top pair is needed

# ---------------------------------
# Angular grid
# ---------------------------------
dtheta_coarse = 2.0             # deg, whole sphere
theta_fine_halfwidth = 0.0     # deg, refined band 90 +- this
dtheta_fine   = 0.25            # deg, inside the band
dphi          = 2.0             # deg
USE_INVERSION_SYMMETRY = True

N_PROCESSES   = 10


# ==================================================
# Model (built once per worker)
# ==================================================
def build_model():
    def load(name):
        return np.load(os.path.join(DATA_DIR, name))

    tag_M = f"Eth{E_dc_theta:.2f}_{E_dc_Vum:.3f}Vum_N{N}_L{L}"      # Mx_Eth83.20_6.000Vum_N100_L300.npy
    tag_R = f"th{E_dc_theta:.2f}_E{E_dc_Vum:.2f}_N{N}_L{L}"        # R_mat_th83.20_E6.00_N100_L300.npy
    M = np.array([load(f"M{a}_{tag_M}.npy") for a in "xyz"])          # (3,2,2) eV/T  g_factor matrix
    R_dc = load(f"R_mat_{tag_R}.npy")            # Rashba matrix
    A_hole = float(load("A_hole_data.npy"))
    Ev = float(load("band_edge.npy"))

    hbar_omega_eV = h_eVs * f_orb_GHz * 1e9
    A_abs = abs(A_hole)
    l0_A = np.sqrt(2.0 * A_abs / hbar_omega_eV)
    Lbox = 2.0 * box_factor * l0_A
    dx = Lbox / (Nx_spin + 1)
    x = (np.arange(Nx_spin) - Nx_spin // 2) * dx

    D2 = diags([np.ones(Nx_spin - 1), -2.0 * np.ones(Nx_spin), np.ones(Nx_spin - 1)],
               offsets=[-1, 0, 1], format="csr") / dx**2
    kappa = hbar_omega_eV**2 / (4.0 * A_abs)
    H_orb = Ev * eye(Nx_spin, format="csr") + A_hole * D2 + diags(-kappa * x**2, 0, format="csr")
    H_dot = kron(H_orb, eye(2, format="csr"), format="csr")

    D1 = diags([-np.ones(Nx_spin - 1), np.ones(Nx_spin - 1)], offsets=[-1, 1],
               format="csr") / (2.0 * dx)
    R_scale = 1.0 if R_INCLUDES_FIELD else E_dc_Vum * 1e-4
    H_SO = R_scale * kron(-1j * D1, csr_matrix(R_dc), format="csr")

    E_ac_VA = E_ac_Vum * 1e-4
    D_drive = E_ac_VA * kron(diags(x, 0, format="csr"), eye(2, format="csr"), format="csr")

    alpha_R = R_scale * 0.5 * np.ptp(np.linalg.eigvalsh(R_dc))        # eV A
    l_so_A = 2.0 * A_abs / alpha_R if alpha_R > 0 else np.inf

    return dict(M=M, R=R_scale * R_dc, H0=(H_dot + H_SO).tocsr(), D=D_drive.tocsr(),
                E_ref=Ev, hbar_omega_eV=hbar_omega_eV, l0_A=l0_A, l_so_A=l_so_A,
                E_ac_VA=E_ac_VA)


MODEL = None


def init_worker():
    global MODEL
    MODEL = build_model()


def unit_vector(theta_deg, phi_deg):
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def compute_point(args):
    """Fixed-fZ EDSR at one field direction -> (fR_MHz, B_T, g_eff, fZ_GHz, converged)."""
    theta_deg, phi_deg = args
    m = MODEL
    n = unit_vector(theta_deg, phi_deg)
    M_dir = np.tensordot(n, m["M"], axes=1)
    M_dir = 0.5 * (M_dir + M_dir.conj().T)
    slope = np.ptp(np.linalg.eigvalsh(M_dir))                      # eV/T
    target_EZ = h_eVs * target_fZ_GHz * 1e9
    nan = (np.nan, np.nan, np.nan, np.nan, False)
    if slope < 1e-15:
        return nan
    B = target_EZ / slope
    if B > B_max_T:
        return nan

    n_sites = m["H0"].shape[0] // 2
    converged = False
    for it in range(max_B_iterations):
        H = m["H0"] + kron(eye(n_sites, format="csr"), csr_matrix(B * M_dir), format="csr")
        E, psi = eigsh(H, k=num_eigs, sigma=m["E_ref"], which="LM")
        idx = np.argsort(E)[::-1]
        E, psi = E[idx], psi[:, idx]
        fZ = abs(E[0] - E[1]) / h_eVs / 1e9
        if abs(fZ - target_fZ_GHz) <= fZ_tolerance_GHz:
            converged = True
            break
        B_new = B * target_fZ_GHz / fZ
        if not np.isfinite(B_new) or B_new <= 0 or B_new > B_max_T:
            return nan
        if it < max_B_iterations - 1:
            B = B_new

    M_EDSR = abs(psi[:, 1].conj() @ (m["D"] @ psi[:, 0]))
    fR_MHz = M_EDSR / h_eVs / 1e6
    g_eff = abs(E[0] - E[1]) / (muB * B)
    return fR_MHz, B, g_eff, fZ, converged


# ==================================================
# Main
# ==================================================
def main():
    t0 = time.time()
    coarse = np.arange(0.0, 180.0 + 1e-9, dtheta_coarse)
    fine = np.arange(90.0 - theta_fine_halfwidth, 90.0 + theta_fine_halfwidth + 1e-9, dtheta_fine)
    theta = np.unique(np.round(np.concatenate([coarse, fine]), 8))   # symmetric about 90
    phi_full = np.arange(0.0, 360.0 + 1e-9, dphi)
    phi_calc = phi_full[phi_full <= 180.0 + 1e-9] if USE_INVERSION_SYMMETRY else phi_full

    jobs = [(t, p) for t in theta for p in phi_calc]
    print(f"E^dc = {E_dc_Vum} V/um at theta_E = {E_dc_theta:.2f} deg, f_Z = {target_fZ_GHz} GHz: "
          f"{len(theta)} theta x {len(phi_calc)} phi = {len(jobs)} points, {N_PROCESSES} processes")

    with Pool(N_PROCESSES, initializer=init_worker) as pool:
        res = list(tqdm(pool.imap(compute_point, jobs, chunksize=4), total=len(jobs), unit="pt"))
    res = np.array(res, dtype=float).reshape(len(theta), len(phi_calc), 5)

    # fill phi in (180, 360] from f(theta, phi+180) = f(180-theta, phi)
    if USE_INVERSION_SYMMETRY:
        out = np.full((len(theta), len(phi_full), 5), np.nan)
        out[:, :len(phi_calc)] = res
        for j, p in enumerate(phi_full):
            if p > 180.0 + 1e-9:
                jj = int(np.argmin(np.abs(phi_calc - (p - 180.0))))
                out[:, j] = res[::-1, jj]                           # theta -> 180 - theta
        res = out

    model = build_model()



# ---------- exporting output as .npz and .dat ----------
    fname = f"EDSR_sphere_map_fz_{target_fZ_GHz:g}_Eth{E_dc_theta:.2f}dc_{E_dc_Vum:g}.npz"
    np.savez(fname,
             theta_deg=theta, phi_deg=phi_full,
             fR_MHz=res[..., 0], B_T=res[..., 1], g_eff=res[..., 2], fZ_GHz=res[..., 3],
             converged=res[..., 4].astype(bool),
             E_dc_Vum=E_dc_Vum, E_dc_theta_deg=E_dc_theta, target_fZ_GHz=target_fZ_GHz,
             E_ac_Vum=E_ac_Vum, f_orb_GHz=f_orb_GHz, L=L, N=N, B_max_T=B_max_T,
             l0_nm=model["l0_A"] / 10, l_so_nm=model["l_so_A"] / 10)

    
    # one row per (theta, phi); theta is the outer loop, phi the inner loop.
    # A blank line separates theta blocks (gnuplot "splot ... with pm3d" reads this directly).
    TH, PH = np.meshgrid(theta, phi_full, indexing="ij")
    table = np.column_stack([
        TH.ravel(), PH.ravel(),
        res[..., 0].ravel(), res[..., 1].ravel(), res[..., 2].ravel(),
        res[..., 3].ravel(), res[..., 4].ravel(),
    ])
    dat_name = fname.replace(".npz", ".dat")
    header = (
        f"EDSR sphere map: E^dc = {E_dc_Vum} V/um at theta_E = {E_dc_theta:.2f} deg (from y), f_Z = {target_fZ_GHz} GHz, "
        f"E_ac = {E_ac_Vum} V/um, f_orb = {f_orb_GHz} GHz, N = {N}, L = {L}, "
        f"l0 = {model['l0_A']/10:.4f} nm, l_so = {model['l_so_A']/10:.4f} nm, B_max = {B_max_T} T\n"
        f"grid: {len(theta)} theta x {len(phi_full)} phi (NaN = needs B > B_max)\n"
        "theta_deg  phi_deg  fR_MHz  B_T  g_eff  fZ_GHz  converged"
    )
    n_phi = len(phi_full)
    with open(dat_name, "w") as fh:
        fh.write("".join(f"# {line}\n" for line in header.splitlines()))
        for i in range(len(theta)):
            np.savetxt(fh, table[i * n_phi:(i + 1) * n_phi], fmt="%.10e")
            fh.write("\n")
    print(f"Saved {dat_name}")

    fR = res[..., 0]
    n_bad = int(np.isnan(fR).sum())
    print(f"f_R range: {np.nanmin(fR):.4g} - {np.nanmax(fR):.4g} MHz")
    if n_bad:
        print(f"{n_bad} points need B > {B_max_T} T or failed; stored as NaN")
    if not np.all(res[..., 4][~np.isnan(fR)]):
        print("WARNING: some points did not reach the f_Z tolerance")
    print(f"Saved {fname}   ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    for E_dc_theta in E_dc_thetas:  # workers are forked after this is set, so they see the current direction
        main()