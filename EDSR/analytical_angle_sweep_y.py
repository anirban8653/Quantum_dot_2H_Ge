import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# Analytical EDSR angle sweep at fixed Zeeman splitting
# ==================================================


# ==================================================
# Parameters
# ==================================================

L = 300

hbar   = 1.05457e-34          # J s
e_mass = 9.11e-31             # kg
e0     = 1.602e-19
dim    = 1.602e-39
gspin  = 2.0

muB   = 5.78e-5               # eV/T
h_eVs = 4.135667696e-15       # eV s

# Target Zeeman/Larmor frequency
fZ_target_GHz = 2.0           # GHz

# Static electric field along z
E_y_dc_Vum = 0.05             # V/um

# AC electric field 
E_ac_Vum = 1.0e-3           # V/um

# Orbital separation
f_orb_GHz = 100.0             # GHz

# Angular resolution
n_angles = 61


# ==================================================
# Load projected matrices
# ==================================================

M_x = np.load("Mx_matrix.npy")
M_y = np.load("My_matrix.npy")
M_z = np.load("Mz_matrix.npy")

R_y = np.load("Ry_matrix.npy")

print("Loaded Mx, My, Mz, and Rz")

print("Mx:\n", M_x.round(6))
print("My:\n", M_y.round(6))
print("Mz:\n", M_z.round(6))
print("Ry:\n", R_y.round(6))


# ==================================================
# Helper function
# ==================================================

def hermitize(A):
    """Return the Hermitian part of a matrix."""
    return 0.5 * (A + A.conj().T)


M_x = hermitize(M_x)
M_y = hermitize(M_y)
M_z = hermitize(M_z)
R_y = hermitize(R_y)


# ==================================================
# Unit conversions and fixed energy scales
# ==================================================

# 1 V/um = 1e-4 V/Angstrom
E_y_dc_VA = E_y_dc_Vum * 1.0e-4
E_ac_VA = E_ac_Vum * 1.0e-4

# Orbital gap: Delta = h f_orb
Delta_eV = h_eVs * f_orb_GHz * 1.0e9

# Fixed Zeeman energy: E_Z = h f_Z
E_Z_target_eV = h_eVs * fZ_target_GHz * 1.0e9

if E_Z_target_eV >= Delta_eV:
    raise ValueError(
        "The fixed Zeeman energy must be smaller than the orbital gap."
    )

print("\nFixed Zeeman condition:")
print(f"fZ target       = {fZ_target_GHz:.6f} GHz")
print(f"E_Z target      = {E_Z_target_eV:.6e} eV")
print(f"E_Z target      = {E_Z_target_eV * 1e6:.6f} ueV")
print(f"Orbital gap     = {Delta_eV * 1e3:.6f} meV")


# ==================================================
# Angle array
# ==================================================

angle_deg = np.linspace(0.0, 360.0, n_angles)
angle_rad = np.deg2rad(angle_deg)


# ==================================================
# Fixed-Zeeman calculation for one field direction
# ==================================================

def calculate_fixed_zeeman_quantities(nx, ny, nz):
    """
    Calculate the analytical EDSR quantities for one magnetic-field
    direction while keeping the Zeeman splitting fixed.

    Returns
    -------
    g_eff : float
        Direction-dependent effective g factor.
    B_required_T : float
        Magnetic-field magnitude required to obtain fZ_target_GHz.
    f_Rabi_Hz : float
        Analytical Rabi frequency at the fixed Zeeman splitting.
    """

    # Normalize the magnetic-field direction
    norm = np.sqrt(nx**2 + ny**2 + nz**2)

    if norm < 1.0e-14:
        raise ValueError("Magnetic-field direction has zero norm.")

    nx /= norm
    ny /= norm
    nz /= norm

    # Direction-dependent Zeeman matrix per tesla
    M_direction = nx * M_x + ny * M_y + nz * M_z
    M_direction = hermitize(M_direction)

    # Zeeman eigenvalues and eigenvectors
    eigenvalues, U_B = np.linalg.eigh(M_direction)

    index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[index]
    U_B = U_B[:, index]

    # Splitting per tesla, in eV/T
    zeeman_slope_eV_per_T = abs(
        eigenvalues[1] - eigenvalues[0]
    )

    if zeeman_slope_eV_per_T < 1.0e-15:
        return np.nan, np.nan, np.nan

    # Effective g factor
    g_eff = zeeman_slope_eV_per_T / muB

    # Required B field for the fixed target splitting
    B_required_T = E_Z_target_eV / zeeman_slope_eV_per_T

    # Transform Ry into the local Zeeman eigenbasis
    R_y_Bbasis = U_B.conj().T @ R_y @ U_B

    # Transverse Rashba matrix element
    alpha_prime_z = abs(R_y_Bbasis[0, 1])

    # Static-field-induced Rashba coefficient
    alpha_z_eVA = alpha_prime_z * E_y_dc_VA

    # Exact analytical EDSR matrix element.
    # E_Z is fixed, so its value is the same at every angle.
    denominator = Delta_eV**2 - E_Z_target_eV**2

    M_EDSR_eV = (
        E_ac_VA
        * alpha_z_eVA
        * E_Z_target_eV
        / denominator
    )

    # Rabi frequency
    f_Rabi_Hz = abs(M_EDSR_eV) / h_eVs

    return g_eff, B_required_T, f_Rabi_Hz


# ==================================================
# Allocate angular-sweep arrays
# ==================================================

g_xy = np.zeros(n_angles)
B_xy_T = np.zeros(n_angles)
fR_xy_Hz = np.zeros(n_angles)

g_xz = np.zeros(n_angles)
B_xz_T = np.zeros(n_angles)
fR_xz_Hz = np.zeros(n_angles)

g_yz = np.zeros(n_angles)
B_yz_T = np.zeros(n_angles)
fR_yz_Hz = np.zeros(n_angles)


# ==================================================
# Magnetic-field sweeps
# ==================================================

for i, angle in enumerate(angle_rad):

    # --------------------------------------------------
    # xy plane
    # phi = 0   -> +x
    # phi = 90  -> +y
    # phi = 180 -> -x
    # --------------------------------------------------

    nx_xy = np.cos(angle)
    ny_xy = np.sin(angle)
    nz_xy = 0.0

    g_xy[i], B_xy_T[i], fR_xy_Hz[i] = \
        calculate_fixed_zeeman_quantities(
            nx_xy,
            ny_xy,
            nz_xy,
        )

    # --------------------------------------------------
    # xz plane
    # theta = 0   -> +z
    # theta = 90  -> +x
    # theta = 180 -> -z
    # --------------------------------------------------

    nx_xz = np.sin(angle)
    ny_xz = 0.0
    nz_xz = np.cos(angle)

    g_xz[i], B_xz_T[i], fR_xz_Hz[i] = \
        calculate_fixed_zeeman_quantities(
            nx_xz,
            ny_xz,
            nz_xz,
        )

    # --------------------------------------------------
    # yz plane
    # theta = 0   -> +z
    # theta = 90  -> +y
    # theta = 180 -> -z
    # --------------------------------------------------

    nx_yz = 0.0
    ny_yz = np.sin(angle)
    nz_yz = np.cos(angle)

    g_yz[i], B_yz_T[i], fR_yz_Hz[i] = \
        calculate_fixed_zeeman_quantities(
            nx_yz,
            ny_yz,
            nz_yz,
        )


# Convert Hz to MHz for plotting and saving
fR_xy_MHz = fR_xy_Hz / 1.0e6
fR_xz_MHz = fR_xz_Hz / 1.0e6
fR_yz_MHz = fR_yz_Hz / 1.0e6


# ==================================================
# Save data
#
# Columns:
# angle,
# g_xy, B_xy, fR_xy,
# g_xz, B_xz, fR_xz,
# g_yz, B_yz, fR_yz
# ==================================================

output_data = np.column_stack([
    angle_deg,

    g_xy,
    B_xy_T,
    fR_xy_MHz,

    g_xz,
    B_xz_T,
    fR_xz_MHz,

    g_yz,
    B_yz_T,
    fR_yz_MHz,
])

output_filename = "analytical_Rabi_fixed_Zeeman_angle_sweep_y.dat"

np.savetxt(
    output_filename,
    output_data,
    fmt="%.10e",
    header=(
        "angle_deg  "
        "g_xy  B_xy_T  fR_xy_MHz  "
        "g_xz  B_xz_T  fR_xz_MHz  "
        "g_yz  B_yz_T  fR_yz_MHz"
    ),
)

print(f"\nSaved data: {output_filename}")


# ==================================================
# Print selected high-symmetry directions
# ==================================================

for index, label in [
    (0, "0 deg"),
    (n_angles // 2, "90 deg"),
    (-1, "180 deg"),
]:
    print(f"\nAngle = {label}")
    print(
        f"xy: g = {g_xy[index]:.6f}, "
        f"B = {B_xy_T[index]:.6f} T, "
        f"fR = {fR_xy_MHz[index]:.6e} MHz"
    )
    print(
        f"xz: g = {g_xz[index]:.6f}, "
        f"B = {B_xz_T[index]:.6f} T, "
        f"fR = {fR_xz_MHz[index]:.6e} MHz"
    )
    print(
        f"yz: g = {g_yz[index]:.6f}, "
        f"B = {B_yz_T[index]:.6f} T, "
        f"fR = {fR_yz_MHz[index]:.6e} MHz"
    )


# ==================================================
# Plot settings
# ==================================================

plt.rcParams.update({
    "figure.figsize": [14, 4],
    "font.family": "Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "font.size": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# ==================================================
# Plot: 1 x 3 Rabi-frequency figure
# ==================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 4),
    sharey=True,
)


# xy plane
axes[0].scatter(
    angle_deg,
    fR_xy_MHz,
    linewidth=2,
)

axes[0].set_title(r"$xy$ plane")
axes[0].set_xlabel(r"$\phi$ (degree)")
axes[0].set_ylabel(r"$f_{\mathrm{Rabi}}$ (MHz)")


# xz plane
axes[1].scatter(
    angle_deg,
    fR_xz_MHz,
    linewidth=2,
)

axes[1].set_title(r"$xz$ plane")
axes[1].set_xlabel(r"$\theta$ (degree)")


# yz plane
axes[2].scatter(
    angle_deg,
    fR_yz_MHz,
    linewidth=2,
)

axes[2].set_title(r"$yz$ plane")
axes[2].set_xlabel(r"$\theta$ (degree)")


for ax in axes:
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])


fig.suptitle(
    fr"Fixed Zeeman splitting: $f_Z={fZ_target_GHz:.1f}$ GHz",
    fontsize=17,
)

fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

fig.savefig(
    "analytical_Rabi_fixed_Zeeman_angle_sweep_y.pdf",
    bbox_inches="tight",
)

fig.savefig(
    "analytical_Rabi_fixed_Zeeman_angle_sweep_y.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
