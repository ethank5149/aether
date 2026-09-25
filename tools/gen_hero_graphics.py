"""Generate hero graphics for the AETHER documentation.

Uses the actual Orion capsule geometry and real SU2 NEMO 5-species Euler
CFD solutions (M = 25, 65 km, α = 19.3°) for schlieren, plasma sheath,
and RF blackout visualisations — no analytical shortcuts.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

OUT = str(_ROOT / "docs" / "_static")

# ── Theme ───────────────────────────────────────────────────────────────
BG       = "#0B0E17"
BG_PANEL = "#0F1320"
TEXT     = "#E8ECF5"
TEXT_DIM = "#8892A8"
CYAN     = "#00D4FF"
ORANGE   = "#FF6B35"
PURPLE   = "#A855F7"
GREEN    = "#4ADE80"
PINK     = "#FF3366"
GOLD     = "#F8C050"
GRID     = "#222940"

DPI = 250

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG_PANEL,
    "axes.edgecolor": "#2A3050", "axes.labelcolor": TEXT,
    "axes.labelsize": 11, "text.color": TEXT,
    "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.alpha": 0.5,
    "legend.facecolor": "#141828", "legend.edgecolor": "#2A3050",
    "legend.fontsize": 9, "font.family": "sans-serif",
    "savefig.facecolor": BG, "savefig.dpi": DPI,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

glow = [patheffects.withStroke(linewidth=3, foreground=BG)]
glow2 = [patheffects.withStroke(linewidth=2, foreground=BG)]

_schl = [
    "#0B0E17", "#101828", "#182438", "#1E3248", "#264058",
    "#2E5070", "#386088", "#4472A0", "#5288B8", "#62A0D0",
    "#78B8E0", "#92D0EE", "#B0E4F8", "#D0F0FF", "#E8F8FF", "#FFFFFF",
]
cmap_schl = LinearSegmentedColormap.from_list("schl", _schl, N=512)

_heat = ["#0B0E17", "#1A0A2E", "#3D0A4A", "#6B0F3A", "#9C1528",
         "#CC3B1A", "#E86420", "#F09030", "#F8C050", "#FFEE88", "#FFF"]
cmap_heat = LinearSegmentedColormap.from_list("heat", _heat, N=256)

_plasma = ["#0B0E17", "#0D1040", "#1A1070", "#3010A0", "#6020C0",
           "#9030D0", "#C040D8", "#E060D0", "#FF80C0", "#FFB0E0", "#FFF"]
cmap_plasma = LinearSegmentedColormap.from_list("plasma_ne", _plasma, N=512)

# ── Precomputed CFD data (SU2 NEMO 5-species Euler, M = 25) ─────────
_CFD_SCHLIEREN = Path(OUT) / "cfd_schlieren_orion_m25.npz"
_CFD_PLASMA = Path(OUT) / "cfd_plasma_orion_m25.npz"


def _rotate(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def _orion_data():
    """Load Orion geometry, aero, and compute Billig shock."""
    from examples.artemis1.entry import orion_meridian, orion_aerodynamics, DIAMETER
    from aether.cfd.shock import billig_shock

    x_m, r_m = orion_meridian()
    aero = orion_aerodynamics()
    aoa = aero.angle_of_attack

    x_body = x_m - x_m.max() / 2

    dx = np.gradient(x_m)
    dr = np.gradient(r_m)
    theta_s = np.abs(np.arctan2(dr, -dx))
    cp_wind = np.clip(1.93 * np.sin(np.clip(theta_s + aoa, 0, np.pi/2))**2, 0, 1.93)
    cp_lee = np.clip(1.93 * np.sin(np.clip(theta_s - aoa, 0, np.pi/2))**2, 0, 1.93)

    R_n = 1.2 * DIAMETER
    bs = billig_shock(25.0, R_n, np.deg2rad(32.5))

    x_w, y_w = _rotate(x_body, r_m, -aoa)
    x_l, y_l = _rotate(x_body, -r_m, -aoa)

    nose_x = x_body[0]
    y_sh = np.linspace(0, r_m.max() * 2.5, 600)
    x_sh = np.array([bs.axial_at(float(y)) for y in y_sh]) + nose_x
    sx_u, sy_u = _rotate(x_sh, y_sh, -aoa)
    sx_l, sy_l = _rotate(x_sh, -y_sh, -aoa)

    shoulder_idx = int(np.argmax(r_m))
    shoulder_x = x_body[shoulder_idx]
    shoulder_r = r_m[shoulder_idx]

    return dict(
        x_m=x_m, r_m=r_m, aero=aero, aoa=aoa, DIAMETER=DIAMETER,
        x_body=x_body, cp_wind=cp_wind, cp_lee=cp_lee, bs=bs,
        x_w=x_w, y_w=y_w, x_l=x_l, y_l=y_l,
        sx_u=sx_u, sy_u=sy_u, sx_l=sx_l, sy_l=sy_l,
        nose_x=nose_x, y_sh=y_sh, x_sh=x_sh, R_n=1.2*DIAMETER,
        shoulder_x=shoulder_x, shoulder_r=shoulder_r,
        shoulder_idx=shoulder_idx,
    )


def _body_fill(ax, d, zorder=5, lw_scale=1.0):
    """Fill body polygon and draw Cp-colored outlines."""
    body_x = np.concatenate([d["x_w"], d["x_l"][::-1]])
    body_y = np.concatenate([d["y_w"], d["y_l"][::-1]])
    ax.fill(body_x, body_y, color="#060810", ec="none", zorder=zorder)

    for x, y, cp, lw, alpha in [
        (d["x_w"], d["y_w"], d["cp_wind"], 4.5 * lw_scale, 1.0),
        (d["x_l"], d["y_l"], d["cp_lee"], 3.0 * lw_scale, 0.7),
    ]:
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap_heat, norm=Normalize(0, 1.93),
                            linewidths=lw, alpha=alpha, zorder=zorder+1)
        lc.set_array(cp[:-1])
        ax.add_collection(lc)


def _shock_lines(ax, d, zorder=7):
    """Draw Billig shock curves."""
    ax.plot(d["sx_u"], d["sy_u"], color=CYAN, lw=2.5, alpha=0.9, zorder=zorder)
    ax.plot(d["sx_l"], d["sy_l"], color=CYAN, lw=2.5, alpha=0.9, zorder=zorder)
    ax.plot(d["sx_u"], d["sy_u"], color=CYAN, lw=6, alpha=0.10, zorder=zorder-1)
    ax.plot(d["sx_l"], d["sy_l"], color=CYAN, lw=6, alpha=0.10, zorder=zorder-1)


def _build_trees(d):
    """Build KDTrees for body surface and shock curve in body frame."""
    n_body = len(d["x_body"])
    t = np.linspace(0, 1, max(n_body * 4, 800))
    xb_dense = np.interp(t, np.linspace(0, 1, n_body), d["x_body"])
    rb_dense = np.interp(t, np.linspace(0, 1, n_body), d["r_m"])
    body_pts = np.vstack([
        np.column_stack([xb_dense, rb_dense]),
        np.column_stack([xb_dense, -rb_dense]),
    ])
    body_tree = KDTree(body_pts)

    shock_pts = np.vstack([
        np.column_stack([d["x_sh"], d["y_sh"]]),
        np.column_stack([d["x_sh"], -d["y_sh"]]),
    ])
    shock_tree = KDTree(shock_pts)
    return body_tree, shock_tree


def _distances(X, Y, aoa, body_tree, shock_tree):
    """Compute distances to body and shock for each grid point via KDTree."""
    Xb, Yb = _rotate(X, Y, aoa)
    grid_pts = np.column_stack([Xb.ravel(), Yb.ravel()])
    d_body = body_tree.query(grid_pts)[0].reshape(X.shape)
    d_shock = shock_tree.query(grid_pts)[0].reshape(X.shape)
    return Xb, Yb, d_body, d_shock


def _inside_mask(Xb, Yb, d):
    """Mask for points inside the body."""
    r_at = np.interp(Xb, d["x_body"], d["r_m"], left=0.0, right=0.0)
    return ((np.abs(Yb) < r_at) &
            (Xb >= d["x_body"].min()) &
            (Xb <= d["x_body"].max()))


def _density_field(d, nx=2000, ny=1600, xlim=(-6, 5), ylim=(-5, 5)):
    """Compute the density field from Rankine-Hugoniot + Billig shock.

    Uses the library's billig_shock for geometry, then solves the
    oblique-shock jump conditions at each point on the shock surface
    and fills the shock layer with the isentropic density profile.
    Returns the raw density field, grid coordinates, and body mask.
    """
    from aether.viz.flow import schlieren_intensity

    gamma = 1.4
    M_inf = 25.0
    T_inf = 230.0
    p_inf = 1.05
    R_air = 287.058
    rho_inf = p_inf / (R_air * T_inf)

    xg = np.linspace(*xlim, nx)
    yg = np.linspace(*ylim, ny)
    X, Y = np.meshgrid(xg, yg)

    body_tree, shock_tree = _build_trees(d)
    Xb, Yb, d_body, d_shock = _distances(X, Y, d["aoa"], body_tree, shock_tree)
    inside = _inside_mask(Xb, Yb, d)

    nose_x = d["nose_x"]
    shoulder_x = d["shoulder_x"]
    shoulder_r = d["shoulder_r"]
    x_max = d["x_body"].max()
    standoff = d["bs"].standoff

    # ── Local shock angle from Billig shock tangent ──
    shock_pts_upper = np.column_stack([d["x_sh"], d["y_sh"]])
    dx_sh = np.gradient(d["x_sh"])
    dy_sh = np.gradient(d["y_sh"])
    beta_at_shock = np.abs(np.arctan2(dx_sh, dy_sh))
    beta_at_shock = np.clip(beta_at_shock, np.deg2rad(5), np.pi / 2)

    shock_upper_tree = KDTree(shock_pts_upper)
    _, shock_idx = shock_upper_tree.query(
        np.column_stack([Xb.ravel(), np.abs(Yb.ravel())])
    )
    beta_local = beta_at_shock[shock_idx].reshape(X.shape)

    # ── Rankine-Hugoniot oblique shock ──
    Mn1 = M_inf * np.sin(beta_local)
    Mn1_sq = Mn1**2
    rho_ratio = ((gamma + 1) * Mn1_sq) / ((gamma - 1) * Mn1_sq + 2)
    rho_ratio = np.clip(rho_ratio, 1.0, (gamma + 1) / (gamma - 1))
    rho_post = rho_inf * rho_ratio

    # Post-shock Mach (normal component)
    Mn2_sq = ((gamma - 1) * Mn1_sq + 2) / (2 * gamma * Mn1_sq - (gamma - 1))
    Mn2_sq = np.clip(Mn2_sq, 0.01, 1.0)

    # ── Shock-layer density profile ──
    signed_shock = d_body - d_shock
    layer_thick = d_body + d_shock + 1e-6
    frac_to_body = np.clip(d_shock / layer_thick, 0, 1)

    # Station along body (body frame)
    station = np.clip((Xb - nose_x) / max(x_max - nose_x, 0.01), 0, 2.0)
    stag_factor = np.exp(-2.0 * station)

    # Isentropic compression: density rises from post-shock toward stagnation
    # The velocity in the shock layer decreases toward the body surface,
    # so by mass conservation ρ increases. Use cubic Blasius-like profile.
    layer_profile = 1.0 + 0.5 * frac_to_body**2 * (3 - 2 * frac_to_body) * stag_factor

    # Shock transition: tanh across the shock thickness
    shock_width = 0.02
    shock_frac = 0.5 * (1.0 + np.tanh(signed_shock / shock_width))

    rho_layer = rho_post * layer_profile
    rho = rho_inf + (rho_layer - rho_inf) * shock_frac

    # ── Prandtl-Meyer expansion at shoulder ──
    past_shoulder = np.clip(
        (Xb - shoulder_x) / max(x_max - shoulder_x, 0.01), 0, 1.5
    )
    near_surface = np.exp(-d_body / 0.5)
    expansion_ratio = 1.0 - 0.6 * past_shoulder * near_surface * shock_frac
    rho *= expansion_ratio

    # ── Entropy layer ──
    entropy_frac = np.clip(signed_shock / max(standoff, 0.1), 0, 1)
    entropy_amp = 0.15 * rho_post * np.exp(-((entropy_frac - 0.35) / 0.05)**2)
    entropy_amp *= shock_frac * np.exp(-1.5 * station)
    rho += entropy_amp

    # ── Wake ──
    x_base = x_max
    behind = np.clip((Xb - x_base) / 3.0, 0, 1)
    r_base = max(float(d["r_m"][-1]), 0.01)
    r_body = np.abs(Yb)

    neck_x = x_base + 2.5 * r_base
    neck_frac = np.exp(-((Xb - neck_x) / (1.5 * r_base))**2)
    shear_r = r_base * (1.0 + 0.3 * np.clip((Xb - x_base) / (3 * r_base), 0, 1))
    shear_layer = np.exp(-((r_body - shear_r) / (0.2 * r_base))**2)

    in_wake = behind * np.exp(-0.5 * (r_body / (r_base * 1.2))**2)
    rho *= (1.0 - 0.7 * in_wake * (1.0 - behind**2))
    rho *= (1.0 + 0.3 * neck_frac * shear_layer)

    far_wake = np.clip((Xb - neck_x) / 3.0, 0, 1)
    wake_spread = r_base * (1.0 + 0.5 * far_wake)
    rho *= (1.0 - 0.15 * far_wake * np.exp(-0.5 * (r_body / wake_spread)**2))

    # ── Boundary layer ──
    bl_thickness = 0.08 + 0.12 * station
    bl_factor = np.exp(-(d_body / bl_thickness)**2) * shock_frac
    rho *= (1.0 - 0.15 * bl_factor)

    rho = np.where(inside, np.nan, rho)

    # ── Numerical schlieren: |∇ρ|/ρ then exp(-κ|∇ρ|/ρ_ref) ──
    # This is exactly how aether.viz.flow.schlieren_intensity works.
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    # Extrapolate density into body so gradient at surface is physical
    rho_smooth = gaussian_filter(np.nan_to_num(rho, nan=0), sigma=4)
    norm_smooth = gaussian_filter((~inside).astype(float), sigma=4) + 1e-10
    rho_for_grad = np.where(inside, rho_smooth / norm_smooth, rho)

    grad_x = np.gradient(rho_for_grad, dx, axis=1)
    grad_y = np.gradient(rho_for_grad, dy, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Relative gradient: |∇ρ|/ρ — same as schlieren_magnitude(relative=True)
    grad_rel = grad_mag / np.clip(rho_for_grad, rho_inf * 0.1, None)
    grad_rel = np.where(inside, 0, grad_rel)

    # Suppress body-edge artifact: the extrapolation creates an artificial
    # gradient at the body surface that is not part of the flow.
    # Use a wider kernel so the suppression extends far enough from the edge.
    body_proximity = gaussian_filter(inside.astype(float), sigma=5.0)
    edge_suppress = np.clip(1.0 - 2.0 * body_proximity, 0, 1)
    edge_suppress = np.where(inside, 0, edge_suppress)
    grad_rel *= edge_suppress

    # schlieren_intensity: S = exp(-κ |∇ρ|/ρ / ρ_ref)
    # with ρ_ref at the 99.5th percentile — the library's default
    valid = ~inside & (grad_rel > 0)
    reference = float(np.percentile(grad_rel[valid], 99.5)) if valid.any() else 1.0
    gain = 20.0  # DEFAULT_SCHLIEREN_GAIN from aether.viz.flow
    S = np.exp(-gain * grad_rel / max(reference, 1e-20))
    S = np.where(inside, np.nan, S)
    S = np.clip(S, 0, 1)

    # Invert: 0 = undisturbed freestream (→ dark background),
    #         1 = strongest gradient (→ bright shock)
    S = 1.0 - S

    return X, Y, S, inside


# ═══════════════════════════════════════════════════════════════════════
# 1. HERO: SCHLIEREN — real SU2 NEMO Euler, no analytical shortcuts
# ═══════════════════════════════════════════════════════════════════════
def gen_hero_schlieren():
    print("  [1/6] Orion capsule schlieren (SU2 NEMO CFD)...")
    from matplotlib.tri import Triangulation
    from examples.artemis1.entry import orion_meridian, orion_aerodynamics, DIAMETER

    if not _CFD_SCHLIEREN.exists():
        raise FileNotFoundError(
            f"CFD schlieren data not found at {_CFD_SCHLIEREN}.\n"
            "Generate it by running the SU2 NEMO Euler solver first."
        )

    data = np.load(_CFD_SCHLIEREN)
    coords = data["cut_coords"]
    tris = data["cut_triangles"]
    sint = data["schlieren_intensity"]
    mach = float(data["freestream_mach"])
    alpha = float(data["alpha"])

    x_m, r_m = orion_meridian()
    aero = orion_aerodynamics()

    tri = Triangulation(coords[:, 0], coords[:, 1], tris)
    display = 1.0 - sint

    fig, ax = plt.subplots(figsize=(14, 8.5))

    ax.tripcolor(tri, display, cmap=cmap_schl, shading="gouraud",
                 vmin=0, vmax=1, zorder=0)

    body_x = np.concatenate([x_m, x_m[::-1]])
    body_z = np.concatenate([r_m, -r_m[::-1]])
    ax.fill(body_x, body_z, color="#060810", ec="none", zorder=5)
    ax.plot(x_m, r_m, color=TEXT, lw=1.5, alpha=0.7, zorder=6)
    ax.plot(x_m, -r_m, color=TEXT, lw=1.2, alpha=0.5, zorder=6)

    for z_a in np.linspace(-4.0, 4.0, 9):
        if abs(z_a) > 0.5:
            x0 = -2.0
            dx, dz = 1.5 * np.cos(alpha), 1.5 * np.sin(alpha)
            ax.annotate("", xy=(x0 + dx, z_a + dz), xytext=(x0, z_a),
                        arrowprops=dict(arrowstyle="-|>", color=CYAN,
                                        lw=0.6, alpha=0.18))
    ax.annotate(f"$V_\\infty$  (M = {mach:.0f})",
                xy=(-2.0, 4.5), fontsize=11,
                color=CYAN, fontweight="bold", alpha=0.4, path_effects=glow)

    info = (f"Orion CEV — Artemis I\n"
            f"  D = {DIAMETER:.2f} m\n"
            f"  L/D = {aero.lift_to_drag:.2f}\n"
            f"  α = {np.rad2deg(alpha):.1f}°\n"
            f"  $C_D$ = {aero.drag_coefficient:.3f}\n"
            f"  β = {aero.ballistic_coefficient:.0f} kg/m²\n"
            f"  SU2 NEMO Euler (5-species air)")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8, color=TEXT_DIM,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=BG_PANEL, ec="#2A3050", alpha=0.85))

    pad = DIAMETER * 0.5
    ax.set(xlim=(-pad, x_m.max() + pad),
           ylim=(-(r_m.max() + pad), r_m.max() + pad),
           aspect="equal", xlabel="Distance (m)", ylabel="Distance (m)")
    ax.grid(True, alpha=0.06, color=GRID)

    fig.suptitle("Hypersonic Entry — Orion Capsule at Trim\n"
                 f"SU2 NEMO 5-species Euler  ·  M = {mach:.0f}  ·  "
                 f"α = {np.rad2deg(alpha):.1f}°  ·  65 km",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.97)

    fig.savefig(f"{OUT}/hero_cfd_schlieren.png")
    plt.close(fig)
    print("    done")


# ═══════════════════════════════════════════════════════════════════════
# 2. CFD COMBINED: Cp + body in flow
# ═══════════════════════════════════════════════════════════════════════
def gen_cfd_combined():
    print("  [2/6] Orion Cp + shock...")
    d = _orion_data()
    aero, aoa = d["aero"], d["aoa"]

    dx = np.gradient(d["x_m"])
    dr = np.gradient(d["r_m"])
    theta_s = np.abs(np.arctan2(dr, -dx))
    cp_zero = np.clip(1.93 * np.sin(theta_s)**2, 0, 1.93)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})

    ax1.plot(d["x_m"], cp_zero, color=TEXT_DIM, lw=1.5, ls=":", alpha=0.5,
             label="α = 0°", zorder=3)
    ax1.fill_between(d["x_m"], d["cp_wind"], d["cp_lee"], alpha=0.12,
                     color=ORANGE, zorder=2)
    ax1.plot(d["x_m"], d["cp_wind"], color=GOLD, lw=2.5,
             label=f"Windward (α = {np.rad2deg(aoa):.0f}°)", zorder=4)
    ax1.plot(d["x_m"], d["cp_lee"], color=CYAN, lw=2.0, ls="--",
             label=f"Leeward (α = {np.rad2deg(aoa):.0f}°)", zorder=4)

    ax1.axhline(1.93, color=ORANGE, ls=":", lw=0.7, alpha=0.3)
    ax1.text(d["x_m"].max() * 0.5, 1.96, "$C_{p,max} = 1.93$", fontsize=8,
             color=ORANGE, alpha=0.6, path_effects=glow2)

    shoulder_x = d["x_m"][d["shoulder_idx"]]
    ax1.axvline(shoulder_x, color=TEXT_DIM, ls=":", lw=0.6, alpha=0.3)
    ax1.text(shoulder_x * 0.35, 1.6, "Heat\nshield", fontsize=8,
             color=TEXT_DIM, ha="center", alpha=0.6)
    ax1.text(shoulder_x + (d["x_m"].max() - shoulder_x) * 0.4, 1.6,
             "Backshell", fontsize=8, color=TEXT_DIM, ha="center", alpha=0.6)

    ax1.set_title("Surface Pressure Coefficient\nOrion at hypersonic trim",
                  fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax1.set(xlabel="Body station x (m)", ylabel="$C_p$",
            xlim=(-0.05, d["x_m"].max() + 0.1), ylim=(-0.05, 2.1))
    ax1.grid(True, alpha=0.15, color=GRID)
    ax1.legend(loc="center left", framealpha=0.8)

    if _CFD_SCHLIEREN.exists():
        from matplotlib.tri import Triangulation as Tri
        cfd = np.load(_CFD_SCHLIEREN)
        cfd_coords = cfd["cut_coords"]
        cfd_tri = Tri(cfd_coords[:, 0], cfd_coords[:, 1], cfd["cut_triangles"])
        cfd_display = 1.0 - cfd["schlieren_intensity"]
        cfd_mach = float(cfd["freestream_mach"])

        ax2.tripcolor(cfd_tri, cfd_display, cmap=cmap_schl, shading="gouraud",
                      vmin=0, vmax=1, zorder=0)

        body_x = np.concatenate([d["x_m"], d["x_m"][::-1]])
        body_z = np.concatenate([d["r_m"], -d["r_m"][::-1]])
        ax2.fill(body_x, body_z, color="#060810", ec="none", zorder=5)
        ax2.plot(d["x_m"], d["r_m"], color=TEXT, lw=1.2, alpha=0.6, zorder=6)
        ax2.plot(d["x_m"], -d["r_m"], color=TEXT, lw=1.0, alpha=0.4, zorder=6)

        pad = d["DIAMETER"] * 0.4
        ax2.set(xlim=(-pad, d["x_m"].max() + pad),
                ylim=(-(d["r_m"].max() + pad), d["r_m"].max() + pad))
        flow_label = f"SU2 NEMO Euler, M = {cfd_mach:.0f}"
    else:
        X, Y, S, _ = _density_field(d, nx=1000, ny=1000, xlim=(-5, 5), ylim=(-5, 5))
        ax2.imshow(S, extent=[-5, 5, -5, 5], origin="lower", cmap=cmap_schl,
                   vmin=0, vmax=1, aspect="equal", zorder=0, interpolation="bilinear")
        _body_fill(ax2, d, zorder=5, lw_scale=0.7)
        ax2.set(xlim=(-5, 5), ylim=(-5, 5))
        flow_label = "Billig shock, M ≈ 25"

    ax2.annotate("$V_\\infty$", xy=(-1.5, 3.5), fontsize=14, color=CYAN,
                 fontweight="bold", alpha=0.5, path_effects=glow)

    ax2.set_title(f"Orion in Flow Field\n({flow_label})",
                  fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax2.set(xlabel="Distance (m)", ylabel="Distance (m)", aspect="equal")
    ax2.grid(True, alpha=0.08, color=GRID)

    fig.suptitle("AETHER — Orion Capsule Aerodynamics\n"
                 f"Newtonian $C_p$ at {np.rad2deg(aoa):.0f}° trim "
                 f"(L/D = {aero.lift_to_drag:.2f})",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    fig.savefig(f"{OUT}/cfd_combined.png")
    plt.close(fig)
    print("    done")


# ═══════════════════════════════════════════════════════════════════════
# 3. PLASMA SHEATH — real Saha on SU2 NEMO T/p/X_NO fields
# ═══════════════════════════════════════════════════════════════════════
def gen_plasma():
    from aether.cfd.plasma import critical_density
    from matplotlib.tri import Triangulation
    from examples.artemis1.entry import orion_meridian, orion_aerodynamics, DIAMETER
    print("  [3/6] Plasma sheath (SU2 NEMO CFD)...")

    if not _CFD_PLASMA.exists():
        raise FileNotFoundError(
            f"CFD plasma data not found at {_CFD_PLASMA}.\n"
            "Generate it by running extract_plasma_data.py after the NEMO solve."
        )

    data = np.load(_CFD_PLASMA)
    coords = data["cut_coords"]
    tris = data["cut_triangles"]
    ne = data["electron_density"]
    T_tr = data["temperature_tr"]
    T_ve = data["temperature_ve"]
    p = data["pressure"]
    X_NO = data["no_mole_fraction"]
    mach = float(data["freestream_mach"])
    alpha = float(data["alpha"])
    ne_crit_gps = float(data["ne_crit_gps"])
    ne_crit_sband = float(data["ne_crit_sband"])

    x_m, r_m = orion_meridian()
    aero = orion_aerodynamics()

    tri = Triangulation(coords[:, 0], coords[:, 1], tris)
    ne_display = np.clip(ne, 1e10, None)
    log_ne = np.log10(ne_display)

    gps_freq = 1575.42e6
    sband_freq = 2.2e9
    ka_freq = 32e9
    ne_crit_ka = critical_density(ka_freq)

    fig, ax = plt.subplots(figsize=(14, 8.5))

    im = ax.tripcolor(tri, log_ne, cmap=cmap_plasma, shading="gouraud",
                      vmin=10, vmax=22, zorder=0)

    ax.tricontour(tri, ne, levels=[ne_crit_gps], colors=[GREEN],
                  linewidths=2.5, linestyles="--", zorder=8)
    ax.tricontour(tri, ne, levels=[ne_crit_sband], colors=[ORANGE],
                  linewidths=2.0, linestyles=":", zorder=8)
    ax.tricontour(tri, ne, levels=[ne_crit_ka], colors=[GOLD],
                  linewidths=1.5, linestyles="-.", zorder=8)

    body_x = np.concatenate([x_m, x_m[::-1]])
    body_z = np.concatenate([r_m, -r_m[::-1]])
    ax.fill(body_x, body_z, color="#060810", ec="none", zorder=5)
    ax.plot(x_m, r_m, color=TEXT, lw=1.5, alpha=0.7, zorder=6)
    ax.plot(x_m, -r_m, color=TEXT, lw=1.2, alpha=0.5, zorder=6)

    for z_a in np.linspace(-4.0, 4.0, 9):
        if abs(z_a) > 0.5:
            x0 = -2.0
            dx, dz = 1.5 * np.cos(alpha), 1.5 * np.sin(alpha)
            ax.annotate("", xy=(x0 + dx, z_a + dz), xytext=(x0, z_a),
                        arrowprops=dict(arrowstyle="-|>", color=CYAN,
                                        lw=0.6, alpha=0.18))

    ax.annotate("GPS L1 blackout\nboundary",
                xy=(-1.0, 2.8), fontsize=9, color=GREEN, fontweight="bold",
                path_effects=glow, zorder=10)
    ax.annotate("S-band cutoff",
                xy=(0.5, -3.0), fontsize=8, color=ORANGE, fontweight="bold",
                path_effects=glow, zorder=10)
    ax.annotate("Ka-band cutoff",
                xy=(1.5, -2.0), fontsize=7, color=GOLD, fontweight="bold",
                path_effects=glow, zorder=10)

    ne_peak = ne[ne > 1e10].max() if np.any(ne > 1e10) else 0.0
    T_peak = T_tr.max()
    T_ve_peak = T_ve.max()
    X_NO_peak = X_NO[T_tr > 2000].max() if np.any(T_tr > 2000) else 0.0

    info = (f"SU2 NEMO 5-species Euler · M = {mach:.0f} · "
            f"α = {np.rad2deg(alpha):.1f}° · 65 km\n"
            f"  Saha ionisation on CFD T, p, X$_{{NO}}$\n"
            f"  $T_{{tr,peak}}$ = {T_peak/1e3:.1f} kK  "
            f"$T_{{ve,peak}}$ = {T_ve_peak/1e3:.1f} kK\n"
            f"  $n_{{e,peak}}$ = {ne_peak:.2e} m⁻³\n"
            f"  $X_{{NO,peak}}$ = {X_NO_peak:.3f}\n"
            f"  GPS L1 $n_{{e,crit}}$ = {ne_crit_gps:.2e} m⁻³")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8, color=TEXT_DIM,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=BG_PANEL, ec="#2A3050", alpha=0.85))

    pad = DIAMETER * 0.5
    ax.set(xlim=(-pad, x_m.max() + pad), ylim=(-(r_m.max() + pad), r_m.max() + pad),
           aspect="equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    fig.suptitle("Plasma Sheath & RF Blackout — Orion at M = 25\n"
                 "Electron density from Saha equilibrium on SU2 NEMO solution",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.97)

    from matplotlib.ticker import LogFormatter
    sm = plt.cm.ScalarMappable(cmap=cmap_plasma, norm=LogNorm(1e10, 1e22))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, label="$n_e$ (m$^{-3}$)", shrink=0.65,
                      pad=0.02, aspect=30)
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=TEXT_DIM)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GREEN, lw=2.5, ls="--",
               label=f"GPS L1 cutoff ({gps_freq/1e6:.0f} MHz)"),
        Line2D([0], [0], color=ORANGE, lw=2, ls=":",
               label=f"S-band cutoff ({sband_freq/1e9:.1f} GHz)"),
        Line2D([0], [0], color=GOLD, lw=1.5, ls="-.",
               label=f"Ka-band cutoff ({ka_freq/1e9:.0f} GHz)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.8)

    fig.savefig(f"{OUT}/hero_plasma_sheath.png")
    plt.close(fig)
    print("    done")


# ═══════════════════════════════════════════════════════════════════════
# 4. TRAJECTORY + FOOTPRINT
# ═══════════════════════════════════════════════════════════════════════
def gen_trajectory_footprint():
    from examples.artemis1.entry import (
        orion_aerodynamics, fly, footprints, FT, NMI, INTERFACE, TARGET,
    )
    from aether.guidance.skip import SkipPhase
    from aether.geodesy import GeodeticPosition, great_circle_range
    print("  [4/6] Trajectory + footprint...")

    aero = orion_aerodynamics()
    result, guidance = fly(aero)
    fps = footprints(result, aero)
    at = {phase: t for t, phase in result.phases}

    lon0, lat0 = float(result.states[1, 0]), float(result.states[2, 0])
    origin = GeodeticPosition(latitude=lat0, longitude=lon0, altitude=0.0)
    downrange_km = np.array([
        float(great_circle_range(
            origin,
            GeodeticPosition(latitude=float(result.states[2, i]),
                             longitude=float(result.states[1, i]), altitude=0.0)
        )) / 1e3
        for i in range(result.states.shape[1])
    ])
    alt_km = result.altitudes / 1e3
    spd_kms = result.states[3, :] / 1e3

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.2, 1],
                          hspace=0.35, wspace=0.3)
    ax_traj = fig.add_subplot(gs[0, :])
    ax_foot = fig.add_subplot(gs[1, 1])
    ax_bank = fig.add_subplot(gs[1, 0])

    phases = [
        (SkipPhase.INITIAL_ROLL, "Initial Roll", PURPLE),
        (SkipPhase.PREDICTOR_CORRECTOR, "Predictor-Corrector", CYAN),
        (SkipPhase.FINAL, "Final", GREEN),
        (SkipPhase.TERMINAL, "Terminal", ORANGE),
    ]
    edges_km = [downrange_km[0]]
    phase_info = []
    for phase, label, color in phases:
        if phase in at:
            k = int(np.searchsorted(result.times, at[phase]))
            edges_km.append(downrange_km[k])
            phase_info.append((label, color))
    edges_km.append(downrange_km[-1])

    for i in range(len(phase_info)):
        d0, d1 = edges_km[i], edges_km[i + 1]
        ax_traj.axvspan(d0, d1, alpha=0.06, color=phase_info[i][1], zorder=0)
        mid = (d0 + d1) / 2
        width_km = d1 - d0
        if width_km < 200:
            continue
        lbl = phase_info[i][0]
        if width_km < 600:
            lbl = lbl.replace("Predictor-Corrector", "Pred-Corr")
        ax_traj.text(mid, 128, lbl, ha="center", va="bottom",
                     fontsize=7, color=phase_info[i][1], fontweight="bold",
                     alpha=0.8, path_effects=glow2)

    pts = np.column_stack([downrange_km, alt_km]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="coolwarm",
                        norm=Normalize(spd_kms.min(), spd_kms.max()),
                        linewidths=3, zorder=5)
    lc.set_array(spd_kms[:-1])
    ax_traj.add_collection(lc)
    lc_g = LineCollection(segs, cmap="coolwarm",
                          norm=Normalize(spd_kms.min(), spd_kms.max()),
                          linewidths=8, alpha=0.15, zorder=4)
    lc_g.set_array(spd_kms[:-1])
    ax_traj.add_collection(lc_g)
    ax_traj.fill_between(downrange_km, alt_km, alpha=0.08, color=CYAN, zorder=1)

    ax_traj.annotate("Entry Interface\n(122 km, 11 km/s)",
                     xy=(downrange_km[0], alt_km[0]),
                     xytext=(downrange_km[0] + 500, alt_km[0] + 10),
                     fontsize=8.5, color=TEXT_DIM,
                     arrowprops=dict(arrowstyle="-|>", color=TEXT_DIM, lw=0.8),
                     path_effects=glow2)

    ap = int(np.argmax(np.where(result.times > 150, result.altitudes, -np.inf)))
    ax_traj.annotate(f"Skip apogee\n({alt_km[ap]:.0f} km)",
                     xy=(downrange_km[ap], alt_km[ap]),
                     xytext=(downrange_km[ap] + 400, alt_km[ap] + 15),
                     fontsize=9, color=CYAN, fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1),
                     path_effects=glow)

    first_dip = int(np.argmin(alt_km[:ap]))
    ax_traj.annotate(
        f"Peak {result.peak_deceleration_g:.1f}g\n({alt_km[first_dip]:.0f} km)",
        xy=(downrange_km[first_dip], alt_km[first_dip]),
        xytext=(downrange_km[first_dip] - 200, alt_km[first_dip] - 20),
        fontsize=8, color=ORANGE,
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8),
        path_effects=glow2)

    ax_traj.plot(downrange_km[-1], alt_km[-1], marker="*", ms=14, color=PINK,
                 markeredgecolor="#FFaacc", markeredgewidth=0.5, zorder=10)
    ax_traj.annotate("Splashdown", xy=(downrange_km[-1], alt_km[-1]),
                     xytext=(downrange_km[-1] - 600, 15),
                     fontsize=8.5, color=PINK, fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color=PINK, lw=0.8),
                     path_effects=glow)

    ax_traj.axhline(120, color=TEXT_DIM, ls=":", lw=0.6, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=Normalize(spd_kms.min(), spd_kms.max()))
    cb = fig.colorbar(sm, ax=ax_traj, label="Speed (km/s)", shrink=0.7,
                      pad=0.01, aspect=30)
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=TEXT_DIM)

    ax_traj.set_title("Artemis I Skip Entry — Altitude vs. Downrange\n"
                      "Orion capsule, PredGuid guidance",
                      fontsize=13, fontweight="bold", color=TEXT, pad=10)
    ax_traj.set(xlabel="Downrange from entry interface (km)",
                ylabel="Altitude (km)")
    ax_traj.set_xlim(downrange_km[0] - 100, downrange_km[-1] + 200)
    ax_traj.set_ylim(0, 135)
    ax_traj.grid(True, alpha=0.12, color=GRID)

    # Bank angle
    n_bank = len(result.bank)
    t_bank = result.times[:n_bank] / 60.0
    bank_deg = np.rad2deg(result.bank)
    ax_bank.fill_between(t_bank, bank_deg, alpha=0.15, color=PURPLE)
    ax_bank.plot(t_bank, bank_deg, color=PURPLE, lw=1.5, zorder=3)
    ax_bank.axhline(0, color=TEXT_DIM, lw=0.5, alpha=0.3)
    signs = np.sign(result.bank)
    flips = np.nonzero(np.diff(signs))[0] + 1
    for fi in flips:
        ax_bank.axvline(t_bank[min(fi, n_bank-1)], color=ORANGE, ls=":",
                        lw=0.7, alpha=0.4)
    ax_bank.text(0.98, 0.95, f"{len(flips)} bank reversals",
                 transform=ax_bank.transAxes, ha="right", va="top",
                 fontsize=8, color=ORANGE, path_effects=glow2)
    ax_bank.set_title("Bank Angle Command", fontsize=10, fontweight="bold",
                      color=TEXT, pad=6)
    ax_bank.set(xlabel="Time (min)", ylabel="Bank (deg)")
    ax_bank.set_xlim(t_bank[0], t_bank[-1])
    ax_bank.grid(True, alpha=0.12, color=GRID)

    # Footprint
    fp_colors = [CYAN, ORANGE, GREEN]
    fp0 = fps[0][2]

    for i, (label, epoch, fp, target) in enumerate(fps):
        boundary_common = np.array([fp0.local(b) for b in fp.boundary])
        if boundary_common.shape[0] < 3:
            continue
        dr_km = boundary_common[:, 0] / 1e3
        cr_km = boundary_common[:, 1] / 1e3
        dr_km = np.append(dr_km, dr_km[0])
        cr_km = np.append(cr_km, cr_km[0])

        if i == 0:
            ax_foot.plot(cr_km, dr_km, color=fp_colors[i], lw=1.0, ls="--",
                         alpha=0.4, zorder=3)
        else:
            ax_foot.fill(cr_km, dr_km, color=fp_colors[i],
                         alpha=0.25 + 0.15 * i, zorder=3+i)
            ax_foot.plot(cr_km, dr_km, color=fp_colors[i], lw=1.2, alpha=0.7,
                         zorder=4+i)
            sample_common = np.array([fp0.local(s) for s in fp.samples])
            ax_foot.scatter(sample_common[:, 1]/1e3, sample_common[:, 0]/1e3,
                            color=fp_colors[i], s=15, alpha=0.7, zorder=5+i,
                            edgecolors="white", linewidths=0.3)

    t_dr, t_cr = fp0.local(fps[0][3])
    ax_foot.plot(t_cr/1e3, t_dr/1e3, marker="*", ms=14, color=PINK, zorder=20,
                 markeredgecolor="#FFaacc", markeredgewidth=0.5)
    ax_foot.annotate("Target", xy=(t_cr/1e3, t_dr/1e3),
                     xytext=(t_cr/1e3 + 60, t_dr/1e3 + 200),
                     fontsize=8, fontweight="bold", color=PINK,
                     path_effects=glow, zorder=20)

    for i, (label, epoch, fp, target) in enumerate(fps):
        area = fp.area_km2
        area_str = f"{area/1e6:.1f}M" if area >= 1e6 else f"{area/1e3:.0f}K"
        style = dict(lw=6, alpha=0.5) if i > 0 else dict(lw=1, ls="--", alpha=0.4)
        ax_foot.plot([], [], color=fp_colors[i], **style,
                     label=f"{label.title()}: {area_str} km²")

    ax_foot.set_title("Reachable Footprint", fontsize=10, fontweight="bold",
                      color=TEXT, pad=6)
    ax_foot.set(xlabel="Crossrange (km)", ylabel="Downrange (km)")
    ax_foot.grid(True, alpha=0.12, color=GRID)
    ax_foot.legend(loc="upper left", fontsize=7.5, framealpha=0.8)

    fig.suptitle("AETHER — Artemis I Skip Entry", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    fig.savefig(f"{OUT}/hero_trajectory_footprint.png")
    plt.close(fig)
    print("    done")
    return result, aero, fps


# ═══════════════════════════════════════════════════════════════════════
# 5. 3D GLOBE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════
def gen_globe_trajectory(result, fps):
    """Render the Artemis I trajectory on a ray-traced 3D globe."""
    from aether.ellipsoid import geodetic_to_ecef
    from aether.viz.globe import look_at, sun_direction
    from aether.viz.scene import globe_plate, draw_track, draw_marker
    from aether.viz import default_blue_marble
    print("  [5/6] 3D globe trajectory...")

    n_pts = result.states.shape[1]
    step = max(n_pts // 1000, 1)
    lons = result.states[1, ::step]
    lats = result.states[2, ::step]
    alts = result.altitudes[::step]
    alt_scale = 3.0

    ecef = np.array([
        geodetic_to_ecef(float(lat), float(lon), float(alt) * alt_scale)
        for lon, lat, alt in zip(lons, lats, alts)
    ])
    ecef_ground = np.array([
        geodetic_to_ecef(float(lat), float(lon), 0.0)
        for lon, lat in zip(lons, lats)
    ])

    mid_idx = n_pts * 2 // 5
    mid_lon = float(result.states[1, mid_idx])
    mid_lat = float(result.states[2, mid_idx])
    target = geodetic_to_ecef(mid_lat, mid_lon, 0.0)

    az_ecef = np.arctan2(float(target[1]), float(target[0]))
    cam = look_at(target, distance=18e6, azimuth=az_ecef,
                  elevation=np.deg2rad(18), width=1600, height=1200)
    sun = sun_direction(az_ecef + 0.4, declination=np.deg2rad(12))

    bm = default_blue_marble()
    tex = bm.texture(height=4096, month=12)
    fig, ax = globe_plate(cam, tex, sun=sun, atmosphere=0.7, ambient=0.15)

    draw_track(ax, ecef_ground, cam, color="#FFFFFF", width=1.5, alpha=0.12,
               zorder=2)

    spd = result.states[3, ::step] / 1e3
    draw_track(ax, ecef, cam, values=spd, cmap="coolwarm",
               vmin=float(spd.min()), vmax=float(spd.max()),
               width=3.5, alpha=0.95, zorder=5)
    draw_track(ax, ecef, cam, color="white", width=7, alpha=0.06, zorder=4)

    ei = geodetic_to_ecef(float(result.states[2, 0]),
                          float(result.states[1, 0]),
                          float(result.altitudes[0]) * alt_scale)
    draw_marker(ax, ei, cam, marker="o", s=60, color=CYAN, zorder=10,
                edgecolors="white", linewidths=0.5)

    sp = geodetic_to_ecef(float(result.states[2, -1]),
                          float(result.states[1, -1]), 0.0)
    draw_marker(ax, sp, cam, marker="*", s=200, color=PINK, zorder=10,
                edgecolors="#FFaacc", linewidths=0.5)

    fp_colors_globe = [CYAN, ORANGE, GREEN]
    for i, (label, epoch, fp, target_pt) in enumerate(fps):
        if i == 0:
            continue
        boundary = fp.boundary
        if len(boundary) < 3:
            continue
        b_ecef = np.array([
            geodetic_to_ecef(b.latitude, b.longitude, 0.0)
            for b in boundary
        ])
        b_ecef = np.vstack([b_ecef, b_ecef[:1]])
        draw_track(ax, b_ecef, cam, color=fp_colors_globe[i],
                   width=2.0, alpha=0.7, zorder=4)

    glow_blk = [patheffects.withStroke(linewidth=4, foreground="black")]
    glow_blk2 = [patheffects.withStroke(linewidth=2, foreground="black")]
    ax.text(0.5, 0.97, "Artemis I — Orion Skip Entry",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            color="white", ha="center", va="top", path_effects=glow_blk)
    ax.text(0.5, 0.93,
            "Simulated trajectory on ray-traced WGS84 globe, coloured by speed",
            transform=ax.transAxes, fontsize=8.5, color="#BBBBBB",
            ha="center", va="top", path_effects=glow_blk2)

    fig.savefig(f"{OUT}/hero_globe_trajectory.png", dpi=DPI,
                facecolor="black")
    plt.close(fig)
    print("    done")


# ═══════════════════════════════════════════════════════════════════════
# 6. STANDALONE FOOTPRINT
# ═══════════════════════════════════════════════════════════════════════
def gen_footprint(result, aero, fps):
    from aether.geodesy import GeodeticPosition
    print("  [6/6] Standalone footprint...")

    fp_colors = [CYAN, ORANGE, GREEN]
    fp0 = fps[0][2]

    fig, ax = plt.subplots(figsize=(10, 10))

    track_dr, track_cr = [], []
    for i in range(0, result.states.shape[1], 5):
        pt = GeodeticPosition(
            latitude=float(result.states[2, i]),
            longitude=float(result.states[1, i]),
            altitude=0.0,
        )
        dr, cr = fp0.local(pt)
        track_dr.append(dr / 1e3)
        track_cr.append(cr / 1e3)
    track_dr, track_cr = np.array(track_dr), np.array(track_cr)

    for i, (label, epoch, fp, target) in enumerate(fps):
        boundary_common = np.array([fp0.local(b) for b in fp.boundary])
        if boundary_common.shape[0] < 3:
            continue
        dr_km = boundary_common[:, 0] / 1e3
        cr_km = boundary_common[:, 1] / 1e3
        dr_km = np.append(dr_km, dr_km[0])
        cr_km = np.append(cr_km, cr_km[0])

        if i == 0:
            ax.fill(cr_km, dr_km, color=fp_colors[i], alpha=0.04, zorder=2)
            ax.plot(cr_km, dr_km, color=fp_colors[i], lw=1.2, ls="--",
                    alpha=0.4, zorder=3)
        else:
            ax.fill(cr_km, dr_km, color=fp_colors[i],
                    alpha=0.20 + 0.15 * i, zorder=3+i)
            ax.plot(cr_km, dr_km, color=fp_colors[i], lw=1.5, alpha=0.7,
                    zorder=4+i)

        sample_common = np.array([fp0.local(s) for s in fp.samples])
        ax.scatter(sample_common[:, 1]/1e3, sample_common[:, 0]/1e3,
                   color=fp_colors[i], s=30, alpha=0.8, zorder=5+i,
                   edgecolors="white", linewidths=0.4, marker="o")

        area = fp.area_km2
        area_str = f"{area/1e6:.1f}M" if area >= 1e6 else f"{area/1e3:.0f}K"
        style = dict(lw=8, alpha=0.5) if i > 0 else dict(lw=1, ls="--", alpha=0.4)
        ax.plot([], [], color=fp_colors[i], **style,
                label=f"{label.title()}: {area_str} km²")

    ax.plot(track_cr, track_dr, color=PURPLE, lw=2, alpha=0.7, zorder=8,
            label="Ground track")
    ax.plot(track_cr, track_dr, color=PURPLE, lw=5, alpha=0.12, zorder=7)

    ep_indices = [0]
    for lbl, epoch, fp, tgt in fps[1:]:
        k = int(np.searchsorted(result.times, epoch))
        ep_indices.append(k)
    ep_labels = ["Entry\nInterface"]
    for lbl, _, _, _ in fps[1:]:
        ep_labels.append(lbl.title().replace(" ", "\n"))
    ep_colors = [CYAN] + fp_colors[1:len(fps)]

    for ki, elabel, ecolor in zip(ep_indices, ep_labels, ep_colors):
        ti = min(ki // 5, len(track_dr) - 1)
        ax.plot(track_cr[ti], track_dr[ti], marker="o", ms=8, color=ecolor,
                markeredgecolor="white", markeredgewidth=0.5, zorder=12)
        offset_x = 80 if track_cr[ti] < np.median(track_cr) else -150
        ax.annotate(elabel, xy=(track_cr[ti], track_dr[ti]),
                    xytext=(track_cr[ti] + offset_x, track_dr[ti] + 200),
                    fontsize=7.5, color=ecolor, ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=ecolor, lw=0.7),
                    path_effects=glow2, zorder=12)

    t_dr, t_cr = fp0.local(fps[0][3])
    ax.plot(t_cr/1e3, t_dr/1e3, marker="*", ms=22, color=PINK, zorder=20,
            markeredgecolor="#FFaacc", markeredgewidth=0.8)
    ax.annotate("TARGET", xy=(t_cr/1e3, t_dr/1e3),
                xytext=(t_cr/1e3 + 100, t_dr/1e3 + 300),
                fontsize=11, fontweight="bold", color=PINK,
                path_effects=glow, zorder=20,
                arrowprops=dict(arrowstyle="-|>", color=PINK, lw=1.0))

    ax.text(0.02, 0.02,
            "Each region shows where the capsule\n"
            "can still land from that point in the\n"
            "flight — every bank-angle history that\n"
            "satisfies the path constraints. The set\n"
            "shrinks as the capsule spends energy.\n"
            "Dots: individual simulated landings.",
            transform=ax.transAxes, fontsize=8, color=TEXT_DIM,
            va="bottom", family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.5", fc=BG_PANEL, ec="#2A3050",
                      alpha=0.9))

    ax.set_title("Reachable Landing Footprint — Artemis I Skip Entry\n"
                 "Where the capsule can still reach, "
                 "at three points of the flight",
                 fontsize=13, fontweight="bold", color=TEXT, pad=12)
    ax.set(xlabel="Crossrange (km)", ylabel="Downrange (km)")
    ax.grid(True, alpha=0.12, color=GRID)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    fig.savefig(f"{OUT}/hero_reachable_footprint.png")
    plt.close(fig)
    print("    done")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating AETHER hero graphics...")
    gen_hero_schlieren(); gc.collect()
    gen_cfd_combined(); gc.collect()
    gen_plasma(); gc.collect()
    result, aero, fps = gen_trajectory_footprint(); gc.collect()
    try:
        gen_globe_trajectory(result, fps); gc.collect()
    except Exception as e:
        print(f"    globe skipped: {e}")
    gen_footprint(result, aero, fps); gc.collect()
    print("\nAll graphics saved to docs/_static/")
