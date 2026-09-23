"""Generate hero graphics for the AETHER documentation.

Uses the actual Orion capsule geometry, Billig shock correlation,
modified-Newtonian panel method, and Saha plasma sheath — all from
the library itself.
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

_schl = [BG, "#060D1A", "#0A1628", "#0E1F38", "#142A4A",
         "#1A3660", "#224478", "#2D5A90", "#3A72A8", "#4A8CC0",
         "#5CA8D8", "#70C0E8", "#88D4F0", CYAN, "#B0EEFF", "#FFF"]
cmap_schl = LinearSegmentedColormap.from_list("schl", _schl, N=512)

_heat = ["#0B0E17", "#1A0A2E", "#3D0A4A", "#6B0F3A", "#9C1528",
         "#CC3B1A", "#E86420", "#F09030", "#F8C050", "#FFEE88", "#FFF"]
cmap_heat = LinearSegmentedColormap.from_list("heat", _heat, N=256)

_plasma = ["#0B0E17", "#0D1040", "#1A1070", "#3010A0", "#6020C0",
           "#9030D0", "#C040D8", "#E060D0", "#FF80C0", "#FFB0E0", "#FFF"]
cmap_plasma = LinearSegmentedColormap.from_list("plasma_ne", _plasma, N=512)


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


def _schlieren_field(d, nx=2000, ny=1600, xlim=(-6, 5), ylim=(-5, 5)):
    """High-resolution synthetic schlieren via KDTree distance fields.

    Builds a physically motivated density field with:
    - Sharp density jump at the Billig bow shock
    - Compressed shock layer with stagnation-dependent density
    - Expansion fan at the shoulder
    - Entropy layer (contact surface)
    - Separated wake behind the body
    Then computes |grad(rho)| and renders as schlieren intensity.
    """
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

    # ── Density field ──
    rho = np.ones_like(X)

    # Shock transition: tanh sigmoid, thickness ~ 0.04 m
    shock_width = 0.04
    signed_shock = d_body - d_shock
    shock_frac = 0.5 * (1.0 + np.tanh(signed_shock / shock_width))

    # Stagnation factor: peaks near centerline & nose
    r_body = np.abs(Yb)
    station = np.clip((Xb - nose_x) / max(x_max - nose_x, 0.01), 0, 2.0)
    stag = np.exp(-1.2 * station) * np.exp(-0.3 * (r_body / max(shoulder_r, 0.01))**2)

    # Post-shock density ratio: 6 at stagnation, ~3 at shoulder, ~2 far away
    rho_ratio_local = 2.0 + 4.0 * stag

    # Layer profile: density decreases from shock-side toward wall
    layer_total = d_body + d_shock + 1e-6
    frac_from_shock = np.clip(d_shock / layer_total, 0, 1)
    wall_factor = 0.7 + 0.3 * frac_from_shock

    # Combine: freestream smoothly transitions to compressed layer
    rho_compressed = 1.0 + (rho_ratio_local - 1.0) * wall_factor
    rho = 1.0 + (rho_compressed - 1.0) * shock_frac

    # Expansion fan at shoulder
    past_shoulder = np.clip((Xb - shoulder_x) / max(x_max - shoulder_x, 0.01), 0, 1)
    near_body = np.exp(-d_body / 0.5)
    expansion = 0.4 * past_shoulder * near_body * shock_frac
    rho = rho * (1.0 - expansion)

    # Wake: low-density recirculation behind the body
    behind_body = np.clip((Xb - x_max) / 2.0, 0, 1)
    r_wake = max(float(d["r_m"][-1]), 0.01) * 1.2
    wake_core = np.exp(-0.5 * (r_body / r_wake)**2)
    wake = 0.6 * behind_body * wake_core * (1 - behind_body)
    rho = rho * (1.0 - wake)

    # Entropy layer: thin contact surface inside the shock layer
    entropy_dist = np.clip(signed_shock / max(d["bs"].standoff, 0.1), 0, 1)
    entropy_layer = 0.3 * np.exp(-((entropy_dist - 0.3) / 0.08)**2) * shock_frac
    rho = rho + entropy_layer * rho_ratio_local

    rho = np.where(inside, np.nan, rho)

    # ── Schlieren = |grad(rho)| ──
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    grad_x = np.gradient(np.nan_to_num(rho, nan=6.0), dx, axis=1)
    grad_y = np.gradient(np.nan_to_num(rho, nan=6.0), dy, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.where(inside, 0, grad_mag)

    # Multi-scale contrast: sharp features + broad layer structure
    valid = ~inside & (grad_mag > 0)
    p999 = np.percentile(grad_mag[valid], 99.9)
    p95 = np.percentile(grad_mag[valid], 95)
    S_sharp = 1.0 - np.exp(-40.0 * grad_mag / max(p999, 1e-10))
    S_broad = gaussian_filter(grad_mag, sigma=4.0)
    S_broad = 1.0 - np.exp(-8.0 * S_broad / max(p95, 1e-10))
    S = 0.55 * S_sharp + 0.45 * S_broad
    S = np.where(inside, 0, S)
    S = gaussian_filter(S, sigma=0.6)
    S = np.clip(S, 0, 1)

    return X, Y, S, inside


# ═══════════════════════════════════════════════════════════════════════
# 1. HERO: SCHLIEREN — the schlieren IS the visual, no shock overlay
# ═══════════════════════════════════════════════════════════════════════
def gen_hero_schlieren():
    print("  [1/6] Orion capsule schlieren (high-res)...")
    d = _orion_data()
    aero, aoa, bs = d["aero"], d["aoa"], d["bs"]
    from examples.artemis1.entry import DIAMETER

    X, Y, S, inside = _schlieren_field(d)

    fig, ax = plt.subplots(figsize=(14, 8.5))

    # The schlieren field IS the hero — NO shock lines on top
    ax.imshow(S, extent=[-6, 5, -5, 5], origin="lower", cmap=cmap_schl,
              vmin=0, vmax=1, aspect="equal", zorder=0, interpolation="bilinear")

    _body_fill(ax, d, zorder=5, lw_scale=0.8)

    for y_a in np.linspace(-3.5, 3.5, 8):
        if abs(y_a) > 0.5:
            ax.annotate("", xy=(-4.0, y_a), xytext=(-5.5, y_a),
                        arrowprops=dict(arrowstyle="-|>", color=CYAN,
                                        lw=0.6, alpha=0.18))
    ax.annotate("$V_\\infty$  (M = 25)", xy=(-5.3, 4.2), fontsize=11,
                color=CYAN, fontweight="bold", alpha=0.4, path_effects=glow)

    ax.annotate("Bow shock",
                xy=(d["sx_u"][80], d["sy_u"][80]),
                xytext=(d["sx_u"][80] + 1.8, d["sy_u"][80] + 0.8),
                fontsize=10, color=CYAN, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=0.8),
                path_effects=glow, zorder=10)

    hs_idx = len(d["x_w"]) // 5
    ax.annotate("Windward (high $C_p$)",
                xy=(d["x_w"][hs_idx], d["y_w"][hs_idx]),
                xytext=(d["x_w"][hs_idx] + 2.2, d["y_w"][hs_idx] - 1.2),
                color=GOLD, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.8),
                path_effects=glow, zorder=10)

    sw, sl = _rotate(d["shoulder_x"], d["shoulder_r"], -aoa)
    ax.annotate("Expansion fan",
                xy=(sw + 0.3, sl - 0.3),
                xytext=(sw + 2.0, sl - 1.5),
                fontsize=8, color="#70B0D0", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#70B0D0", lw=0.7),
                path_effects=glow2, zorder=10)

    nx_r, ny_r = _rotate(d["nose_x"], 0, -aoa)
    sx_r, sy_r = _rotate(d["nose_x"] - bs.standoff, 0, -aoa)
    ax.annotate("", xy=(nx_r, ny_r), xytext=(sx_r, sy_r),
                arrowprops=dict(arrowstyle="<->", color=CYAN, lw=1.0), zorder=9)
    ax.text((nx_r+sx_r)/2 - 0.3, (ny_r+sy_r)/2 + 0.4,
            f"$\\Delta$ = {bs.standoff:.2f} m", fontsize=8.5, color=CYAN,
            path_effects=glow2, zorder=10)

    info = (f"Orion CEV — Artemis I\n"
            f"  D = {DIAMETER:.2f} m\n"
            f"  L/D = {aero.lift_to_drag:.2f}\n"
            f"  α = {np.rad2deg(aoa):.1f}°\n"
            f"  $C_D$ = {aero.drag_coefficient:.3f}\n"
            f"  β = {aero.ballistic_coefficient:.0f} kg/m²")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8, color=TEXT_DIM,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=BG_PANEL, ec="#2A3050", alpha=0.85))

    ax.set(xlim=(-6, 5), ylim=(-5, 5), aspect="equal",
           xlabel="Distance (m)", ylabel="Distance (m)")
    ax.grid(True, alpha=0.06, color=GRID)

    fig.suptitle("Hypersonic Entry — Orion Capsule at Trim\n"
                 "Numerical schlieren · Modified-Newtonian $C_p$ · Billig shock",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.97)

    sm = plt.cm.ScalarMappable(cmap=cmap_heat, norm=Normalize(0, 1.93))
    cb = fig.colorbar(sm, ax=ax, label="$C_p$", shrink=0.65, pad=0.02, aspect=30)
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=TEXT_DIM)

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

    X, Y, S, _ = _schlieren_field(d, nx=1000, ny=1000, xlim=(-5, 5), ylim=(-5, 5))
    ax2.imshow(S, extent=[-5, 5, -5, 5], origin="lower", cmap=cmap_schl,
               vmin=0, vmax=1, aspect="equal", zorder=0, interpolation="bilinear")

    _body_fill(ax2, d, zorder=5, lw_scale=0.7)

    ax2.annotate("$V_\\infty$", xy=(-4.5, 4.2), fontsize=14, color=CYAN,
                 fontweight="bold", alpha=0.5, path_effects=glow)
    for y_a in [-3, -1.5, 1.5, 3]:
        ax2.annotate("", xy=(-3, y_a), xytext=(-4.5, y_a),
                     arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=0.8, alpha=0.25))

    ax2.set_title("Orion in Flow Field\n(Billig shock, M ≈ 25)",
                  fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax2.set(xlabel="Distance (m)", ylabel="Distance (m)",
            xlim=(-5, 5), ylim=(-5, 5), aspect="equal")
    ax2.grid(True, alpha=0.08, color=GRID)

    fig.suptitle("AETHER — Orion Capsule Aerodynamics\n"
                 f"Modified Newtonian at {np.rad2deg(aoa):.0f}° trim "
                 f"(L/D = {aero.lift_to_drag:.2f})",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    fig.savefig(f"{OUT}/cfd_combined.png")
    plt.close(fig)
    print("    done")


# ═══════════════════════════════════════════════════════════════════════
# 3. PLASMA SHEATH (high resolution)
# ═══════════════════════════════════════════════════════════════════════
def gen_plasma():
    from aether.cfd.plasma import (
        saha_electron_density, plasma_frequency, critical_density,
    )
    from examples.artemis1.entry import DIAMETER
    print("  [3/6] Plasma sheath (high-res)...")
    d = _orion_data()
    aoa, bs = d["aoa"], d["bs"]

    T_freestream = 247.0
    p_freestream = 21.0
    V_inf = 7880.0
    q_inf = 0.5 * 3e-4 * V_inf**2
    T_stag = 11000.0
    p_stag = q_inf
    T_wall = 2500.0

    nx, ny = 1800, 1800
    xg = np.linspace(-5, 4, nx)
    yg = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(xg, yg)

    body_tree, shock_tree = _build_trees(d)
    Xb, Yb, d_body, d_shock = _distances(X, Y, aoa, body_tree, shock_tree)
    inside = _inside_mask(Xb, Yb, d)

    nose_x = d["nose_x"]
    x_max = d["x_body"].max()

    layer_thick = d_body + d_shock + 1e-6
    frac_from_wall = np.clip(d_body / layer_thick, 0, 1)
    in_layer = (d_shock < d_body * 3) & ~inside & (d_body < 3.5)

    station_frac = np.clip((Xb - nose_x) / max(x_max - nose_x, 0.01), 0, 1.5)

    T_shock_local = T_stag * np.exp(-1.5 * station_frac)
    T_shock_local = np.clip(T_shock_local, 3000, T_stag)
    T_field = T_wall + (T_shock_local - T_wall) * frac_from_wall**0.6
    T_field = np.where(in_layer, T_field, T_freestream)
    T_field = np.where(inside, T_wall, T_field)

    p_shock_local = p_stag * np.exp(-1.0 * station_frac)
    p_shock_local = np.clip(p_shock_local, p_freestream, p_stag)
    p_field = np.where(in_layer, p_shock_local, p_freestream)

    ne_flat = saha_electron_density(
        T_field.ravel().astype(np.float64),
        p_field.ravel().astype(np.float64),
    )
    ne = ne_flat.reshape(T_field.shape)
    ne = np.where(inside, 0, ne)
    ne = np.where(ne < 1e10, 1e10, ne)

    gps_freq = 1575.42e6
    sband_freq = 2.2e9
    ne_crit_gps = critical_density(gps_freq)
    ne_crit_sband = critical_density(sband_freq)

    fig, ax = plt.subplots(figsize=(14, 8.5))

    ne_plot = np.where(inside, np.nan, ne)
    levels = np.logspace(14, 22, 100)
    im = ax.contourf(X, Y, ne_plot, levels=levels, cmap=cmap_plasma,
                     norm=LogNorm(1e14, 1e22), extend="both", zorder=0)

    cs_gps = ax.contour(X, Y, ne_plot, levels=[ne_crit_gps], colors=[GREEN],
                        linewidths=2.5, linestyles="--", zorder=8)
    cs_sband = ax.contour(X, Y, ne_plot, levels=[ne_crit_sband], colors=[ORANGE],
                          linewidths=1.8, linestyles=":", zorder=8)

    _body_fill(ax, d, zorder=5, lw_scale=0.8)
    ax.plot(d["sx_u"], d["sy_u"], color=CYAN, lw=1.5, alpha=0.5, zorder=7)
    ax.plot(d["sx_l"], d["sy_l"], color=CYAN, lw=1.5, alpha=0.5, zorder=7)

    for y_a in np.linspace(-3.5, 3.5, 7):
        if abs(y_a) > 0.5:
            ax.annotate("", xy=(-3.5, y_a), xytext=(-4.8, y_a),
                        arrowprops=dict(arrowstyle="-|>", color=CYAN,
                                        lw=0.6, alpha=0.15))

    ax.annotate("GPS L1 blackout\nboundary",
                xy=(-1.0, 2.8), fontsize=9, color=GREEN, fontweight="bold",
                path_effects=glow, zorder=10)
    ax.annotate("Bow shock",
                xy=(d["sx_u"][60], d["sy_u"][60]),
                xytext=(d["sx_u"][60] + 1.5, d["sy_u"][60] + 0.6),
                fontsize=9, color=CYAN, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=0.8),
                path_effects=glow, zorder=10)

    info = (f"Saha equilibrium ionisation\n"
            f"  $T_{{stag}}$ = {T_stag/1e3:.0f},000 K\n"
            f"  $p_{{stag}}$ = {p_stag:.0f} Pa\n"
            f"  $n_{{e,peak}}$ ≈ {ne[in_layer].max():.1e} m⁻³\n"
            f"  GPS L1 $n_{{e,crit}}$ = {ne_crit_gps:.1e} m⁻³\n"
            f"  Species: NO (9.26 eV)")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8, color=TEXT_DIM,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=BG_PANEL, ec="#2A3050", alpha=0.85))

    ax.set(xlim=(-5, 4), ylim=(-5, 5), aspect="equal",
           xlabel="Distance (m)", ylabel="Distance (m)")
    ax.grid(True, alpha=0.06, color=GRID)

    fig.suptitle("Plasma Sheath — Orion at M ≈ 25\n"
                 "Electron density from Saha equilibrium ionisation",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.97)

    cb = fig.colorbar(im, ax=ax, label="$n_e$ (m$^{-3}$)", shrink=0.65,
                      pad=0.02, aspect=30)
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=TEXT_DIM)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GREEN, lw=2, ls="--",
               label=f"GPS L1 cutoff ({gps_freq/1e6:.0f} MHz)"),
        Line2D([0], [0], color=ORANGE, lw=1.5, ls=":",
               label=f"S-band cutoff ({sband_freq/1e9:.1f} GHz)"),
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
