"""
ai_engine/heatmap.py

Renders authentic FotMob-style continuous positional heatmaps as base64-encoded
PNGs from pitch-space (x_m, y_m) coordinates — meters, origin at center circle,
x running along the length (-52.5 to 52.5), y across width (-34.0 to 34.0).

Features:
- Full broadcast stadium turf with 10 alternating lawn stripes
- Clean white/pale-mint field markings: center circle, penalty boxes, 6-yard boxes,
  penalty spots, penalty D arcs, and corner arcs
- Silky-smooth Gaussian Kernel Density Estimation (KDE) instead of blocky bins
- Progressive alpha thermal gradient: transparent pitch where inactive, glowing
  mint/lime -> radiant gold -> fiery orange -> intense crimson hotspot
- Configurable attack direction indicator (e.g. Attacking ->)
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless: runs inside worker / request
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

# Standard full-size pitch (FIFA recommendations: 105m x 68m)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

PITCH_FACE_COLOR = "#183327"
PITCH_LINE_COLOR = "#84D8AB"
STRIPE_COLOR_EVEN = "#1B382B"
STRIPE_COLOR_ODD = "#152E23"


def _draw_pitch_outline(ax, length=PITCH_LENGTH_M, width=PITCH_WIDTH_M, attack_direction=None):
    """
    Draws a broadcast-grade top-down football pitch with alternating turf stripes,
    penalty boxes, goal boxes, penalty spots, penalty D arcs, and corner arcs.
    Lines are drawn with high zorder so they remain crisp and legible through the heat glow.
    """
    half_l, half_w = length / 2, width / 2

    ax.set_facecolor(PITCH_FACE_COLOR)

    # 1. 10 Alternating vertical lawn stripes
    stripe_w = length / 10
    for i in range(10):
        color = STRIPE_COLOR_EVEN if i % 2 == 0 else STRIPE_COLOR_ODD
        ax.axvspan(-half_l + i * stripe_w, -half_l + (i + 1) * stripe_w, color=color, zorder=1)

    # 2. Boundary & Halfway lines
    ax.plot(
        [-half_l, half_l, half_l, -half_l, -half_l],
        [-half_w, -half_w, half_w, half_w, -half_w],
        color=PITCH_LINE_COLOR, linewidth=1.4, zorder=2, alpha=0.9,
    )
    ax.plot([0, 0], [-half_w, half_w], color=PITCH_LINE_COLOR, linewidth=1.4, zorder=2, alpha=0.9)

    # 3. Center circle (9.15m radius) & center mark
    cc = Circle((0, 0), 9.15, fill=False, color=PITCH_LINE_COLOR, linewidth=1.4, zorder=2, alpha=0.9)
    ax.add_patch(cc)
    ax.plot(0, 0, marker="o", markersize=3, color=PITCH_LINE_COLOR, zorder=2, alpha=0.9)

    # 4. Penalty boxes (16.5m deep, 40.32m wide) & Goal boxes (5.5m deep, 18.32m wide)
    for sign in (-1, 1):
        x_edge = sign * half_l
        # 18-yard penalty area
        x_box = x_edge - sign * 16.5
        ax.plot(
            [x_edge, x_box, x_box, x_edge],
            [-20.16, -20.16, 20.16, 20.16],
            color=PITCH_LINE_COLOR, linewidth=1.4, zorder=2, alpha=0.9,
        )
        # 6-yard goal area
        x_gbox = x_edge - sign * 5.5
        ax.plot(
            [x_edge, x_gbox, x_gbox, x_edge],
            [-9.16, -9.16, 9.16, 9.16],
            color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2, alpha=0.9,
        )
        # Penalty spot (11m)
        spot_x = x_edge - sign * 11.0
        ax.plot(spot_x, 0, marker="o", markersize=3, color=PITCH_LINE_COLOR, zorder=2, alpha=0.9)
        # Penalty D arc
        theta1, theta2 = (-53, 53) if sign == -1 else (127, 233)
        arc = Arc((spot_x, 0), 18.3, 18.3, angle=0, theta1=theta1, theta2=theta2, color=PITCH_LINE_COLOR, linewidth=1.3, zorder=2, alpha=0.9)
        ax.add_patch(arc)

    # 5. Corner arcs (1.0m radius)
    ax.add_patch(Arc((-half_l, -half_w), 2.0, 2.0, angle=0, theta1=0, theta2=90, color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2, alpha=0.9))
    ax.add_patch(Arc((-half_l, half_w), 2.0, 2.0, angle=0, theta1=270, theta2=360, color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2, alpha=0.9))
    ax.add_patch(Arc((half_l, -half_w), 2.0, 2.0, angle=0, theta1=90, theta2=180, color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2, alpha=0.9))
    ax.add_patch(Arc((half_l, half_w), 2.0, 2.0, angle=0, theta1=180, theta2=270, color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2, alpha=0.9))

    # 6. Attacking direction indicator
    if attack_direction == "right":
        ax.text(28, -half_w - 2.6, "Attacking  →", ha="center", va="top", color="#A2E8C2", fontsize=7.5, weight="bold")
    elif attack_direction == "left":
        ax.text(-28, -half_w - 2.6, "←  Attacking", ha="center", va="top", color="#A2E8C2", fontsize=7.5, weight="bold")

    ax.set_xlim(-half_l - 2, half_l + 2)
    ax.set_ylim(-half_w - 4, half_w + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def _get_fotmob_colormap():
    """
    Builds the custom FotMob thermal gradient:
    Transparent Pitch -> Soft Mint/Lime -> Radiant Yellow -> Fiery Orange -> Crimson Core
    """
    colors_list = [
        (0.00, (0.23, 0.88, 0.60, 0.00)),  # 0% density: fully transparent pitch
        (0.05, (0.23, 0.88, 0.60, 0.16)),  # subtle edge glow
        (0.20, (0.35, 0.94, 0.35, 0.45)),  # lime
        (0.40, (1.00, 0.92, 0.10, 0.70)),  # bright golden yellow
        (0.65, (1.00, 0.50, 0.05, 0.85)),  # warm fiery orange
        (0.85, (0.95, 0.18, 0.10, 0.92)),  # vivid red
        (1.00, (0.80, 0.00, 0.00, 0.98)),  # intense scarlet core
    ]
    return LinearSegmentedColormap.from_list("fotmob_heat", [(p[0], p[1]) for p in colors_list])


def render_heatmap_png(
    positions,
    pitch_length=PITCH_LENGTH_M,
    pitch_width=PITCH_WIDTH_M,
    bins=(210, 136),
    sigma=None,
    attack_direction=None,
):
    """
    positions: list of (x_m, y_m) tuples in pitch-space meters.
    attack_direction: 'right', 'left', or None.

    Returns a base64-encoded PNG string (without "data:image/png;base64," prefix).
    Returns None if there are no positions to plot.
    """
    if not positions:
        return None

    xs = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[1] for p in positions], dtype=float)

    # Clamp any extreme sensor outliers outside the pitch boundary
    half_l, half_w = pitch_length / 2, pitch_width / 2
    valid_mask = (xs >= -half_l - 2) & (xs <= half_l + 2) & (ys >= -half_w - 2) & (ys <= half_w + 2)
    xs = xs[valid_mask]
    ys = ys[valid_mask]

    if len(xs) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160, facecolor="#12241D")
    fig.patch.set_facecolor("#12241D")

    # Draw the pitch outline and markings
    _draw_pitch_outline(ax, pitch_length, pitch_width, attack_direction=attack_direction)

    # If only 1 or 2 points, duplicate with tiny jitter so 2D binning functions properly
    if len(xs) < 3:
        xs = np.concatenate([xs, xs + 0.1, xs - 0.1])
        ys = np.concatenate([ys, ys + 0.1, ys - 0.1])

    # High-resolution fine binning
    H, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=bins,
        range=[[-half_l, half_l], [-half_w, half_w]],
    )

    # Adaptive Gaussian smoothing
    if sigma is None:
        sigma = 4.5 if len(xs) > 300 else (3.8 if len(xs) > 40 else 2.8)

    smoothed = gaussian_filter(H.T, sigma=sigma)
    max_val = np.max(smoothed)

    if max_val > 0:
        norm_density = smoothed / max_val
        cmap = _get_fotmob_colormap()

        ax.imshow(
            norm_density,
            extent=[-half_l, half_l, -half_w, half_w],
            origin="lower",
            cmap=cmap,
            interpolation="bicubic",
            zorder=3,
        )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")