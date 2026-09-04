"""
ai_engine/heatmap.py

Renders a team positional heatmap as a base64-encoded PNG from a list of
pitch-space (x_m, y_m) positions — meters, origin at the center circle,
x running along the length toward a goal, y across the width. This is
the same convention Stage 5's `image_point_to_pitch` already outputs
(see stage5_pitch_mapping/homography.py), so callers can pass its output
straight in.

Design choice (per the project handoff): heatmaps are generated
on-demand at request time from the saved player_tracking_csv's
pitch_x/pitch_y columns, not stored anywhere — no new model fields, no
persisted image files. See apps/matches/views.py's match_results() for
the caller.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless: runs inside a Django request/worker, no display available
import matplotlib.pyplot as plt
import numpy as np

# Standard full-size pitch. Matches the (36..52.5, -20.16..20.16) box-view
# calibration in tasks.py, which is a sub-region of a pitch centered at
# (0, 0) with these full dimensions.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

PITCH_LINE_COLOR = "#e8f5ee"
PITCH_FACE_COLOR = "#2f6f4d"


def _draw_pitch_outline(ax, length=PITCH_LENGTH_M, width=PITCH_WIDTH_M):
    """
    Draws a simple top-down pitch outline centered at (0, 0): boundary,
    halfway line, center circle, and both penalty boxes. Lines are drawn
    at zorder=2 so they stay visible on top of the heatmap fill
    (zorder=1) rather than getting painted over by it.
    """
    half_l, half_w = length / 2, width / 2

    ax.set_facecolor(PITCH_FACE_COLOR)

    ax.plot(
        [-half_l, half_l, half_l, -half_l, -half_l],
        [-half_w, -half_w, half_w, half_w, -half_w],
        color=PITCH_LINE_COLOR, linewidth=1.3, zorder=2,
    )
    ax.plot([0, 0], [-half_w, half_w], color=PITCH_LINE_COLOR, linewidth=1.3, zorder=2)

    center_circle = plt.Circle((0, 0), 9.15, fill=False, color=PITCH_LINE_COLOR, linewidth=1.3, zorder=2)
    ax.add_patch(center_circle)
    ax.plot(0, 0, marker="o", markersize=2, color=PITCH_LINE_COLOR, zorder=2)

    # 18-yard penalty boxes at each end (16.5m deep, 40.32m wide)
    box_depth, box_width = 16.5, 40.32
    for sign in (-1, 1):
        x_edge = sign * half_l
        x_box = x_edge - sign * box_depth
        ax.plot(
            [x_edge, x_box, x_box, x_edge],
            [-box_width / 2, -box_width / 2, box_width / 2, box_width / 2],
            color=PITCH_LINE_COLOR, linewidth=1.3, zorder=2,
        )

    ax.set_xlim(-half_l, half_l)
    ax.set_ylim(-half_w, half_w)
    ax.set_aspect("equal")
    ax.axis("off")


def render_heatmap_png(positions, pitch_length=PITCH_LENGTH_M, pitch_width=PITCH_WIDTH_M, bins=(21, 14)):
    """
    positions: list of (x_m, y_m) tuples in pitch-space meters.

    Returns a base64-encoded PNG string (no "data:image/png;base64,"
    prefix — callers that need it for an <img> src should prepend that
    themselves, so this stays reusable outside HTML contexts). Returns
    None if there are no positions to plot, so callers can fall back to
    a placeholder instead of rendering an empty pitch.

    Uses a plain 2D histogram (bins, not a smoothed KDE) — deliberately
    simple for v1, matching what a partial-pitch calibration (only the
    box-view region has real data for test_11.mp4) can actually support
    honestly. cmin=1 makes zero-count cells fully transparent so the
    green pitch shows through everywhere there's no data, rather than
    painting the whole pitch in the colormap's lowest color.
    """
    if not positions:
        return None

    xs = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[1] for p in positions], dtype=float)

    half_l, half_w = pitch_length / 2, pitch_width / 2

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(PITCH_FACE_COLOR)  # matches ax facecolor so the whole image is a solid pitch, no white margin

    _draw_pitch_outline(ax, pitch_length, pitch_width)

    ax.hist2d(
        xs, ys,
        bins=bins,
        range=[[-half_l, half_l], [-half_w, half_w]],
        cmap="inferno",
        alpha=0.75,
        cmin=1,      # zero-count cells stay transparent, not solid dark
        zorder=1,
    )

    buf = io.BytesIO()
    # NOTE: transparent=True here would override ax.set_facecolor() and
    # force the pitch green transparent too (matplotlib forces BOTH figure
    # and axes patches transparent when this flag is set, not just the
    # figure) — that's what made the pitch render white instead of green.
    # Explicit facecolor kwargs below is what actually keeps the green.
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")