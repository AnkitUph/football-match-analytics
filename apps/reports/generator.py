"""
Generates a publication-grade PDF match report from apps.matches.views.build_match_report_context()
— the exact same data the results page (results.html) renders as HTML, so
the PDF can never show a number that disagrees with the website for the
same match.

Triggered automatically upon match processing / calibration, and on-demand
via /matches/<uuid>/report/pdf/.
"""

import base64
import io

from django.core.files.base import ContentFile

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from apps.reports.models import Report

PRIMARY_COLOR = colors.HexColor("#1A2E35")
SECONDARY_HEADER = colors.HexColor("#2C4A54")
ACCENT_GREEN = colors.HexColor("#4FAE79")
ACCENT_BLUE = colors.HexColor("#5B9BD5")
ROW_TINT = colors.HexColor("#F8F9FA")
GRID_LINE = colors.HexColor("#E5E7EB")
NOTE_AMBER = colors.HexColor("#B8860B")
BADGE_GRAY = colors.HexColor("#6B7280")

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, GRID_LINE),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_TINT]),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("TOPPADDING", (0, 0), (-1, -1), 3.0),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
])


def _b64_heatmap_to_image(b64_string, width_mm=82):
    """
    Decodes a base64 PNG into a reportlab Image flowable, preserving the
    source aspect ratio. Returns None if there's no image.
    """
    if not b64_string:
        return None

    from PIL import Image as PILImage

    data = base64.b64decode(b64_string)
    pil_img = PILImage.open(io.BytesIO(data))
    aspect = pil_img.height / pil_img.width
    width = width_mm * mm
    height = width * aspect

    return Image(io.BytesIO(data), width=width, height=height)


def _render_tactical_pitch_png(tactical_data, home_short, away_short, width_mm=174):
    """
    Renders a high-resolution top-down tactical pitch graphic containing:
    - Pitch markings & lawn striping
    - Team convex hull polygons & tactical centroids
    - Passing network links weighted by pass volume
    - Player nodes with jersey numbers and names
    Returns a ReportLab Image flowable, or None if data is missing.
    """
    if not tactical_data:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from PIL import Image as PILImage

    pitch_length = 105.0
    pitch_width = 68.0
    half_l, half_w = pitch_length / 2, pitch_width / 2

    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="#14211D")
    ax.set_facecolor("#1A342B")

    # Alternating lawn stripes (10 vertical stripes)
    stripe_w = pitch_length / 10
    for s_idx in range(10):
        stripe_color = "#1E3B31" if s_idx % 2 == 0 else "#183229"
        ax.axvspan(-half_l + s_idx * stripe_w, -half_l + (s_idx + 1) * stripe_w, color=stripe_color, zorder=1)

    line_col = "#68BA92"

    # Boundary & Halfway line
    ax.plot([-half_l, half_l, half_l, -half_l, -half_l], [-half_w, -half_w, half_w, half_w, -half_w], color=line_col, lw=1.5, zorder=2)
    ax.plot([0, 0], [-half_w, half_w], color=line_col, lw=1.5, zorder=2)

    # Center circle (9.15m radius) & spot
    cc = Circle((0, 0), 9.15, fill=False, color=line_col, lw=1.5, zorder=2)
    ax.add_patch(cc)
    ax.plot(0, 0, marker="o", markersize=3, color=line_col, zorder=2)

    # Penalty boxes (16.5m deep, 40.32m wide) & Goal boxes (5.5m deep, 18.32m wide)
    for sign in (-1, 1):
        x_edge = sign * half_l
        # 18-yard box
        x_box = x_edge - sign * 16.5
        ax.plot([x_edge, x_box, x_box, x_edge], [-20.16, -20.16, 20.16, 20.16], color=line_col, lw=1.5, zorder=2)
        # 6-yard box
        x_gbox = x_edge - sign * 5.5
        ax.plot([x_edge, x_gbox, x_gbox, x_edge], [-9.16, -9.16, 9.16, 9.16], color=line_col, lw=1.2, zorder=2)
        # Penalty spot (11m)
        ax.plot(x_edge - sign * 11.0, 0, marker="o", markersize=3, color=line_col, zorder=2)

    team_configs = [
        ("home", "#4FAE79", home_short),
        ("away", "#5B9BD5", away_short),
    ]

    for side, color, short_name in team_configs:
        t_data = tactical_data.get(side)
        if not t_data:
            continue

        shape = t_data.get("shape")
        # 1. Convex Hull Polygon
        if shape and shape.get("hull_vertices") and len(shape["hull_vertices"]) >= 3:
            pts = [[v["x"], v["y"]] for v in shape["hull_vertices"]]
            poly = Polygon(pts, closed=True, facecolor=color, alpha=0.22, edgecolor=color, lw=2.2, ls="--", zorder=3)
            ax.add_patch(poly)

            # Centroid
            c = shape.get("centroid")
            if c:
                ax.plot(c["x"], c["y"], marker="o", markersize=7, color="#ffffff", markeredgecolor=color, markeredgewidth=2.0, zorder=5)
                ax.plot(c["x"], c["y"], marker="o", markersize=16, color="none", markeredgecolor=color, markeredgewidth=1.2, ls=":", zorder=5)
                ax.text(c["x"], c["y"] - 3.8, f"{short_name} Centroid", ha="center", va="top", color="#ffffff", fontsize=7.5, weight="bold", zorder=6)

        # 2. Passing Links
        node_map = {n["id"]: n for n in t_data.get("nodes", [])}
        for link in t_data.get("links", []):
            src = node_map.get(link.get("source"))
            tgt = node_map.get(link.get("target"))
            if src and tgt:
                lw = min(4.5, max(1.2, (link.get("count", 1) ** 0.5) * 1.3))
                ax.plot([src["x"], tgt["x"]], [src["y"], tgt["y"]], color=color, alpha=0.55, lw=lw, zorder=4)

        # 3. Player Nodes
        for node in t_data.get("nodes", []):
            nx, ny = node["x"], node["y"]
            radius = min(15, max(9, 9 + (node.get("passes_made", 0) ** 0.5) * 1.0))
            ax.plot(nx, ny, marker="o", markersize=radius, color=color, markeredgecolor="#ffffff", markeredgewidth=1.2, zorder=6)
            jersey_str = str(node.get("jersey") or "")
            ax.text(nx, ny, jersey_str, ha="center", va="center", color="#ffffff", fontsize=6.5, weight="bold", zorder=7)
            ax.text(nx, ny + 2.5, node.get("name", ""), ha="center", va="bottom", color="#ffffff", fontsize=5.8, weight="semibold", alpha=0.92, zorder=7)

    ax.set_xlim(-half_l - 2, half_l + 2)
    ax.set_ylim(-half_w - 2, half_w + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.2)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", dpi=160, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    img_buf.seek(0)

    pil_img = PILImage.open(img_buf)
    aspect = pil_img.height / pil_img.width
    width = width_mm * mm
    height = width * aspect
    img_buf.seek(0)
    return Image(img_buf, width=width, height=height)


def _lineup_table(players):
    header = ["#", "Name", "Position", "Passes", "Distance (km)", "Status"]
    data = [header]
    for p in players:
        dist_str = f"{p['distance_km']}"
        status_str = "Tracked" if p.get("distance_confirmed") else ("Tracked (Auto)" if p.get("distance_is_real") else "Estimated")
        passes_str = str(p.get("passes_completed", p.get("passes", "—")))
        data.append([
            p.get("jersey_number", ""),
            p.get("name", ""),
            p.get("position", ""),
            passes_str,
            dist_str,
            status_str,
        ])

    table = Table(data, colWidths=[10 * mm, 60 * mm, 22 * mm, 22 * mm, 28 * mm, 32 * mm])
    table.setStyle(TABLE_STYLE)
    return table


def generate_match_report(match):
    """
    Builds the publication-grade PDF and saves it to this match's Report.
    Always produces a consistent 3-page report:
    Page 1: Match Overview, Key Metrics & Categorized Statistics, Heatmaps
    Page 2: Tactical Pitch Map (Shapes & Passing Network), Compactness Metrics, Shot Analysis
    Page 3: Team Lineups, Tracking Provenance & Analytical Footnotes
    """
    from apps.matches.views import build_match_report_context

    ctx = build_match_report_context(match)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=PRIMARY_COLOR, alignment=0)
    meta_style = ParagraphStyle("ReportMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=BADGE_GRAY)
    h2_style = ParagraphStyle("ReportH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=PRIMARY_COLOR, spaceAfter=4, spaceBefore=6)
    note_style = ParagraphStyle("ReportNote", parent=styles["Normal"], fontName="Helvetica-Oblique", textColor=NOTE_AMBER, fontSize=8.5, leading=11)
    foot_style = ParagraphStyle("ReportFoot", parent=styles["Normal"], fontName="Helvetica", textColor=BADGE_GRAY, fontSize=8, leading=11)

    story = []

    # ================= PAGE 1: OVERVIEW & TEAM STATISTICS =================
    story.append(Paragraph(f"{match.home_team.name} {match.home_score} &ndash; {match.away_score} {match.away_team.name}", title_style))
    date_str = match.match_date.strftime("%B %d, %Y") if hasattr(match.match_date, "strftime") else str(match.match_date)
    meta_bits = [date_str]
    if match.competition:
        meta_bits.append(match.competition)
    if match.stadium:
        meta_bits.append(match.stadium)
    meta_bits.append("Computer Vision Match Analytics Report")
    story.append(Paragraph(" &bull; ".join(meta_bits), meta_style))
    story.append(Spacer(1, 4 * mm))

    if not ctx.get("using_real_stats"):
        story.append(Paragraph(
            "Note: This match report is based on uncalibrated video frames. Metrics reflect deterministic tactical baseline models.",
            note_style,
        ))
        story.append(Spacer(1, 3 * mm))

    # Key Metrics Banner (Table with 4 KPI cards)
    home_ts = ctx["team_stats"]["home"]
    away_ts = ctx["team_stats"]["away"]

    kpi_headers = ["Possession", "Expected Goals (xG)", "Total Shots", "Pass Accuracy"]
    kpi_values = [
        f"{ctx['home_possession']}% - {ctx['away_possession']}%",
        f"{home_ts['xg']:.2f} - {away_ts['xg']:.2f}",
        f"{home_ts['shots']} - {away_ts['shots']}",
        f"{home_ts.get('pass_accuracy', 0)}% - {away_ts.get('pass_accuracy', 0)}%",
    ]
    kpi_table = Table([[Paragraph(f"<b>{h}</b>", styles["Normal"]) for h in kpi_headers], kpi_values], colWidths=[43 * mm, 45 * mm, 43 * mm, 43 * mm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTSIZE", (0, 1), (-1, 1), 12),
        ("TEXTCOLOR", (0, 1), (-1, 1), PRIMARY_COLOR),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID_LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 4 * mm))

    # Comprehensive Categorized Team Statistics Table
    story.append(Paragraph("Match Performance Statistics", h2_style))
    stats_rows = [
        ["Metric", match.home_team.short_name, match.away_team.short_name],
        # Group 1: Attacking
        ["ATTACKING & FINISHING", "", ""],
        ["Expected Goals (xG)", f"{home_ts['xg']:.2f}", f"{away_ts['xg']:.2f}"],
        ["Total Shots", str(home_ts["shots"]), str(away_ts["shots"])],
        ["Shots on Target", str(home_ts["shots_on_target"]), str(away_ts["shots_on_target"])],
        ["Shot Accuracy", f"{home_ts.get('shot_accuracy', 0)}%", f"{away_ts.get('shot_accuracy', 0)}%"],
        # Group 2: Passing
        ["PASSING & POSSESSION", "", ""],
        ["Possession", f"{ctx['home_possession']}%", f"{ctx['away_possession']}%"],
        ["Passes Completed", str(home_ts.get("passes_completed", home_ts.get("passes", 0))), str(away_ts.get("passes_completed", away_ts.get("passes", 0)))],
        ["Passes Attempted", str(home_ts.get("passes_attempted", home_ts.get("passes", 0))), str(away_ts.get("passes_attempted", away_ts.get("passes", 0)))],
        ["Pass Accuracy", f"{home_ts.get('pass_accuracy', 0)}%", f"{away_ts.get('pass_accuracy', 0)}%"],
        ["Corner Kicks", str(home_ts["corners"]), str(away_ts["corners"])],
        # Group 3: Defending
        ["DEFENDING & BALL RECOVERY", "", ""],
        ["Tackles Won", str(home_ts.get("tackles", 0)), str(away_ts.get("tackles", 0))],
        ["Interceptions", str(home_ts.get("interceptions", 0)), str(away_ts.get("interceptions", 0))],
        ["Clearances", str(home_ts.get("clearances", 0)), str(away_ts.get("clearances", 0))],
        ["Fouls / Yellow / Red", f"{home_ts['fouls']} / {home_ts['yellow_cards']} / {home_ts['red_cards']}", f"{away_ts['fouls']} / {away_ts['yellow_cards']} / {away_ts['red_cards']}"],
        # Group 4: Physical
        ["PHYSICAL PERFORMANCE", "", ""],
        ["Total Distance Covered", f"{home_ts['distance_km']} km", f"{away_ts['distance_km']} km"],
        ["Average Team Speed", f"{home_ts.get('average_speed', 0)} km/h", f"{away_ts.get('average_speed', 0)} km/h"],
    ]

    cat_indices = [1, 6, 12, 17]
    stats_t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID_LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_TINT]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]
    for ci in cat_indices:
        stats_t_style.extend([
            ("SPAN", (0, ci), (-1, ci)),
            ("BACKGROUND", (0, ci), (-1, ci), SECONDARY_HEADER),
            ("TEXTCOLOR", (0, ci), (-1, ci), colors.white),
            ("FONTNAME", (0, ci), (-1, ci), "Helvetica-Bold"),
            ("FONTSIZE", (0, ci), (-1, ci), 7.5),
        ])

    stats_table = Table(stats_rows, colWidths=[74 * mm, 50 * mm, 50 * mm])
    stats_table.setStyle(TableStyle(stats_t_style))
    story.append(stats_table)
    story.append(Spacer(1, 4 * mm))

    # Heatmaps side-by-side
    home_img = _b64_heatmap_to_image(ctx.get("home_heatmap"), width_mm=84)
    away_img = _b64_heatmap_to_image(ctx.get("away_heatmap"), width_mm=84)
    if home_img or away_img:
        story.append(Paragraph("Team Spatial Heatmaps", h2_style))
        row_labels = [
            Paragraph(f"<b>{match.home_team.name}</b>", styles["Normal"]),
            Paragraph(f"<b>{match.away_team.name}</b>", styles["Normal"]),
        ]
        row_images = [
            home_img if home_img else Paragraph("Heatmap not available", note_style),
            away_img if away_img else Paragraph("Heatmap not available", note_style),
        ]
        heatmap_table = Table([row_labels, row_images], colWidths=[87 * mm, 87 * mm])
        heatmap_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ]))
        story.append(heatmap_table)

    # ================= PAGE 2: TACTICAL PITCH & TEAM SHAPE =================
    story.append(PageBreak())
    story.append(Paragraph("Tactical Pitch Dynamics & Passing Networks", h2_style))

    tactical_img = _render_tactical_pitch_png(
        ctx.get("tactical_data"),
        match.home_team.short_name,
        match.away_team.short_name,
        width_mm=174,
    )
    if tactical_img:
        story.append(tactical_img)
        story.append(Spacer(1, 3 * mm))

    # Tactical Shape & Compactness Comparison Table
    t_data = ctx.get("tactical_data") or {}
    h_shape = (t_data.get("home") or {}).get("shape") or {}
    a_shape = (t_data.get("away") or {}).get("shape") or {}

    h_c = h_shape.get("centroid") or {"x": "—", "y": "—"}
    a_c = a_shape.get("centroid") or {"x": "—", "y": "—"}
    h_c_str = f"({h_c.get('x')}m, {h_c.get('y')}m)" if isinstance(h_c.get("x"), (int, float)) else "—"
    a_c_str = f"({a_c.get('x')}m, {a_c.get('y')}m)" if isinstance(a_c.get("x"), (int, float)) else "—"

    shape_rows = [
        ["Tactical Dimension", match.home_team.short_name, match.away_team.short_name],
        ["Compactness Area", f"{h_shape.get('area_sqm', '—')} m²", f"{a_shape.get('area_sqm', '—')} m²"],
        ["Tactical Length (Pitch X)", f"{h_shape.get('length_m', '—')} m", f"{a_shape.get('length_m', '—')} m"],
        ["Tactical Width (Pitch Y)", f"{h_shape.get('width_m', '—')} m", f"{a_shape.get('width_m', '—')} m"],
        ["Tactical Centroid (X, Y)", h_c_str, a_c_str],
    ]
    shape_table = Table(shape_rows, colWidths=[64 * mm, 55 * mm, 55 * mm])
    shape_table.setStyle(TABLE_STYLE)
    story.append(shape_table)
    story.append(Spacer(1, 4 * mm))

    # Shot Analysis Table
    shots = ctx.get("shots", [])
    if shots:
        story.append(Paragraph("Shot Analysis & Expected Goals", h2_style))
        shot_headers = ["Min", "Side", "Player", "Distance", "Speed", "xG", "Outcome"]
        shot_rows = [shot_headers]
        for s in shots:
            dist_str = f"{s['distance_m']:.1f} m" if s.get("distance_m") else "—"
            speed_str = f"{s['speed_mps']:.1f} m/s" if s.get("speed_mps") else "—"
            inferred_str = " (inferred)" if s.get("player_inferred") else ""
            shot_rows.append([
                f"{s['minute']}'",
                str(s.get("side", "")).upper(),
                f"{s['player']}{inferred_str}",
                dist_str,
                speed_str,
                f"{s['xg']:.2f}",
                str(s.get("outcome", "Shot")),
            ])
        shots_table = Table(shot_rows, colWidths=[12 * mm, 16 * mm, 54 * mm, 24 * mm, 24 * mm, 16 * mm, 28 * mm])
        shots_table.setStyle(TABLE_STYLE)
        story.append(shots_table)
    else:
        story.append(Paragraph("No direct goal attempts recorded in analyzed match frames.", note_style))

    # ================= PAGE 3: LINEUPS & TRACKING PROVENANCE =================
    story.append(PageBreak())
    story.append(Paragraph("Lineup & Player Tracking Performance", h2_style))

    story.append(Paragraph(f"<b>{match.home_team.name}</b>", styles["Heading3"]))
    story.append(_lineup_table(ctx.get("home_players", [])))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(f"<b>{match.away_team.name}</b>", styles["Heading3"]))
    story.append(_lineup_table(ctx.get("away_players", [])))
    story.append(Spacer(1, 5 * mm))

    # Data Provenance & Methodology Notes
    story.append(Paragraph("Data Methodology & Provenance", h2_style))
    story.append(Paragraph(
        "<b>Computer Vision Tracking:</b> Coordinates and physical distances are calculated via homography "
        "projection from multi-camera or broadcast tracking coordinates onto standardized FIFA pitch geometry (105m &times; 68m). "
        "<br/><b>Player Attribution:</b> 'Tracked' indicates confirmed player tracking association. 'Tracked (Auto)' represents "
        "automated jersey and visual track assignment pending manual review. "
        "<br/><b>Tactical Metrics:</b> Team compactness area reflects the 2D convex hull of outfield players. Centroids "
        "denote the geometric center of active outfield team shape.",
        foot_style,
    ))

    doc.build(story)
    buf.seek(0)

    report, _ = Report.objects.update_or_create(
        match=match,
        report_type=Report.ReportType.MATCH,
        defaults={
            "title": f"{match.home_team.short_name} vs {match.away_team.short_name} — Match Report",
            "generated_by_ai": True,
        },
    )
    report.pdf_file.save(f"match_{match.id}_report.pdf", ContentFile(buf.read()), save=True)
    return report