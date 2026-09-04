"""
Generates a PDF match report from apps.matches.views.build_match_report_context()
— the exact same data the results page (results.html) renders as HTML, so
the PDF can never show a number that disagrees with the website for the
same match.

Triggered automatically (see apps/matches/tasks.py:process_match and
:compute_pitch_mapping) — NOT behind a button. A report is generated the
moment a match finishes Stage 1-4 processing (using whatever data exists
at that point, real or Phase-5-dummy, same duality the results page
already handles), and regenerated whenever compute_pitch_mapping produces
real stats for it (initial calibration or a later recalibration).
update_or_create on (match, report_type=MATCH) — backed by a DB-level
unique_together on Report — means this always replaces the match's one
report in place rather than accumulating duplicates.
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

PITCH_GREEN = colors.HexColor("#1B4332")
ROW_TINT = colors.HexColor("#F4F6F5")
GRID_LINE = colors.HexColor("#CCCCCC")
NOTE_AMBER = colors.HexColor("#B8860B")

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PITCH_GREEN),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, GRID_LINE),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_TINT]),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
])


def _b64_heatmap_to_image(b64_string, width_mm=80):
    """
    Decodes a base64 PNG (from ai_engine.heatmap.render_heatmap_png) into
    a reportlab Image flowable, preserving the source aspect ratio rather
    than hardcoding a height that would stretch/squash it. Returns None
    if there's no image to embed (calibration hasn't produced one yet).
    """
    if not b64_string:
        return None

    from PIL import Image as PILImage

    data = base64.b64decode(b64_string)
    pil_img = PILImage.open(io.BytesIO(data))
    aspect = pil_img.height / pil_img.width
    width = width_mm * mm
    height = width * aspect

    # A fresh BytesIO for reportlab — kept alive via the Image flowable's
    # own reference until doc.build() runs, separate from the BytesIO PIL
    # already consumed above.
    return Image(io.BytesIO(data), width=width, height=height)


def _lineup_table(players):
    header = ["#", "Name", "Position", "Distance (km)"]
    data = [header]
    for p in players:
        distance_label = str(p["distance_km"])
        if p.get("distance_is_real"):
            distance_label += " (tracked, unverified)" if not p.get("distance_confirmed") else " (tracked)"
        data.append([p["jersey_number"], p["name"], p["position"], distance_label])

    table = Table(data, colWidths=[10 * mm, 70 * mm, 25 * mm, 35 * mm])
    table.setStyle(TABLE_STYLE)
    return table


def generate_match_report(match):
    """
    Builds the PDF and saves it to this match's Report (creating it the
    first time, replacing the file on every later call). Returns the
    Report instance. Never raises for "no real stats yet" or "no
    heatmap yet" — those render as an honest note in the PDF instead,
    same pattern as the results page's dummy-data badges.
    """
    from apps.matches.views import build_match_report_context

    ctx = build_match_report_context(match)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    note_style = ParagraphStyle("Note", parent=styles["Normal"], textColor=NOTE_AMBER, fontSize=9)

    story = []

    # --- Header ---
    story.append(Paragraph(
        f"{match.home_team.name} {match.home_score} - {match.away_score} {match.away_team.name}",
        styles["Title"],
    ))
    meta_bits = [match.match_date.strftime("%B %d, %Y")]
    if match.competition:
        meta_bits.append(match.competition)
    if match.stadium:
        meta_bits.append(match.stadium)
    story.append(Paragraph(" &bull; ".join(meta_bits), styles["Normal"]))
    story.append(Spacer(1, 4 * mm))

    if not ctx["using_real_stats"]:
        story.append(Paragraph(
            "Note: this match has not been calibrated yet. Team and player statistics "
            "below are preliminary estimates, not measurements from the video.",
            note_style,
        ))
        story.append(Spacer(1, 4 * mm))

    # --- Possession ---
    story.append(Paragraph("Possession", styles["Heading2"]))
    story.append(Paragraph(
        f"{match.home_team.short_name} {ctx['home_possession']}% &ndash; "
        f"{ctx['away_possession']}% {match.away_team.short_name}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 4 * mm))

    # --- Team stats table ---
    story.append(Paragraph("Team Statistics", styles["Heading2"]))
    home_ts, away_ts = ctx["team_stats"]["home"], ctx["team_stats"]["away"]
    rows = [
        ["", match.home_team.short_name, match.away_team.short_name],
        ["Shots", home_ts["shots"], away_ts["shots"]],
        ["Shots on Target", home_ts["shots_on_target"], away_ts["shots_on_target"]],
        ["Passes Completed", home_ts["passes"], away_ts["passes"]],
        ["Pass Accuracy", f"{home_ts['pass_accuracy']}%", f"{away_ts['pass_accuracy']}%"],
        ["Corners", home_ts["corners"], away_ts["corners"]],
        ["Fouls", home_ts["fouls"], away_ts["fouls"]],
        ["Yellow Cards", home_ts["yellow_cards"], away_ts["yellow_cards"]],
        ["Red Cards", home_ts["red_cards"], away_ts["red_cards"]],
        ["xG", home_ts["xg"], away_ts["xg"]],
        ["Distance (km)", home_ts["distance_km"], away_ts["distance_km"]],
    ]
    stats_table = Table(rows, colWidths=[50 * mm, 55 * mm, 55 * mm])
    stats_table.setStyle(TABLE_STYLE)
    story.append(stats_table)
    story.append(Spacer(1, 6 * mm))

    # --- Heatmaps ---
    home_img = _b64_heatmap_to_image(ctx["home_heatmap"])
    away_img = _b64_heatmap_to_image(ctx["away_heatmap"])
    story.append(Paragraph("Heatmaps", styles["Heading2"]))
    if home_img or away_img:
        col_data = [
            [Paragraph(match.home_team.short_name, styles["Normal"]),
             home_img if home_img else Paragraph("Not available", styles["Normal"])],
            [Paragraph(match.away_team.short_name, styles["Normal"]),
             away_img if away_img else Paragraph("Not available", styles["Normal"])],
        ]
        heatmap_table = Table([col_data], colWidths=[85 * mm, 85 * mm])
        heatmap_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(heatmap_table)
    else:
        story.append(Paragraph("Not available yet for this match — needs calibration.", note_style))
    story.append(Spacer(1, 6 * mm))

    # --- Lineup ---
    story.append(PageBreak())
    story.append(Paragraph("Lineup", styles["Heading2"]))
    story.append(Paragraph(match.home_team.name, styles["Heading3"]))
    story.append(_lineup_table(ctx["home_players"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(match.away_team.name, styles["Heading3"]))
    story.append(_lineup_table(ctx["away_players"]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Distance marked \"(tracked)\" or \"(tracked, unverified)\" comes from real pitch-mapped video "
        "tracking. \"Unverified\" means the player identification was auto-guessed (see this match's "
        "Identify Players page) and hasn't been checked by a human yet — the distance itself is real, "
        "but it may be attributed to the wrong name. Every other individual stat above is a placeholder "
        "estimate, not measured from video.",
        note_style,
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