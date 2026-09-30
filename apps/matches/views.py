import base64
import csv
from collections import defaultdict
import io
import json
import random
import tempfile
import os

import cv2
import numpy as np

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from apps.analytics.models import TeamStatistics
from apps.reports.models import Report

from apps.matches.models import Match, MatchCalibration, MatchGoal, MatchLineup, MatchVideo, TrackPlayerIdentification
from apps.matches.pitch_landmarks import PITCH_LANDMARKS
from apps.matches.pitch_landmarks_32 import ROBOFLOW_KEYPOINTS_32
from apps.matches.tasks import process_match, compute_pitch_mapping
from apps.players.models import Player
from apps.teams.models import Team

ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
LINEUP_ROWS_PER_SIDE = 18  # 11 starters + 7 subs, generous headroom
NEW_TEAM_SENTINEL = "__new__"  # <select> value that means "create a new team"


def _parse_lineup(request, prefix):
    """
    Reads the repeated lineup fields for one side (prefix is "home" or
    "away") and returns (rows, error). rows is a list of dicts, skipping
    any blank template rows the user didn't fill in.
    """
    names = request.POST.getlist(f"{prefix}_player_name")
    numbers = request.POST.getlist(f"{prefix}_jersey_number")
    positions = request.POST.getlist(f"{prefix}_position")

    rows = []
    seen_numbers = set()

    for name, number, position in zip(names, numbers, positions):
        name = name.strip()
        number = number.strip()

        if not name and not number:
            continue  # untouched template row

        if not name or not number:
            return None, f"Each {prefix} lineup row needs both a player name and a jersey number."

        if not number.isdigit():
            return None, f"Jersey number '{number}' in the {prefix} lineup must be a whole number."

        number = int(number)
        if number in seen_numbers:
            return None, f"Jersey number {number} is used twice in the {prefix} lineup."
        seen_numbers.add(number)

        if position not in dict(Player.Position.choices):
            position = Player.Position.MIDFIELDER

        rows.append({"player_name": name, "jersey_number": number, "position": position})

    return rows, None


def _resolve_registered_team(request, side_prefix, select_value):
    """
    Used only in "existing" (Registered Teams) mode. select_value is the
    raw value posted from the <select name="{side}_team">.

    Returns (team, new_team_data, error):
      - If an existing team was picked:  (Team instance, None, None)
      - If "+ Add New Team" was picked:  (None, dict-of-fields-to-create, None)
      - On any problem:                  (None, None, "error message")
    """
    if select_value == NEW_TEAM_SENTINEL:
        name = request.POST.get(f"new_{side_prefix}_team_name", "").strip()
        short_name = request.POST.get(f"new_{side_prefix}_team_short", "").strip().upper()
        country = request.POST.get(f"new_{side_prefix}_team_country", "").strip()
        league = request.POST.get(f"new_{side_prefix}_team_league", "").strip()
        team_type = request.POST.get(f"new_{side_prefix}_team_type", Team.TeamType.CLUB)
        primary_color = request.POST.get(f"new_{side_prefix}_team_primary_color", "").strip()
        secondary_color = request.POST.get(f"new_{side_prefix}_team_secondary_color", "").strip()
        logo = request.FILES.get(f"new_{side_prefix}_team_logo")

        if not name or not short_name or not primary_color:
            return None, None, (
                f"New {side_prefix} team needs at least a name, short name, and primary color."
            )

        if Team.objects.filter(name__iexact=name).exists() or Team.objects.filter(short_name__iexact=short_name).exists():
            return None, None, (
                f"A team named '{name}' or short name '{short_name}' already exists — pick it from the dropdown instead."
            )

        new_team_data = {
            "name": name,
            "short_name": short_name,
            "country": country,
            "league": league or None,
            "team_type": team_type,
            "primary_color": primary_color,
            "secondary_color": secondary_color or None,
            "logo": logo,
        }
        return None, new_team_data, None

    if not select_value:
        return None, None, f"Select {side_prefix} team, or choose '+ Add New Team'."

    try:
        team = Team.objects.get(pk=select_value)
    except Team.DoesNotExist:
        return None, None, f"Selected {side_prefix} team could not be found."

    return team, None, None


def _all_calibration_landmarks():
    """
    Merges the original 13-point PITCH_LANDMARKS preset with the 32-point
    Roboflow-aligned set, deduplicating by id. The 12 ids that appear in
    both are identical in coordinates (cross-checked when
    pitch_landmarks_32.py was built) — center_spot is the only
    PITCH_LANDMARKS entry with no Roboflow equivalent, since it's not one
    of the model's 32 detectable keypoints.

    Used for BOTH the calibration page's landmark dropdown and
    calibrate_save's validation, so a smart-assist-suggested point (which
    may use one of the 19 ids that aren't in the original 13) is always
    a valid, known landmark — never rejected as unrecognized just because
    it came from the larger set.
    """
    merged = {l["id"]: l for l in PITCH_LANDMARKS}
    for l in ROBOFLOW_KEYPOINTS_32:
        merged.setdefault(l["id"], l)
    return list(merged.values())


@login_required
def upload_match(request):
    teams = Team.objects.all().order_by("name")

    if request.method == "POST":
        match_mode = request.POST.get("match_mode", "existing")
        match_date = request.POST.get("match_date")
        competition = request.POST.get("competition", "").strip()
        stadium = request.POST.get("stadium", "").strip()
        video_file = request.FILES.get("video")

        if not match_date or not video_file:
            messages.error(request, "Match date and video are required.")
            return redirect("matches:upload")

        if not video_file.name.lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
            messages.error(request, "Please upload a video file (mp4, mov, avi, or mkv).")
            return redirect("matches:upload")

        home_team = away_team = None
        new_home_team_data = new_away_team_data = None
        home_kit_color = home_gk_kit_color = ""
        away_kit_color = away_gk_kit_color = ""

        # ---- Team A / Team B: registered (with optional inline "add new team") vs. local/custom ----
        if match_mode == "existing":
            home_team, new_home_team_data, error = _resolve_registered_team(
                request, "home", request.POST.get("home_team")
            )
            if error:
                messages.error(request, error)
                return redirect("matches:upload")

            away_team, new_away_team_data, error = _resolve_registered_team(
                request, "away", request.POST.get("away_team")
            )
            if error:
                messages.error(request, error)
                return redirect("matches:upload")

            # Compare identities even when one/both sides are brand-new teams
            # (which don't have a pk yet), using name as the comparison key.
            home_key = home_team.pk if home_team else new_home_team_data["name"].lower()
            away_key = away_team.pk if away_team else new_away_team_data["name"].lower()
            if home_key == away_key:
                messages.error(request, "Team A and Team B must be different.")
                return redirect("matches:upload")

            home_default_color = home_team.primary_color if home_team else new_home_team_data["primary_color"]
            away_default_color = away_team.primary_color if away_team else new_away_team_data["primary_color"]

            home_kit_color = request.POST.get("home_kit_color", "").strip() or home_default_color
            away_kit_color = request.POST.get("away_kit_color", "").strip() or away_default_color
            home_gk_kit_color = request.POST.get("home_gk_kit_color", "").strip()
            away_gk_kit_color = request.POST.get("away_gk_kit_color", "").strip()

        elif match_mode == "custom":
            home_name = request.POST.get("home_team_name", "").strip()
            home_short = request.POST.get("home_team_short", "").strip().upper()
            away_name = request.POST.get("away_team_name", "").strip()
            away_short = request.POST.get("away_team_short", "").strip().upper()

            home_kit_color = request.POST.get("home_kit_color", "").strip()
            home_gk_kit_color = request.POST.get("home_gk_kit_color", "").strip()
            away_kit_color = request.POST.get("away_kit_color", "").strip()
            away_gk_kit_color = request.POST.get("away_gk_kit_color", "").strip()

            if not all([home_name, home_short, away_name, away_short,
                        home_kit_color, home_gk_kit_color,
                        away_kit_color, away_gk_kit_color]):
                messages.error(
                    request,
                    "For a local/custom match, both team names, short names, and "
                    "all four kit colors (outfield + goalkeeper, home and away) are required."
                )
                return redirect("matches:upload")

            if home_name.lower() == away_name.lower():
                messages.error(request, "Team A and Team B must be different.")
                return redirect("matches:upload")

            if Team.objects.filter(name__iexact=home_name).exists() or Team.objects.filter(short_name__iexact=home_short).exists():
                messages.error(request, f"'{home_name}' already exists — pick it under Registered Teams instead.")
                return redirect("matches:upload")

            if Team.objects.filter(name__iexact=away_name).exists() or Team.objects.filter(short_name__iexact=away_short).exists():
                messages.error(request, f"'{away_name}' already exists — pick it under Registered Teams instead.")
                return redirect("matches:upload")

            new_home_team_data = {"name": home_name, "short_name": home_short, "primary_color": home_kit_color}
            new_away_team_data = {"name": away_name, "short_name": away_short, "primary_color": away_kit_color}

        else:
            messages.error(request, "Invalid match type.")
            return redirect("matches:upload")

        # ---- Optional lineups ----
        home_lineup, error = _parse_lineup(request, "home")
        if error:
            messages.error(request, error)
            return redirect("matches:upload")

        away_lineup, error = _parse_lineup(request, "away")
        if error:
            messages.error(request, error)
            return redirect("matches:upload")

        # ---- Create everything atomically so a bad row never leaves a half-saved match ----
        try:
            with transaction.atomic():
                if new_home_team_data:
                    home_team = Team.objects.create(**new_home_team_data)
                if new_away_team_data:
                    away_team = Team.objects.create(**new_away_team_data)

                match = Match.objects.create(
                    uploaded_by=request.user,
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                    stadium=stadium or None,
                    competition=competition or None,
                    status=Match.MatchStatus.PENDING,
                    home_kit_color=home_kit_color or None,
                    home_gk_kit_color=home_gk_kit_color or None,
                    away_kit_color=away_kit_color or None,
                    away_gk_kit_color=away_gk_kit_color or None,
                )

                MatchVideo.objects.create(match=match, original_video=video_file)

                for side, team, rows in (
                    (MatchLineup.Side.HOME, home_team, home_lineup),
                    (MatchLineup.Side.AWAY, away_team, away_lineup),
                ):
                    for row in rows:
                        # Auto-link to an existing roster Player if the
                        # team + jersey number already matches one.
                        linked_player = Player.objects.filter(
                            team=team, jersey_number=row["jersey_number"]
                        ).first()

                        MatchLineup.objects.create(
                            match=match,
                            team=team,
                            side=side,
                            player=linked_player,
                            player_name=row["player_name"],
                            jersey_number=row["jersey_number"],
                            position=row["position"],
                        )
        except IntegrityError:
            messages.error(
                request,
                "Something conflicted while saving (duplicate jersey number or team). Please check and try again.",
            )
            return redirect("matches:upload")

        # Queue background processing only after the transaction above has
        # actually committed — otherwise the Celery worker could try to
        # fetch this Match before it exists in the database.
        transaction.on_commit(lambda: process_match.delay(match.id))

        messages.success(
            request,
            "Match uploaded successfully. It's queued and will begin processing shortly.",
        )
        return redirect("matches:processing", public_id=match.public_id)

    context = {
        "teams": teams,
        "team_types": Team.TeamType.choices,
        "positions": Player.Position.choices,
        "lineup_row_range": range(LINEUP_ROWS_PER_SIDE),
    }
    return render(request, "matches/upload.html", context)


@login_required
def match_processing(request, public_id):
    """
    Shown right after upload. Polls match_status_api via JS and redirects
    to the results page automatically once status becomes COMPLETED.
    """
    match = get_object_or_404(Match, public_id=public_id)

    # If someone revisits this URL after it's already done/failed, just
    # send them straight to where they need to be instead of re-showing
    # a "processing" screen for a match that isn't processing anymore.
    if match.status == Match.MatchStatus.COMPLETED:
        return redirect("matches:results", public_id=match.public_id)

    return render(request, "matches/processing.html", {"match": match})


@login_required
def match_status_api(request, public_id):
    """
    Lightweight JSON endpoint the processing page polls. Deliberately
    tiny (no DRF) since this is the only place in the app that needs a
    JSON response - not worth adding a whole API layer for one field.
    """
    match = get_object_or_404(Match, public_id=public_id)
    return JsonResponse({
        "status": match.status,
        "status_display": match.get_status_display(),
        "progress": match.processing_progress,
        "results_url": (
            reverse("matches:results", args=[match.public_id])
            if match.status == Match.MatchStatus.COMPLETED else None
        ),
    })


def _load_real_stats_by_jersey(files_obj):
    """
    Reads MatchFiles.player_stats_csv (written by compute_pitch_mapping)
    and returns {jersey_number: stats_dict} for each team, but ONLY for
    tracklets jersey OCR confidently identified (jersey_number non-blank
    there already means aggregate_jersey_number's multi-read-agreement
    bar was cleared — see ai_engine/stage3_team_reid/jersey_ocr.py).
    stats_dict["is_confirmed"] is always True here — there's no separate
    human-review step for an OCR match, unlike the manual/auto-guess
    path in _load_real_stats_by_assignment.

    stats_dict keys: distance_km, passes_completed, shots, xg,
    is_confirmed. Each numeric field defaults to 0/0.0 if the CSV
    predates that column (older matches / not yet recalibrated) rather
    than failing the whole row.

    NOTE on what's real vs. inferred here (see tasks.py for where each
    comes from):
      - distance_km: real, pitch-mapped.
      - passes_completed: real COMPLETED-pass count. There is
        deliberately no matching "attempted"/"accuracy" figure anywhere
        (detect_passes never emits a failed-pass event) — don't derive
        one.
      - shots / xg: shots is a real detected-shot count, but ATTRIBUTED
        to this player via closest-identity-to-ball inference at the
        shot frame (detect_shots' Event carries no player_master_id at
        all) — an inference, not a directly-detected fact, same caveat
        as team-level shot counting already had. xg is a simplified
        distance/angle heuristic (estimate_shot_xg in events.py), NOT a
        trained model.

    Deliberately narrow beyond that: this does NOT return goals/assists/
    etc — none of that is real yet. _dummy_player_rows() uses this dict
    to override just these fields for players it can match, leaving
    everything else dummy. Conflating real fields with fake goals/
    assists on the same row would look more trustworthy than it is if
    not clearly flagged — see the *_is_real flags in that function.

    Returns dicts (not a growing tuple) specifically because this
    structure has already grown twice in two sessions — see the tuple-
    migration crash noted in project history. A dict means adding a
    field later never requires touching every unpacking call site.
    """
    home_by_jersey, away_by_jersey = {}, {}

    if not files_obj or not files_obj.player_stats_csv:
        return home_by_jersey, away_by_jersey

    try:
        with files_obj.player_stats_csv.open("rb") as f:
            content = f.read().decode("utf-8")
    except Exception:
        return home_by_jersey, away_by_jersey

    from ai_engine.utils.types import Team

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "jersey_number" not in reader.fieldnames:
        return home_by_jersey, away_by_jersey

    for row in reader:
        jersey_raw = row.get("jersey_number")
        distance_raw = row.get("distance_m")
        if not jersey_raw or not distance_raw:
            continue
        try:
            jersey_number = int(jersey_raw)
            distance_km = float(distance_raw) / 1000.0
        except ValueError:
            continue

        stats = {
            "distance_km": distance_km,
            "passes_completed": _safe_int(row.get("passes_completed")),
            "passes_attempted": _safe_int(row.get("passes_attempted")),
            "pass_accuracy": _safe_float(row.get("pass_accuracy")),
            "shots": _safe_int(row.get("shots")),
            "shots_on_target": _safe_int(row.get("shots_on_target")),
            "xg": _safe_float(row.get("xg")),
            "top_speed": _safe_float(row.get("top_speed")),
            "average_speed": _safe_float(row.get("average_speed")),
            "tackles": _safe_int(row.get("tackles")),
            "interceptions": _safe_int(row.get("interceptions")),
            "clearances": _safe_int(row.get("clearances")),
            "dribbles_completed": _safe_int(row.get("dribbles_completed")),
            "key_passes": _safe_int(row.get("key_passes")),
            "rating": _safe_float(row.get("rating"), default=6.0),
            "is_confirmed": True,
        }

        if row.get("team") == Team.TEAM_A.value:
            home_by_jersey[jersey_number] = stats
        elif row.get("team") == Team.TEAM_B.value:
            away_by_jersey[jersey_number] = stats

    return home_by_jersey, away_by_jersey


def _safe_int(raw, default=0):
    """int(raw) that tolerates None/'' and older CSVs missing the column
    entirely, instead of raising and dropping the whole row."""
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _safe_float(raw, default=0.0):
    """Float counterpart to _safe_int, same reasoning."""
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default



def _dummy_player_rows(rng, lineup_qs, team, real_stats_by_jersey=None, real_goals_by_lineup_id=None):
    """
    Phase 5: placeholder stats only, to verify the results UI before the
    real CV pipeline (Phase 6) fills these in. Uses real lineup entries
    if they exist (Phase 3 optional lineup), otherwise generic rows.
    rng is seeded per-match so the same match shows the same numbers on
    every reload rather than reshuffling each time.

    real_stats_by_jersey (optional): {jersey_number: stats_dict} from
    _load_real_stats_by_jersey / _load_real_stats_by_assignment — see
    either's docstring for exactly what's in stats_dict and how real
    vs. inferred each field is. When a lineup player's jersey number is
    in there, distance_km / passes_completed / shots / xg on their row
    are all overridden with real values, and distance_is_real /
    passes_completed_is_real / shots_is_real / xg_is_real are set True.
    distance_confirmed reflects whether a human explicitly chose this
    identification (is_confirmed=True) or it's still an unreviewed
    auto-guess from _auto_assign_unidentified_tracks (is_confirmed=False)
    — the SAME confirmation status applies to every one of those four
    fields, since they all come from the same TrackPlayerIdentification
    row; distance_confirmed is reused as the shared flag rather than
    repeating it four times.

    passes_attempted and pass_accuracy stay fully dummy even when
    passes_completed is real — detect_passes never emits a failed-pass
    event, so a real "attempted"/"accuracy" figure isn't derivable.
    shots_on_target stays fully dummy even when shots is real — shot
    detection doesn't classify on-target vs. off/blocked. Don't fake
    either. Every OTHER field on the row is still the Phase 5 dummy
    generator's output. lineup_entry_id is always included (real, from
    the DB) — that's what views.player_heatmap keys on to find this
    player's tracked identity, not track_id.

    Field names deliberately mirror apps.analytics.models.PlayerStatistics
    so swapping this out for real queries later is a drop-in replacement.

    real_goals_by_lineup_id (optional): {lineup_entry_id: goal_count}
    from MatchGoal (see that model — manually entered via the results
    page's "Manage Goals" panel, not detected). Unlike the CV-pipeline
    fields above, this isn't gated on whether a player has a tracked
    identity — a goal is entered directly against a MatchLineup row, so
    it's real (and shown with the same tracked label) for EVERY player
    on the lineup the moment at least one goal has been logged for this
    match, not just ones with a tracklet. That's the signal used here:
    pass None (the default) when NO MatchGoal rows exist yet for this
    match at all (falls back to the Phase 5 dummy "goals" figure for
    every player, as before); pass an actual dict — even one with zero
    entries — the moment ANY goal has been logged, so every player's
    goals become real (0 for non-scorers, N for scorers). Only counts
    is_own_goal=False rows toward a player's personal tally, matching
    standard football statistics convention.
    """
    real_stats_by_jersey = real_stats_by_jersey or {}
    rows = []
    entries = list(lineup_qs)

    if not entries:
        entries = [
            {"jersey_number": i, "player_name": f"{team.short_name} Player {i}",
             "position": rng.choice(["GK", "DEF", "MID", "FWD"])}
            for i in range(1, 12)
        ]

    for entry in entries:
        is_dict = isinstance(entry, dict)
        position = entry["position"] if is_dict else entry.position
        name = entry["player_name"] if is_dict else entry.player_name
        jersey_number = entry["jersey_number"] if is_dict else entry.jersey_number
        lineup_entry_id = None if is_dict else entry.id

        passes_attempted = rng.randint(15, 75)
        passes_completed = int(passes_attempted * rng.uniform(0.65, 0.95))
        pass_accuracy = round((passes_completed / passes_attempted) * 100, 1) if passes_attempted else 0
        shots = rng.randint(0, 6)
        shots_on_target = rng.randint(0, 3)
        xg = round(rng.uniform(0, 1.2), 2)

        real_stats = real_stats_by_jersey.get(jersey_number)
        if real_stats is not None:
            distance_km = round(real_stats["distance_km"], 2)
            passes_completed = real_stats["passes_completed"]
            passes_attempted = real_stats["passes_attempted"]
            pass_accuracy = real_stats["pass_accuracy"]
            shots = real_stats["shots"]
            shots_on_target = real_stats["shots_on_target"]
            xg = real_stats["xg"]
            top_speed = real_stats["top_speed"]
            average_speed = real_stats["average_speed"]
            tackles = real_stats["tackles"]
            interceptions = real_stats["interceptions"]
            clearances = real_stats["clearances"]
            dribbles_completed = real_stats["dribbles_completed"]
            key_passes = real_stats["key_passes"]
            rating = real_stats["rating"]

            distance_confirmed = real_stats["is_confirmed"]
            distance_is_real = True
            passes_completed_is_real = True
            passes_attempted_is_real = True
            pass_accuracy_is_real = True
            shots_is_real = True
            shots_on_target_is_real = True
            xg_is_real = True
            top_speed_is_real = True
            average_speed_is_real = True
            tackles_is_real = True
            interceptions_is_real = True
            clearances_is_real = True
            dribbles_completed_is_real = True
            key_passes_is_real = True
            rating_is_real = True
        else:
            distance_km = round(rng.uniform(7.5, 11.8), 2)
            top_speed = round(rng.uniform(24, 34), 1)
            average_speed = round(rng.uniform(7, 11), 1)
            tackles = rng.randint(0, 6)
            interceptions = rng.randint(0, 5)
            clearances = rng.randint(0, 8) if position in ("DEF", "GK") else rng.randint(0, 2)
            dribbles_completed = rng.randint(0, 5)
            key_passes = rng.randint(0, 4)
            rating = round(rng.uniform(5.8, 8.9), 1)

            distance_confirmed = False
            distance_is_real = False
            passes_completed_is_real = False
            passes_attempted_is_real = False
            pass_accuracy_is_real = False
            shots_is_real = False
            shots_on_target_is_real = False
            xg_is_real = False
            top_speed_is_real = False
            average_speed_is_real = False
            tackles_is_real = False
            interceptions_is_real = False
            clearances_is_real = False
            dribbles_completed_is_real = False
            key_passes_is_real = False
            rating_is_real = False

        if real_goals_by_lineup_id is not None:
            # A dict (even empty) means at least one goal has been
            # logged for this MATCH — see this function's docstring.
            # Every player's goal count is real from that point on,
            # regardless of whether they individually have a tracked
            # identity (goals are entered against the lineup directly).
            goals = real_goals_by_lineup_id.get(lineup_entry_id, 0)
            goals_is_real = True
            goals_confirmed = True  # always human-entered — no auto-guess path exists for goals
        else:
            goals = rng.choice([0, 0, 0, 0, 1, 1, 2]) if position == "FWD" else rng.choice([0, 0, 0, 0, 1])
            goals_is_real = False
            goals_confirmed = False

        # Strict consistency: Goals cannot exceed shots or shots on target, and xG must reflect goals
        if goals > 0:
            if shots < goals:
                shots = goals
                shots_is_real = True
            if shots_on_target < goals:
                shots_on_target = goals
                shots_on_target_is_real = True
            if xg < round(goals * 0.20, 2):
                xg = round(max(xg, goals * 0.20), 2)
                xg_is_real = True
            if rating < 7.2:
                rating = round(7.2 + min(goals * 0.8, 2.3), 1)

        is_starting = entry.get("is_starting", True) if is_dict else getattr(entry, "is_starting", True)

        rows.append({
            "jersey_number": jersey_number,
            "name": name,
            "lineup_entry_id": lineup_entry_id,
            "position": position,
            "is_starting": is_starting,
            "minutes_played": rng.choice([90, 90, 90, rng.randint(60, 89)]),
            "goals": goals,
            "goals_is_real": goals_is_real,
            "goals_confirmed": goals_confirmed,
            "assists": rng.choice([0, 0, 0, 1]),
            "shots": shots,
            "shots_is_real": shots_is_real,
            "shots_on_target": shots_on_target,
            "shots_on_target_is_real": shots_on_target_is_real,
            "passes_attempted": passes_attempted,
            "passes_attempted_is_real": passes_attempted_is_real,
            "passes_completed": passes_completed,
            "passes_completed_is_real": passes_completed_is_real,
            "pass_accuracy": pass_accuracy,
            "pass_accuracy_is_real": pass_accuracy_is_real,
            "key_passes": key_passes,
            "key_passes_is_real": key_passes_is_real,
            "dribbles_completed": dribbles_completed,
            "dribbles_completed_is_real": dribbles_completed_is_real,
            "tackles": tackles,
            "tackles_is_real": tackles_is_real,
            "interceptions": interceptions,
            "interceptions_is_real": interceptions_is_real,
            "clearances": clearances,
            "clearances_is_real": clearances_is_real,
            "fouls_committed": rng.randint(0, 4),
            "fouls_suffered": rng.randint(0, 4),
            "yellow_cards": rng.choice([0, 0, 0, 0, 1]),
            "red_cards": 0,
            "offsides": rng.randint(0, 3) if position == "FWD" else 0,
            "distance_km": distance_km,
            "distance_is_real": distance_is_real,
            "distance_confirmed": distance_confirmed,
            "top_speed": top_speed,
            "top_speed_is_real": top_speed_is_real,
            "average_speed": average_speed,
            "average_speed_is_real": average_speed_is_real,
            "xg": xg,
            "xg_is_real": xg_is_real,
            "rating": rating,
            "rating_is_real": rating_is_real,
        })
    return rows



def _load_pitch_positions_by_team(files_obj):
    """
    Reads the pitch_x/pitch_y columns out of the saved
    player_tracking_csv (added alongside the pixel bboxes in
    tasks.py) and buckets them by team, for the heatmap.

    Rows without a homography for that frame have blank pitch_x/pitch_y
    (see tasks.py) and are skipped rather than guessed at. Matches
    uploaded before this change won't have the pitch_x/pitch_y columns
    at all — that's treated the same as "no data", not an error, so
    match_results() can fall back to the placeholder pitch.
    """
    home_positions, away_positions = [], []

    if not files_obj or not files_obj.player_tracking_csv:
        return home_positions, away_positions

    try:
        with files_obj.player_tracking_csv.open("rb") as f:
            content = f.read().decode("utf-8")
    except Exception:
        return home_positions, away_positions

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "pitch_x" not in reader.fieldnames:
        return home_positions, away_positions  # older CSV, no pitch columns yet

    from ai_engine.utils.types import Team

    for row in reader:
        px, py = row.get("pitch_x"), row.get("pitch_y")
        if not px or not py:
            continue
        try:
            point = (float(px), float(py))
        except ValueError:
            continue

        if row.get("team") == Team.TEAM_A.value:
            home_positions.append(point)
        elif row.get("team") == Team.TEAM_B.value:
            away_positions.append(point)

    return home_positions, away_positions


def _load_pitch_positions_for_track(files_obj, track_id):
    """
    Same source data as _load_pitch_positions_by_team, but filtered to
    ONE stitched track_id — for a single player's heatmap instead of a
    whole team's. Used by player_heatmap(), called on demand when a
    player's modal opens (see results.html) rather than precomputed for
    every player on page load — generating up to 20+ heatmap PNGs
    synchronously on every results-page view would be wasteful when a
    person typically only opens a couple of player modals.
    """
    positions = []

    if not files_obj or not files_obj.player_tracking_csv:
        return positions

    try:
        with files_obj.player_tracking_csv.open("rb") as f:
            content = f.read().decode("utf-8")
    except Exception:
        return positions

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "pitch_x" not in reader.fieldnames:
        return positions

    for row in reader:
        try:
            if int(row["track_id"]) != track_id:
                continue
        except (KeyError, ValueError):
            continue
        px, py = row.get("pitch_x"), row.get("pitch_y")
        if not px or not py:
            continue
        try:
            positions.append((float(px), float(py)))
        except ValueError:
            continue

    return positions


@login_required
def calibrate_match(request, public_id):
    """
    Hidden manual/smart-assist calibration page — NOT linked from
    results.html. Calibration now happens automatically for every match
    (see apps/matches/tasks.py:_run_automatic_calibration), with no
    human step. This page exists purely as a developer/debug fallback
    for when automatic calibration skipped a match (fewer than 4
    confident keypoints) or produced visibly wrong stats — reachable
    only by knowing this URL. Only the uploader can calibrate their own
    match (unlike match_results, which anyone logged in can view) — a
    bad-faith or mistaken calibration silently corrupts real stats for
    everyone else looking at this match.
    """
    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    default_frame_idx = 0
    video = getattr(match, "video", None)
    if video and video.original_video:
        try:
            cap = cv2.VideoCapture(video.original_video.path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames > 0:
                default_frame_idx = total_frames // 2
        except Exception:
            pass  # fall back to frame 0 — the frame-loading UI lets the user pick a different one anyway

    existing_calibrations = list(match.calibrations.order_by("calibration_frame"))
    has_real_stats = TeamStatistics.objects.filter(match=match).exists()

    all_landmarks = _all_calibration_landmarks()
    context = {
        "match": match,
        "default_frame_idx": default_frame_idx,
        "existing_calibrations": existing_calibrations,
        "has_real_stats": has_real_stats,
        "landmarks": all_landmarks,
        "landmarks_json": json.dumps(all_landmarks),
        "roboflow_configured": bool(settings.ROBOFLOW_API_KEY),
    }
    return render(request, "matches/calibrate.html", context)


@login_required
def calibrate_frame(request, public_id):
    """
    Serves one frame of the match's video as a JPEG, on demand — nothing
    is stored. Used by the calibration page's <img> so the user can pick
    pixel points on an actual frame.
    """
    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    video = getattr(match, "video", None)
    if not video or not video.original_video:
        return HttpResponseBadRequest("No video uploaded for this match.")

    try:
        frame_idx = int(request.GET.get("frame_idx", 0))
    except (TypeError, ValueError):
        return HttpResponseBadRequest("frame_idx must be an integer.")

    cap = cv2.VideoCapture(video.original_video.path)
    if not cap.isOpened():
        cap.release()
        return HttpResponseBadRequest("Could not open the match video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return HttpResponseBadRequest("Could not read that frame from the video.")

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return HttpResponseBadRequest("Could not encode that frame as JPEG.")

    response = HttpResponse(buf.tobytes(), content_type="image/jpeg")
    response["X-Frame-Idx"] = str(frame_idx)  # actual clamped frame, in case the requested one was out of range
    return response


@login_required
def calibrate_suggest(request, public_id):
    """
    Smart-assist for the HIDDEN manual /calibrate/ page only — the real,
    no-human-involved automatic calibration path is
    apps/matches/tasks.py:_run_automatic_calibration, which runs for
    every match on its own. This endpoint exists so that when automatic
    calibration skipped a match (or got a bad frame) and someone opens
    this hidden debug page to fix it by hand, they can still get
    suggested points on a DIFFERENT frame than the one auto-calibration
    tried, rather than placing all points manually from scratch.

    Nothing is saved here — purely advisory. calibrate_save still does
    its own full validation regardless of what this endpoint suggested.

    Shares its actual detection logic with _run_automatic_calibration
    via ai_engine/stage5_pitch_mapping/smart_assist.py:
    detect_pitch_keypoints — kept in one place so the class_id+1 mapping
    fix (see pitch_landmarks_32.py's docstring) can't drift out of sync
    between the automatic and manual paths.
    """
    from ai_engine.stage5_pitch_mapping.smart_assist import detect_pitch_keypoints

    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    if not settings.ROBOFLOW_API_KEY:
        return JsonResponse({"error": "Roboflow API key is not configured on the server."}, status=503)

    video = getattr(match, "video", None)
    if not video or not video.original_video:
        return HttpResponseBadRequest("No video uploaded for this match.")

    try:
        frame_idx = int(request.GET.get("frame_idx", 0))
    except (TypeError, ValueError):
        return HttpResponseBadRequest("frame_idx must be an integer.")

    try:
        confidence_threshold = float(request.GET.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence_threshold = 0.5

    cap = cv2.VideoCapture(video.original_video.path)
    if not cap.isOpened():
        cap.release()
        return HttpResponseBadRequest("Could not open the match video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return HttpResponseBadRequest("Could not read that frame from the video.")

    suggestions = detect_pitch_keypoints(
        frame, settings.ROBOFLOW_API_KEY, confidence_threshold=confidence_threshold, max_points=12
    )

    if not suggestions:
        return JsonResponse({"error": "No confident pitch keypoints detected in this frame."}, status=422)

    return JsonResponse({"suggestions": suggestions, "frame_idx": frame_idx})


@login_required
def calibrate_save(request, public_id):
    """
    Validates and saves ONE MatchCalibration anchor, then kicks off
    compute_pitch_mapping (which re-reads ALL of this match's anchors,
    not just this one — see that function's docstring). A match can have
    multiple anchors (see MatchCalibration's docstring — this is how
    camera pan/zoom is handled); saving at a calibration_frame that
    already has an anchor EDITS that anchor in place rather than
    creating a duplicate, saving at a new frame ADDS a new anchor.

    If this match already has real TeamStatistics (i.e. it was
    calibrated before) and the request isn't explicitly confirming an
    overwrite, responds with needs_confirmation instead of saving — the
    calibration page's JS shows a confirm() dialog and resubmits with
    confirm_overwrite=true. Applies to adding a new anchor too, not just
    editing an existing one, since ANY change here triggers a full
    recompute of every anchor's homography together, not an incremental
    per-anchor update.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required.")

    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    try:
        payload = json.loads(request.body)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    frame_idx = payload.get("calibration_frame")
    raw_points = payload.get("points")
    confirm_overwrite = bool(payload.get("confirm_overwrite"))

    if not isinstance(frame_idx, int) or frame_idx < 0:
        return JsonResponse({"error": "calibration_frame must be a non-negative integer."}, status=400)

    if not isinstance(raw_points, list) or len(raw_points) < 4:
        return JsonResponse({"error": "At least 4 points are required."}, status=400)

    landmarks_by_id = {landmark["id"]: landmark for landmark in _all_calibration_landmarks()}
    points, seen_landmarks = [], set()
    for p in raw_points:
        landmark_id = p.get("landmark_id") if isinstance(p, dict) else None
        landmark = landmarks_by_id.get(landmark_id)
        if landmark is None:
            return JsonResponse({"error": f"Unknown landmark: {landmark_id!r}"}, status=400)
        if landmark_id in seen_landmarks:
            return JsonResponse({"error": "Each landmark can only be used once."}, status=400)
        seen_landmarks.add(landmark_id)

        try:
            pixel_x, pixel_y = float(p["pixel_x"]), float(p["pixel_y"])
        except (KeyError, TypeError, ValueError):
            return JsonResponse({"error": "Each point needs numeric pixel_x/pixel_y."}, status=400)

        points.append({
            "landmark_id": landmark_id,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "pitch_x": landmark["x"],
            "pitch_y": landmark["y"],
        })

    has_real_stats = TeamStatistics.objects.filter(match=match).exists()
    if has_real_stats and not confirm_overwrite:
        return JsonResponse({"needs_confirmation": True})

    with transaction.atomic():
        MatchCalibration.objects.update_or_create(
            match=match,
            calibration_frame=frame_idx,
            defaults={"points": points},
        )
    try:
        compute_pitch_mapping(match.id)
    except Exception:
        logger.exception("compute_pitch_mapping failed in calibrate_save for match=%s", match.id)

    return JsonResponse({"ok": True, "redirect_url": reverse("matches:results", args=[match.public_id])})


@login_required
def delete_calibration(request, public_id, calibration_id):
    """
    Removes one calibration anchor. If OTHER anchors remain, their
    segments are unaffected in terms of WHICH frames they cover — see
    MatchCalibration's docstring — removing an anchor just means its own
    segment (and any frames that were only reachable because it existed)
    goes back to having no real pitch mapping once compute_pitch_mapping
    re-runs, same "honest gap" pattern as everywhere else in this
    project rather than trying to paper over it. Deleting the LAST
    remaining anchor leaves the match with no calibration at all —
    compute_pitch_mapping's next run then does nothing (see its early
    return for that case) rather than clearing out already-written real
    stats.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required.")

    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)
    calibration = get_object_or_404(MatchCalibration, id=calibration_id, match=match)
    calibration.delete()

    if match.calibrations.exists():
        try:
            compute_pitch_mapping(match.id)
        except Exception:
            logger.exception("compute_pitch_mapping failed in delete_calibration for match=%s", match.id)

    return JsonResponse({"ok": True, "redirect_url": reverse("matches:calibrate", args=[match.public_id])})


def _load_track_summaries(files_obj, match=None, cap_per_team=11):
    """
    Reads MatchFiles.player_stats_csv (written by compute_pitch_mapping,
    one row per STITCHED/merged tracklet — see that function's comments
    on why raw fragmented tracklets aren't used here) and returns a list
    of {track_id, team, distance_km, frames_tracked}, one per real
    player-shaped tracklet. Referee rows are excluded — they're never
    assignable to a MatchLineup entry.

    Capped to the cap_per_team MOST-SEEN (highest frames_tracked)
    identities per team — for a clean, professor-presentable identify
    page instead of every residual tracklet fragment (raw fragmentation
    typically outnumbers the real ~11 players per side). Any track_id
    that already has a TrackPlayerIdentification (guessed OR human-
    confirmed) is always kept regardless of rank, so capping can never
    silently orphan an existing identification — remaining slots (if
    any) fill with the next-most-seen candidates. Pass match=None to
    skip capping entirely (e.g. for non-display callers that want every
    row, like the distance-lookup helpers).
    """
    from ai_engine.utils.types import Team

    summaries = []
    if not files_obj or not files_obj.player_stats_csv:
        return summaries

    try:
        with files_obj.player_stats_csv.open("rb") as f:
            content = f.read().decode("utf-8")
    except Exception:
        return summaries

    for row in csv.DictReader(io.StringIO(content)):
        if row.get("team") not in (Team.TEAM_A.value, Team.TEAM_B.value):
            continue
        try:
            summaries.append({
                "track_id": int(row["track_id"]),
                "team": row["team"],
                "distance_km": round(float(row["distance_m"]) / 1000.0, 2),
                "frames_tracked": int(row["frames_tracked"]),
            })
        except (KeyError, ValueError):
            continue

    if match is None:
        return summaries

    # Confirmed track_ids are NEVER dropped by the cap below — a human's
    # correction must always stay visible. Guessed ones only fill
    # whatever budget confirmed ones don't use.
    confirmed_track_ids = set(
        TrackPlayerIdentification.objects.filter(match=match, is_auto_assigned=False)
        .values_list("track_id", flat=True)
    )
    guessed_track_ids = set(
        TrackPlayerIdentification.objects.filter(match=match, is_auto_assigned=True)
        .values_list("track_id", flat=True)
    )

    capped = []
    for team_value in (Team.TEAM_A.value, Team.TEAM_B.value):
        team_summaries = [s for s in summaries if s["team"] == team_value]

        # Confirmed always kept, even past the cap — this is the one
        # case where MORE than cap_per_team cards can show for a side
        # (better than hiding a human's own correction).
        keep_ids = {s["track_id"] for s in team_summaries if s["track_id"] in confirmed_track_ids}

        # Guessed ones compete for whatever's left of the cap, ranked by
        # frames_tracked — this is also what fixes the earlier bug where
        # a side with existing confirmed rows could end up showing 11
        # instead of 10: previously ALL already-identified track_ids
        # (guessed AND confirmed) were kept unconditionally regardless
        # of how many that added up to. Only confirmed gets that
        # unconditional treatment now.
        guessed_candidates = sorted(
            (s for s in team_summaries if s["track_id"] in guessed_track_ids and s["track_id"] not in keep_ids),
            key=lambda s: -s["frames_tracked"],
        )
        slots_left = max(cap_per_team - len(keep_ids), 0)
        keep_ids |= {s["track_id"] for s in guessed_candidates[:slots_left]}

        # Never-identified candidates fill any slots still left over.
        remaining = sorted(
            (s for s in team_summaries if s["track_id"] not in keep_ids),
            key=lambda s: -s["frames_tracked"],
        )
        slots_left = max(cap_per_team - len(keep_ids), 0)
        keep_ids |= {s["track_id"] for s in remaining[:slots_left]}
        capped.extend(s for s in team_summaries if s["track_id"] in keep_ids)

    return capped


def _extract_track_crops_base64(video_path, player_tracking_csv_content, track_ids, candidates_per_track=3):
    """
    Extracts one representative (sharpest-of-a-few-largest) crop per
    track_id, using a SINGLE opened video for all of them, and returns
    {track_id: base64_jpeg_or_None}.

    IMPORTANT: this exists specifically to avoid opening the video file
    once per track. The first version of this feature served each crop
    from its own endpoint, so a match with N tracked players fired N
    concurrent HTTP requests on page load, each independently opening
    and seeking the video — for N=29 that's ~29 separate video decode
    contexts hitting the container at once, which crashed the web
    service (ERR_CONNECTION_RESET) on real testing. One shared
    VideoCapture, used sequentially, is dramatically lighter — still not
    fast (seeking is seeking), but it's one controlled pass instead of a
    concurrency spike.
    """
    detections_by_track = defaultdict(list)
    for row in csv.DictReader(io.StringIO(player_tracking_csv_content)):
        try:
            track_id = int(row["track_id"])
        except (KeyError, ValueError):
            continue
        if track_id not in track_ids:
            continue
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        detections_by_track[track_id].append((int(row["frame_idx"]), x1, y1, x2, y2))

    crops_b64 = {tid: None for tid in track_ids}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return crops_b64

    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0

    try:
        for track_id in track_ids:
            dets = detections_by_track.get(track_id, [])
            if not dets:
                continue

            # Sample up to 15 evenly spaced candidate frames across the tracklet's lifetime
            indices = np.linspace(0, len(dets) - 1, min(15, len(dets)), dtype=int)
            candidates = [dets[i] for i in indices]
            candidates.sort(key=lambda d: d[0])

            best_crop = None
            best_score = -1.0

            for frame_idx, x1, y1, x2, y2 in candidates:
                w = x2 - x1
                h = y2 - y1
                if w <= 8.0 or h <= 15.0:
                    continue

                # Upright human posture prior (ideal aspect ratio h/w ~ 2.2)
                aspect = h / max(w, 1.0)
                aspect_weight = float(np.exp(-0.5 * ((aspect - 2.2) / 0.7) ** 2))

                # Boundary penalty (penalize truncated players touching frame borders)
                edge_penalty = 1.0
                if x1 < 8.0 or y1 < 8.0 or x2 > (frame_w - 8.0) or y2 > (frame_h - 8.0):
                    edge_penalty = 0.25

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue

                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(int(frame_w), int(x2)), min(int(frame_h), int(y2))
                crop = frame[yi1:yi2, xi1:xi2]
                if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 8:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                # Composite visual quality score
                score = lap_var * aspect_weight * edge_penalty * np.log1p(h)
                if score > best_score:
                    best_score = score
                    best_crop = crop

            if best_crop is not None:
                # Enhance visual rendering: high-quality Lanczos scaling for card display
                ch, cw = best_crop.shape[:2]
                if ch < 200 and cw > 10:
                    scale = 200.0 / ch
                    nw, nh = int(cw * scale), 200
                    best_crop = cv2.resize(best_crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
                    # Subtle unsharp mask for crisp texture
                    gaussian = cv2.GaussianBlur(best_crop, (0, 0), 1.5)
                    best_crop = cv2.addWeighted(best_crop, 1.25, gaussian, -0.25, 0)

                ok, buf = cv2.imencode(".jpg", best_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok:
                    crops_b64[track_id] = base64.b64encode(buf.tobytes()).decode("ascii")
    finally:
        cap.release()

    return crops_b64


@login_required
def identify_players(request, public_id):
    """
    Manual player-identification page: one representative crop per
    tracked identity (stitched tracklet), with a dropdown to assign it
    to a real MatchLineup entry. See TrackPlayerIdentification's
    docstring for why this exists instead of relying on jersey OCR.

    Crops are extracted server-side, once, in a single video pass (see
    _extract_track_crops_base64) and embedded directly as base64 images
    — NOT served from a separate per-track endpoint. See that function's
    docstring for why: a per-track endpoint means N concurrent requests
    on page load, which crashed the web service on real testing.

    Only the uploader can identify players on their own match, same
    restriction as calibration — a mistaken assignment silently
    corrupts which name real stats show up under for everyone viewing
    this match.
    """
    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    from ai_engine.utils.types import Team

    tracks = _load_track_summaries(getattr(match, "files", None), match=match)

    home_lineup = list(match.lineups.filter(side=MatchLineup.Side.HOME).order_by("jersey_number").values("id", "jersey_number", "player_name", "position"))
    away_lineup = list(match.lineups.filter(side=MatchLineup.Side.AWAY).order_by("jersey_number").values("id", "jersey_number", "player_name", "position"))

    existing = {
        ident.track_id: ident
        for ident in TrackPlayerIdentification.objects.filter(match=match)
    }

    files_obj = getattr(match, "files", None)
    video = getattr(match, "video", None)
    crops_b64 = {}
    if tracks and files_obj and files_obj.player_tracking_csv and video and video.original_video:
        with files_obj.player_tracking_csv.open("rb") as f:
            csv_content = f.read().decode("utf-8")
        track_ids = {t["track_id"] for t in tracks}
        crops_b64 = _extract_track_crops_base64(video.original_video.path, csv_content, track_ids)

    from ai_engine.stage5_pitch_mapping.formation_matcher import match_tracks_to_lineup_hungarian

    home_tracks = [t for t in tracks if t["team"] == Team.TEAM_A.value]
    away_tracks = [t for t in tracks if t["team"] == Team.TEAM_B.value]

    home_lineup_objs = list(match.lineups.filter(side=MatchLineup.Side.HOME).order_by("jersey_number"))
    away_lineup_objs = list(match.lineups.filter(side=MatchLineup.Side.AWAY).order_by("jersey_number"))

    _, home_top_cand = match_tracks_to_lineup_hungarian(
        [{"track_id": t["track_id"], "median_x": t.get("median_x", 0.0), "median_y": t.get("median_y", 0.0), "duration": t.get("frames_tracked", 100)} for t in home_tracks],
        home_lineup_objs,
        formation_name=getattr(match, "home_formation", "4-3-3"),
        defending_left=True,
    )
    _, away_top_cand = match_tracks_to_lineup_hungarian(
        [{"track_id": t["track_id"], "median_x": t.get("median_x", 0.0), "median_y": t.get("median_y", 0.0), "duration": t.get("frames_tracked", 100)} for t in away_tracks],
        away_lineup_objs,
        formation_name=getattr(match, "away_formation", "4-3-3"),
        defending_left=False,
    )

    for t in tracks:
        ident = existing.get(t["track_id"])
        t["assigned_lineup_entry_id"] = ident.lineup_entry_id if ident else None
        t["is_auto_assigned"] = ident.is_auto_assigned if ident else False
        t["lineup_options"] = home_lineup if t["team"] == Team.TEAM_A.value else away_lineup
        t["top_candidates"] = home_top_cand.get(t["track_id"], []) if t["team"] == Team.TEAM_A.value else away_top_cand.get(t["track_id"], [])
        t["crop_b64"] = crops_b64.get(t["track_id"])

    context = {
        "match": match,
        "tracks": tracks,
        "has_pitch_data": bool(tracks),
    }
    return render(request, "matches/identify_players.html", context)


@login_required
def identify_players_save(request, public_id):
    """
    Replaces ALL of this match's TrackPlayerIdentification rows with the
    submitted set in one transaction — the page presents the full current
    state and the user edits it as a whole, rather than incremental
    per-track patches, so a full replace is both simpler and matches what
    the user actually sees on screen.

    Each assignment carries a `confirmed` flag from the frontend (true
    only if the user actually changed that dropdown — see identify_players.html's
    "touched" tracking). This determines is_auto_assigned on the new row:
    - confirmed=true  -> is_auto_assigned=False (human explicitly chose it)
    - confirmed=false -> preserves whatever this track_id's PRIOR row had
      (guessed stays guessed, previously-confirmed stays confirmed) — a
      save action must never silently promote an unreviewed auto-guess to
      "confirmed" just because the user didn't touch that one dropdown.
      A track with no prior row defaults to guessed (True) as the safe
      default if confirmed wasn't explicitly sent.

    Rejects (400, no DB changes) if the SAME lineup_entry_id appears
    against more than one track_id in the payload — that's exactly what
    TrackPlayerIdentification's unique_together is protecting against,
    checked here first so the error message can name the conflict instead
    of surfacing a raw IntegrityError.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required.")

    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    try:
        payload = json.loads(request.body)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        return JsonResponse({"error": "assignments must be a list."}, status=400)

    prior_is_auto_assigned = {
        ident.track_id: ident.is_auto_assigned
        for ident in TrackPlayerIdentification.objects.filter(match=match)
    }

    lineup_ids_seen = {}
    to_create = []
    for entry in assignments:
        if not isinstance(entry, dict):
            return JsonResponse({"error": "Each assignment must be an object."}, status=400)
        track_id = entry.get("track_id")
        lineup_entry_id = entry.get("lineup_entry_id")
        if not isinstance(track_id, int):
            return JsonResponse({"error": "Each assignment needs an integer track_id."}, status=400)
        if lineup_entry_id is None:
            continue  # left unassigned — nothing to create for this track
        if not isinstance(lineup_entry_id, int):
            return JsonResponse({"error": "lineup_entry_id must be an integer or null."}, status=400)
        if lineup_entry_id in lineup_ids_seen:
            return JsonResponse({
                "error": f"Lineup player (id={lineup_entry_id}) is assigned to more than one tracked player — "
                         f"tracks {lineup_ids_seen[lineup_entry_id]} and {track_id}. Each real player can only be linked to one."
            }, status=400)
        lineup_ids_seen[lineup_entry_id] = track_id

        confirmed = bool(entry.get("confirmed"))
        is_auto_assigned = False if confirmed else prior_is_auto_assigned.get(track_id, True)
        to_create.append((track_id, lineup_entry_id, is_auto_assigned))

    valid_lineup_ids = set(match.lineups.filter(id__in=lineup_ids_seen.keys()).values_list("id", flat=True))
    invalid_ids = set(lineup_ids_seen.keys()) - valid_lineup_ids
    if invalid_ids:
        return JsonResponse({"error": f"Unknown lineup entry id(s) for this match: {sorted(invalid_ids)}"}, status=400)

    with transaction.atomic():
        TrackPlayerIdentification.objects.filter(match=match).delete()
        TrackPlayerIdentification.objects.bulk_create([
            TrackPlayerIdentification(match=match, track_id=track_id, lineup_entry_id=lineup_entry_id, is_auto_assigned=is_auto_assigned)
            for track_id, lineup_entry_id, is_auto_assigned in to_create
        ])

    try:
        from apps.analytics.services import sync_player_statistics
        sync_player_statistics(match)
    except Exception:
        pass

    # Regenerate the PDF report so it reflects these confirmations —
    # previously missing entirely: generate_match_report was only
    # triggered from process_match/compute_pitch_mapping, so a report
    # could sit stale (showing "unverified" for players a human had
    # already confirmed, or "2 tracked" when 6 had actually been
    # confirmed) until the next full recalibration regenerated it.
    # Synchronous, not a Celery task — PDF generation here is fast
    # (no video processing, just the already-computed CSV data), so
    # there's no need for async complexity for a page the user is about
    # to navigate away from anyway.
    try:
        from apps.reports.generator import generate_match_report
        generate_match_report(match)
    except Exception:
        pass

    return JsonResponse({"ok": True, "redirect_url": reverse("matches:results", args=[match.public_id])})


def _recompute_match_score(match):
    """
    Single source of truth for match.home_score/away_score: counts
    MatchGoal rows credited to each side (an own goal's `team` is
    already the BENEFITING side, not the scorer's side, so this is a
    plain count — no extra own-goal handling needed here). Called by
    add_goal/delete_goal after every change so the two integer fields on
    Match stay in sync with the goals table without the page needing to
    recompute anything itself (match_results just reads match.home_score
    directly, unchanged).
    """
    match.home_score = MatchGoal.objects.filter(match=match, team_id=match.home_team_id).count()
    match.away_score = MatchGoal.objects.filter(match=match, team_id=match.away_team_id).count()
    match.save(update_fields=["home_score", "away_score"])


@login_required
def add_goal(request, public_id):
    """
    Logs one manually-entered goal. No automatic detection anywhere in
    this path — see MatchGoal's docstring for why. team_side ("HOME" or
    "AWAY") is required; scorer_lineup_id and minute are both optional
    (a goal can be logged before the scorer is confirmed, or with no
    known minute).
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required.")

    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)

    try:
        payload = json.loads(request.body)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    team_side = payload.get("team_side")
    if team_side not in (MatchLineup.Side.HOME, MatchLineup.Side.AWAY):
        return JsonResponse({"error": "team_side must be 'HOME' or 'AWAY'."}, status=400)
    team = match.home_team if team_side == MatchLineup.Side.HOME else match.away_team

    scorer = None
    scorer_lineup_id = payload.get("scorer_lineup_id")
    if scorer_lineup_id is not None:
        try:
            scorer = match.lineups.get(id=scorer_lineup_id)
        except (MatchLineup.DoesNotExist, ValueError, TypeError):
            return JsonResponse({"error": "Unknown scorer_lineup_id for this match."}, status=400)

    is_own_goal = bool(payload.get("is_own_goal"))
    if is_own_goal and scorer is None:
        return JsonResponse({"error": "An own goal needs a scorer (the player who scored it)."}, status=400)
    if is_own_goal and scorer.side == team_side:
        # An own goal credited to `team` must have been put in by a
        # player from the OPPOSING side — a same-side "own goal" isn't
        # a coherent thing to record and is almost certainly a UI
        # mistake (wrong side picked), so reject it here rather than
        # silently storing bad data.
        return JsonResponse({"error": "An own goal credited to this team must be scored by a player on the OTHER side."}, status=400)

    minute_raw = payload.get("minute")
    minute = None
    if minute_raw not in (None, ""):
        try:
            minute = int(minute_raw)
        except (TypeError, ValueError):
            return JsonResponse({"error": "minute must be an integer, if provided."}, status=400)
        if not (0 <= minute <= 130):
            return JsonResponse({"error": "minute must be between 0 and 130."}, status=400)

    with transaction.atomic():
        MatchGoal.objects.create(match=match, team=team, scorer=scorer, is_own_goal=is_own_goal, minute=minute)
        _recompute_match_score(match)

    try:
        from apps.analytics.services import sync_player_statistics
        sync_player_statistics(match)
    except Exception:
        pass

    try:
        from apps.reports.generator import generate_match_report
        generate_match_report(match)
    except Exception:
        pass

    return JsonResponse({"ok": True, "redirect_url": reverse("matches:results", args=[match.public_id])})


@login_required
def delete_goal(request, public_id, goal_id):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required.")

    match = get_object_or_404(Match, public_id=public_id, uploaded_by=request.user)
    goal = get_object_or_404(MatchGoal, id=goal_id, match=match)

    with transaction.atomic():
        goal.delete()
        _recompute_match_score(match)

    try:
        from apps.analytics.services import sync_player_statistics
        sync_player_statistics(match)
    except Exception:
        pass

    try:
        from apps.reports.generator import generate_match_report
        generate_match_report(match)
    except Exception:
        pass

    return JsonResponse({"ok": True, "redirect_url": reverse("matches:results", args=[match.public_id])})


def _load_real_stats_by_assignment(match, files_obj):
    """
    Manual-identification counterpart to _load_real_stats_by_jersey:
    joins TrackPlayerIdentification (human-assigned OR auto-guessed
    track -> lineup entry, see that model's docstring) against
    player_stats_csv's per-track stats, keyed by jersey_number so it's a
    drop-in for the same real_stats_by_jersey parameter _dummy_player_rows
    accepts. This is the primary path on footage where jersey OCR isn't
    viable (see jersey_ocr.py) — OCR-based data is still checked as a
    fallback in build_match_report_context, in case OCR is ever
    re-enabled on higher-resolution footage.

    Returns {jersey_number: stats_dict} per team — stats_dict is the
    same shape as _load_real_stats_by_jersey's (see that function's
    docstring for what each field means and how real vs. inferred each
    one is). is_confirmed is False for rows _auto_assign_unidentified_tracks
    guessed and a human hasn't reviewed yet (see tasks.py), True for
    ones a human explicitly chose on /identify/ — that single flag
    applies to every field in the dict, since they all come from the
    same TrackPlayerIdentification row.
    """
    home_by_jersey, away_by_jersey = {}, {}

    if not files_obj or not files_obj.player_stats_csv:
        return home_by_jersey, away_by_jersey

    assignments = {
        ident.track_id: ident
        for ident in TrackPlayerIdentification.objects.filter(match=match).select_related("lineup_entry")
    }
    if not assignments:
        return home_by_jersey, away_by_jersey

    try:
        with files_obj.player_stats_csv.open("rb") as f:
            content = f.read().decode("utf-8")
    except Exception:
        return home_by_jersey, away_by_jersey

    for row in csv.DictReader(io.StringIO(content)):
        try:
            track_id = int(row["track_id"])
        except (KeyError, ValueError):
            continue
        ident = assignments.get(track_id)
        if ident is None:
            continue
        try:
            distance_km = float(row["distance_m"]) / 1000.0
        except (KeyError, ValueError):
            continue

        stats = {
            "distance_km": distance_km,
            "passes_completed": _safe_int(row.get("passes_completed")),
            "passes_attempted": _safe_int(row.get("passes_attempted")),
            "pass_accuracy": _safe_float(row.get("pass_accuracy")),
            "shots": _safe_int(row.get("shots")),
            "shots_on_target": _safe_int(row.get("shots_on_target")),
            "xg": _safe_float(row.get("xg")),
            "top_speed": _safe_float(row.get("top_speed")),
            "average_speed": _safe_float(row.get("average_speed")),
            "tackles": _safe_int(row.get("tackles")),
            "interceptions": _safe_int(row.get("interceptions")),
            "clearances": _safe_int(row.get("clearances")),
            "dribbles_completed": _safe_int(row.get("dribbles_completed")),
            "key_passes": _safe_int(row.get("key_passes")),
            "rating": _safe_float(row.get("rating"), default=6.0),
            "is_confirmed": not ident.is_auto_assigned,
        }

        lineup_entry = ident.lineup_entry
        if lineup_entry.side == MatchLineup.Side.HOME:
            home_by_jersey[lineup_entry.jersey_number] = stats
        elif lineup_entry.side == MatchLineup.Side.AWAY:
            away_by_jersey[lineup_entry.jersey_number] = stats

    return home_by_jersey, away_by_jersey


def _load_real_shots(match, files_obj):
    """
    Loads detected shots from MatchFiles.shots_csv.
    Pairs track_id with TrackPlayerIdentification to identify the shooter.
    If track_id is unassigned (e.g. due to CV tracklet fragmentation):
      - Goal shots are attributed to the match goal scorer (or starting forward).
      - Long-range shots (distance >= 22m) are attributed to attacking midfielders.
      - Box / close-range shots are attributed to forwards.
    """
    shots = []
    if not files_obj or not files_obj.shots_csv:
        return shots

    assignments = {
        ident.track_id: ident.lineup_entry
        for ident in TrackPlayerIdentification.objects.filter(match=match).select_related("lineup_entry")
    }

    home_fwds = list(match.lineups.filter(side=MatchLineup.Side.HOME, position="FWD").order_by("jersey_number"))
    home_mids = list(match.lineups.filter(side=MatchLineup.Side.HOME, position="MID").order_by("jersey_number"))
    away_fwds = list(match.lineups.filter(side=MatchLineup.Side.AWAY, position="FWD").order_by("jersey_number"))
    away_mids = list(match.lineups.filter(side=MatchLineup.Side.AWAY, position="MID").order_by("jersey_number"))

    shot_distrib = {"home_fwd": 0, "home_mid": 0, "away_fwd": 0, "away_mid": 0}

    try:
        with files_obj.shots_csv.open("rb") as f:
            reader = csv.DictReader(io.StringIO(f.read().decode("utf-8")))
            rows = list(reader)

        for row in rows:
            track_id = _safe_int(row.get("track_id"))
            lineup_entry = assignments.get(track_id)
            team_val = row.get("team", "")

            if lineup_entry:
                side = "home" if lineup_entry.side == MatchLineup.Side.HOME else "away"
                player_inferred = False
            else:
                side = "home" if team_val == "team_a" else "away"
                player_inferred = True

            team_obj = match.home_team if side == "home" else match.away_team
            outcome_raw = row.get("outcome") or ""
            is_goal = outcome_raw.lower() == "goal"
            dist_val = _safe_float(row.get("distance_m"))
            speed_val = _safe_float(row.get("speed_mps"))
            minute_val = _safe_int(row.get("minute"), default=1)
            xg_val = _safe_float(row.get("xg"))
            is_on_target = row.get("is_on_target", "").lower() in ("true", "1") or is_goal
            outcome = "Goal" if is_goal else (row.get("outcome") or ("On Target" if is_on_target else "Off Target"))

            fwds = home_fwds if side == "home" else away_fwds
            mids = home_mids if side == "home" else away_mids

            if not lineup_entry:
                if is_goal:
                    g_match = match.goals.filter(team=team_obj, is_own_goal=False).select_related("scorer").first()
                    if g_match and g_match.scorer:
                        lineup_entry = g_match.scorer
                    elif fwds:
                        lineup_entry = fwds[0]
                else:
                    if dist_val >= 22.0 and mids:
                        ck = f"{side}_mid"
                        lineup_entry = mids[shot_distrib[ck] % len(mids)]
                        shot_distrib[ck] += 1
                    elif fwds:
                        ck = f"{side}_fwd"
                        lineup_entry = fwds[shot_distrib[ck] % len(fwds)]
                        shot_distrib[ck] += 1
                    elif mids:
                        ck = f"{side}_mid"
                        lineup_entry = mids[shot_distrib[ck] % len(mids)]
                        shot_distrib[ck] += 1

            if lineup_entry:
                player_label = f"#{lineup_entry.jersey_number} {lineup_entry.player_name}"
                jersey_number = lineup_entry.jersey_number
                player_name = lineup_entry.player_name
                lineup_entry_id = lineup_entry.id
            else:
                player_label = f"{team_obj.short_name} Player"
                jersey_number = None
                player_name = f"{team_obj.short_name} Player"
                lineup_entry_id = None

            shots.append({
                "frame_idx": _safe_int(row.get("frame_idx")),
                "minute": minute_val,
                "player": player_label,
                "player_inferred": player_inferred,
                "player_name": player_name,
                "jersey_number": jersey_number,
                "lineup_entry_id": lineup_entry_id,
                "side": side,
                "xg": xg_val,
                "outcome": outcome,
                "is_on_target": is_on_target,
                "pitch_x": _safe_float(row.get("pitch_x")),
                "pitch_y": _safe_float(row.get("pitch_y")),
                "target_goal_x": _safe_float(row.get("target_goal_x")),
                "target_goal_y": _safe_float(row.get("target_goal_y")),
                "distance_m": dist_val,
                "speed_mps": speed_val,
            })
    except Exception:
        logger.exception("Failed to parse shots_csv for match=%s", match.id)

    return shots


def _load_real_passes(match, files_obj, shots=None):
    """
    Loads detected passes from MatchFiles.passes_csv.
    Pairs passer_track_id and receiver_track_id with TrackPlayerIdentification
    to identify the passer and receiver.
    Enriches each pass with tactical attributes:
      - is_progressive (gains >= 9.5m toward attacking goal)
      - is_cross (originates from wide channel |y| >= 14m into penalty area)
      - is_key_pass (leads to a shot within 10 seconds / 250 frames)
    """
    passes = []
    if not files_obj or not files_obj.passes_csv:
        return passes

    assignments = {
        ident.track_id: ident.lineup_entry
        for ident in TrackPlayerIdentification.objects.filter(match=match).select_related("lineup_entry")
    }

    shots_list = shots or []

    try:
        with files_obj.passes_csv.open("rb") as f:
            reader = csv.DictReader(io.StringIO(f.read().decode("utf-8")))
            for row in reader:
                passer_tid = _safe_int(row.get("passer_track_id"))
                receiver_tid = _safe_int(row.get("receiver_track_id"))
                passer_entry = assignments.get(passer_tid)
                receiver_entry = assignments.get(receiver_tid)
                passer_team_val = row.get("passer_team", "")

                if passer_entry:
                    side = "home" if passer_entry.side == MatchLineup.Side.HOME else "away"
                    passer_label = f"#{passer_entry.jersey_number} {passer_entry.player_name}"
                else:
                    side = "home" if passer_team_val == "team_a" else "away"
                    team_obj = match.home_team if side == "home" else match.away_team
                    passer_label = f"{team_obj.short_name} Player"

                if receiver_entry:
                    receiver_label = f"#{receiver_entry.jersey_number} {receiver_entry.player_name}"
                elif receiver_tid:
                    rec_side = "home" if row.get("receiver_team", "") == "team_a" else "away"
                    rec_team_obj = match.home_team if rec_side == "home" else match.away_team
                    receiver_label = f"{rec_team_obj.short_name} Player"
                else:
                    receiver_label = "Incomplete / Intercepted"

                is_completed = str(row.get("is_completed", "")).strip().lower() in ("true", "1")
                sx = _safe_float(row.get("start_x"))
                sy = _safe_float(row.get("start_y"))
                ex = _safe_float(row.get("end_x"))
                ey = _safe_float(row.get("end_y"))
                dist_m = _safe_float(row.get("distance_m"))
                speed_mps = _safe_float(row.get("speed_mps"))
                frame_idx = _safe_int(row.get("frame_idx"))
                start_frame = _safe_int(row.get("start_frame"), default=frame_idx)

                # Tactical Classification
                # Progressive: advances >= 9.5m toward opponent goal line
                if side == "home":
                    is_progressive = (ex - sx) >= 9.5
                    is_cross = (abs(sy) >= 14.0) and (ex >= 35.0) and (abs(ey) <= 20.16)
                else:
                    is_progressive = (sx - ex) >= 9.5
                    is_cross = (abs(sy) >= 14.0) and (ex <= -35.0) and (abs(ey) <= 20.16)

                # Key pass: leads directly to a shot within 10 seconds (250 frames)
                is_key_pass = False
                for s in shots_list:
                    if s.get("side") == side:
                        s_frame = s.get("frame_idx", 0)
                        if 0 <= s_frame - frame_idx <= 250 or 0 <= s_frame - start_frame <= 250:
                            is_key_pass = True
                            break

                passes.append({
                    "frame_idx": frame_idx,
                    "start_frame": start_frame,
                    "minute": _safe_int(row.get("minute"), default=1),
                    "passer": passer_label,
                    "passer_track_id": passer_tid,
                    "receiver": receiver_label,
                    "receiver_track_id": receiver_tid,
                    "side": side,
                    "start_x": sx,
                    "start_y": sy,
                    "end_x": ex,
                    "end_y": ey,
                    "distance_m": dist_m,
                    "speed_mps": speed_mps,
                    "is_completed": is_completed,
                    "is_progressive": is_progressive,
                    "is_cross": is_cross,
                    "is_key_pass": is_key_pass,
                })
    except Exception:
        logger.exception("Failed to parse passes_csv for match=%s", match.id)

    return passes


def _load_team_defensive_stats(files_obj):
    """
    Computes total tackles, interceptions, clearances for home (team_a) and away (team_b)
    from files_obj.player_stats_csv.
    """
    totals = {
        "home": {"tackles": 0, "interceptions": 0, "clearances": 0},
        "away": {"tackles": 0, "interceptions": 0, "clearances": 0},
    }
    if not files_obj or not files_obj.player_stats_csv:
        return totals

    try:
        with files_obj.player_stats_csv.open("rb") as f:
            reader = csv.DictReader(io.StringIO(f.read().decode("utf-8")))
            for row in reader:
                team_key = "home" if row.get("team") == "team_a" else "away" if row.get("team") == "team_b" else None
                if not team_key:
                    continue
                totals[team_key]["tackles"] += _safe_int(row.get("tackles"))
                totals[team_key]["interceptions"] += _safe_int(row.get("interceptions"))
                totals[team_key]["clearances"] += _safe_int(row.get("clearances"))
    except Exception:
        logger.exception("Failed to parse player_stats_csv for defensive totals")

    return totals


def _compute_tactical_pitch_data(match, files_obj, using_real_stats, home_players, away_players):
    """
    Computes passing networks and team shape (convex hull, compactness area,
    length, width, centroid) for Home and Away teams.
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    def build_fallback_team_tactical(players, side, match_seed):
        rng = random.Random(f"{match_seed}_{side}_tactics")
        dir_mult = 1.0 if side == "home" else -1.0

        pos_templates = {
            "GK": [(-44.0 * dir_mult, 0.0)],
            "DEF": [
                (-28.0 * dir_mult, -20.0),
                (-30.0 * dir_mult, -7.0),
                (-30.0 * dir_mult, 7.0),
                (-28.0 * dir_mult, 20.0),
            ],
            "MID": [
                (-15.0 * dir_mult, -14.0),
                (-12.0 * dir_mult, 0.0),
                (-15.0 * dir_mult, 14.0),
            ],
            "FWD": [
                (12.0 * dir_mult, -18.0),
                (16.0 * dir_mult, 0.0),
                (12.0 * dir_mult, 18.0),
            ],
        }

        pos_counters = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        nodes = []
        for p in players:
            pos_type = p.get("position", "MID")
            if pos_type not in pos_templates:
                pos_type = "MID"
            idx = pos_counters[pos_type]
            pos_counters[pos_type] += 1
            if idx < len(pos_templates[pos_type]):
                base_x, base_y = pos_templates[pos_type][idx]
            else:
                base_x = rng.uniform(-15.0, 15.0) * dir_mult
                base_y = rng.uniform(-22.0, 22.0)

            jitter_x = rng.uniform(-2.0, 2.0)
            jitter_y = rng.uniform(-2.0, 2.0)
            node_x = round(base_x + jitter_x, 2)
            node_y = round(base_y + jitter_y, 2)

            nodes.append({
                "id": p.get("jersey_number") or p.get("name"),
                "jersey": p.get("jersey_number"),
                "name": p.get("name"),
                "position": pos_type,
                "x": node_x,
                "y": node_y,
                "is_tracked": False,
                "passes_made": rng.randint(18, 55),
                "passes_received": rng.randint(15, 50),
            })

        links = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                d = ((n1["x"] - n2["x"])**2 + (n1["y"] - n2["y"])**2)**0.5
                if d < 28.0:
                    p_count = rng.randint(3, 14)
                    links.append({
                        "source": n1["id"],
                        "target": n2["id"],
                        "source_name": n1["name"],
                        "target_name": n2["name"],
                        "count": p_count,
                        "completed": int(p_count * rng.uniform(0.8, 0.95)),
                        "first_frame": rng.randint(50, 600),
                    })

        outfield = [n for n in nodes if n["position"] != "GK"]
        if len(outfield) >= 3:
            pts = np.array([[n["x"], n["y"]] for n in outfield])
            try:
                hull = ConvexHull(pts)
                hull_verts = [{"x": round(float(pts[v, 0]), 2), "y": round(float(pts[v, 1]), 2)} for v in hull.vertices]
                centroid = {
                    "x": round(float(np.mean(pts[:, 0])), 2),
                    "y": round(float(np.mean(pts[:, 1])), 2),
                }
                shape = {
                    "hull_vertices": hull_verts,
                    "centroid": centroid,
                    "length_m": round(float(np.ptp(pts[:, 0])), 1),
                    "width_m": round(float(np.ptp(pts[:, 1])), 1),
                    "area_sqm": round(float(hull.volume), 1),
                }
            except Exception:
                shape = None
        else:
            shape = None

        return {
            "nodes": nodes,
            "links": links,
            "shape": shape,
            "is_real": False,
        }

    if not files_obj or not files_obj.player_tracking_csv:
        return {
            "home": build_fallback_team_tactical(home_players, "home", str(match.public_id)),
            "away": build_fallback_team_tactical(away_players, "away", str(match.public_id)),
        }

    track_pts = defaultdict(list)
    try:
        with files_obj.player_tracking_csv.open("rb") as pf:
            for r in csv.DictReader(io.StringIO(pf.read().decode("utf-8"))):
                px = r.get("pitch_x") if r.get("pitch_x") is not None else r.get("x")
                py = r.get("pitch_y") if r.get("pitch_y") is not None else r.get("y")
                if px not in (None, "") and py not in (None, ""):
                    tid = _safe_int(r.get("track_id"))
                    if tid is not None:
                        track_pts[tid].append((float(px), float(py)))
    except Exception:
        logger.exception("Failed to parse player_tracking_csv for tactical data")

    pass_pair_counts = defaultdict(lambda: {"count": 0, "completed": 0, "first_frame": 999999})
    passes_made_by_tid = defaultdict(int)
    passes_rec_by_tid = defaultdict(int)
    if files_obj.passes_csv:
        try:
            with files_obj.passes_csv.open("rb") as pf:
                for r in csv.DictReader(io.StringIO(pf.read().decode("utf-8"))):
                    p_tid = _safe_int(r.get("passer_track_id"))
                    r_tid = _safe_int(r.get("receiver_track_id"))
                    is_comp = str(r.get("is_completed", "")).strip().lower() in ("true", "1")
                    frame = _safe_int(r.get("frame_idx"))
                    if p_tid:
                        passes_made_by_tid[p_tid] += 1
                    if r_tid:
                        passes_rec_by_tid[r_tid] += 1
                    if p_tid and r_tid and p_tid != r_tid:
                        key = (p_tid, r_tid)
                        pass_pair_counts[key]["count"] += 1
                        if is_comp:
                            pass_pair_counts[key]["completed"] += 1
                        if frame < pass_pair_counts[key]["first_frame"]:
                            pass_pair_counts[key]["first_frame"] = frame
        except Exception:
            logger.exception("Failed to parse passes_csv for tactical links")

    home_nodes, away_nodes = [], []
    for ident in TrackPlayerIdentification.objects.filter(match=match).select_related("lineup_entry"):
        entry = ident.lineup_entry
        tid = ident.track_id
        pts = track_pts.get(tid, [])
        is_tracked = len(pts) >= 5

        dir_mult = 1.0 if entry.side == MatchLineup.Side.HOME else -1.0
        if is_tracked:
            avg_x = round(float(np.mean([p[0] for p in pts])), 2)
            avg_y = round(float(np.mean([p[1] for p in pts])), 2)
        else:
            def_pos = {"GK": -42.0, "DEF": -28.0, "MID": -12.0, "FWD": 14.0}.get(entry.position, -10.0)
            avg_x = round(def_pos * dir_mult, 2)
            avg_y = 0.0

        node = {
            "id": tid,
            "track_id": tid,
            "lineup_id": entry.id,
            "name": entry.player_name,
            "jersey": entry.jersey_number,
            "position": entry.position,
            "x": avg_x,
            "y": avg_y,
            "is_tracked": is_tracked,
            "passes_made": passes_made_by_tid.get(tid, 0),
            "passes_received": passes_rec_by_tid.get(tid, 0),
            "samples": len(pts),
        }
        if entry.side == MatchLineup.Side.HOME:
            home_nodes.append(node)
        else:
            away_nodes.append(node)

    def compute_team_shape(nodes):
        if not nodes:
            return None
        tracked_nodes = [n for n in nodes if n.get("is_tracked")]
        outfield = [n for n in tracked_nodes if n.get("position") != "GK"]
        pts_to_use = outfield if len(outfield) >= 3 else tracked_nodes
        if len(pts_to_use) < 3:
            outfield_all = [n for n in nodes if n.get("position") != "GK"]
            pts_to_use = outfield_all if len(outfield_all) >= 3 else nodes
        if len(pts_to_use) < 3:
            return None

        pts = np.array([[n["x"], n["y"]] for n in pts_to_use])
        try:
            hull = ConvexHull(pts)
            hull_verts = [{"x": round(float(pts[v, 0]), 2), "y": round(float(pts[v, 1]), 2)} for v in hull.vertices]
            centroid = {
                "x": round(float(np.mean(pts[:, 0])), 2),
                "y": round(float(np.mean(pts[:, 1])), 2),
            }
            return {
                "hull_vertices": hull_verts,
                "centroid": centroid,
                "length_m": round(float(np.ptp(pts[:, 0])), 1),
                "width_m": round(float(np.ptp(pts[:, 1])), 1),
                "area_sqm": round(float(hull.volume), 1),
            }
        except Exception:
            return None

    def extract_team_links(nodes):
        node_ids = {n["id"] for n in nodes}
        node_name_map = {n["id"]: n["name"] for n in nodes}
        team_links = []
        for (p_tid, r_tid), data in pass_pair_counts.items():
            if p_tid in node_ids and r_tid in node_ids:
                team_links.append({
                    "source": p_tid,
                    "target": r_tid,
                    "source_name": node_name_map.get(p_tid, f"#{p_tid}"),
                    "target_name": node_name_map.get(r_tid, f"#{r_tid}"),
                    "count": data["count"],
                    "completed": data["completed"],
                    "first_frame": data["first_frame"] if data["first_frame"] < 999999 else 0,
                })
        return team_links

    home_tracked_count = sum(1 for n in home_nodes if n.get("is_tracked"))
    away_tracked_count = sum(1 for n in away_nodes if n.get("is_tracked"))

    home_data = (
        {
            "nodes": home_nodes,
            "links": extract_team_links(home_nodes),
            "shape": compute_team_shape(home_nodes),
            "is_real": home_tracked_count >= 3,
        }
        if len(home_nodes) >= 3
        else build_fallback_team_tactical(home_players, "home", str(match.public_id))
    )

    away_data = (
        {
            "nodes": away_nodes,
            "links": extract_team_links(away_nodes),
            "shape": compute_team_shape(away_nodes),
            "is_real": away_tracked_count >= 3,
        }
        if len(away_nodes) >= 3
        else build_fallback_team_tactical(away_players, "away", str(match.public_id))
    )

    return {
        "home": home_data,
        "away": away_data,
    }


def build_match_report_context(match):
    """
    Gathers everything shown on the results page for one match — team
    stats, per-player rows, heatmaps, shots, timeline — real where
    available, falling back to the Phase 5 dummy generator otherwise.

    Factored out of match_results() so apps.reports' PDF generator can
    call this directly and guarantee the PDF always shows exactly the
    same numbers as the web page for the same match, rather than two
    separate implementations silently drifting apart over time.
    """
    rng = random.Random(str(match.public_id))

    home_lineup = match.lineups.filter(side=MatchLineup.Side.HOME).order_by("jersey_number")
    away_lineup = match.lineups.filter(side=MatchLineup.Side.AWAY).order_by("jersey_number")

    files_obj = getattr(match, "files", None)
    home_stats_ocr, away_stats_ocr = _load_real_stats_by_jersey(files_obj)
    home_stats_manual, away_stats_manual = _load_real_stats_by_assignment(match, files_obj)
    home_stats_by_jersey = {**home_stats_ocr, **home_stats_manual}
    away_stats_by_jersey = {**away_stats_ocr, **away_stats_manual}

    detected_shots = _load_real_shots(match, files_obj)

    match_goals = list(MatchGoal.objects.filter(match=match).select_related("scorer", "team"))

    # Auto-resolve any missing scorer on existing goals
    for g in match_goals:
        if g.scorer_id is None:
            side_val = MatchLineup.Side.HOME if g.team_id == match.home_team_id else MatchLineup.Side.AWAY
            fwd = match.lineups.filter(side=side_val, position="FWD").first()
            if fwd:
                g.scorer = fwd
                g.save(update_fields=["scorer"])

    # If no MatchGoal records exist, auto-create from detected shots if available
    if not match_goals and detected_shots:
        for s in detected_shots:
            if s.get("outcome", "").lower() == "goal":
                side = s.get("side", "away")
                team_obj = match.home_team if side == "home" else match.away_team
                side_val = MatchLineup.Side.HOME if side == "home" else MatchLineup.Side.AWAY
                scorer_entry = match.lineups.filter(id=s.get("lineup_entry_id")).first() or match.lineups.filter(side=side_val, position="FWD").first()
                mg = MatchGoal.objects.create(
                    match=match,
                    team=team_obj,
                    scorer=scorer_entry,
                    minute=s.get("minute", 1),
                    is_own_goal=False,
                )
                match_goals.append(mg)
        if match_goals:
            _recompute_match_score(match)

    if match_goals:
        goals_by_lineup_id = defaultdict(int)
        for g in match_goals:
            if not g.is_own_goal and g.scorer_id is not None:
                goals_by_lineup_id[g.scorer_id] += 1
    else:
        goals_by_lineup_id = None

    # Reconcile detected shots directly into home_stats_by_jersey and away_stats_by_jersey
    # so that each player's row receives their actual detected shots, shots on target, and xG.
    if detected_shots:
        teams_with_detected_shots = {s.get("side") for s in detected_shots if s.get("side")}
        for side in teams_with_detected_shots:
            target_dict = home_stats_by_jersey if side == "home" else away_stats_by_jersey
            for j in target_dict:
                target_dict[j]["shots"] = 0
                target_dict[j]["shots_on_target"] = 0
                target_dict[j]["xg"] = 0.0

        for s in detected_shots:
            j = s.get("jersey_number")
            side = s.get("side")
            if j is not None and side:
                stats_dict = home_stats_by_jersey if side == "home" else away_stats_by_jersey
                if j not in stats_dict:
                    stats_dict[j] = {
                        "distance_km": 0.0,
                        "passes_completed": 0,
                        "passes_attempted": 0,
                        "pass_accuracy": 0.0,
                        "shots": 0,
                        "shots_on_target": 0,
                        "xg": 0.0,
                        "top_speed": 0.0,
                        "average_speed": 0.0,
                        "tackles": 0,
                        "interceptions": 0,
                        "clearances": 0,
                        "dribbles_completed": 0,
                        "key_passes": 0,
                        "rating": 6.5,
                        "is_confirmed": not s.get("player_inferred", False),
                    }
                stats_dict[j]["shots"] += 1
                if s.get("is_on_target") or s.get("outcome", "").lower() in ("goal", "on target", "saved"):
                    stats_dict[j]["shots_on_target"] += 1
                stats_dict[j]["xg"] = round(stats_dict[j]["xg"] + s.get("xg", 0.0), 2)

    home_players = _dummy_player_rows(rng, home_lineup, match.home_team, home_stats_by_jersey, goals_by_lineup_id)
    away_players = _dummy_player_rows(rng, away_lineup, match.away_team, away_stats_by_jersey, goals_by_lineup_id)

    # --- Team-level stats: real if this match has them, dummy otherwise ---
    real_team_stats = {
        ts.team_id: ts
        for ts in TeamStatistics.objects.filter(match=match)
    }
    using_real_stats = bool(real_team_stats)

    if using_real_stats:
        home_ts = real_team_stats.get(match.home_team_id)
        away_ts = real_team_stats.get(match.away_team_id)

        total_possession = (home_ts.possession if home_ts else 0) + (away_ts.possession if away_ts else 0)
        if total_possession > 0:
            home_possession = round((home_ts.possession / total_possession) * 100) if home_ts else 50
        else:
            home_possession = 50
        away_possession = 100 - home_possession

        def_stats = _load_team_defensive_stats(files_obj)

        def real_team_stat_block(ts, def_info=None):
            def_info = def_info or {"tackles": 0, "interceptions": 0, "clearances": 0}
            if ts is None:
                return {
                    "goals": 0,
                    "shots": 0, "shots_on_target": 0, "shot_accuracy": 0.0,
                    "passes": 0, "passes_completed": 0, "passes_attempted": 0, "pass_accuracy": 0.0,
                    "corners": 0, "fouls": 0, "yellow_cards": 0, "red_cards": 0,
                    "xg": 0.0, "distance_km": 0.0, "average_speed": 0.0,
                    "tackles": 0, "interceptions": 0, "clearances": 0,
                }
            s_count = ts.shots
            sot_count = ts.shots_on_target
            shot_acc = round(100.0 * sot_count / s_count, 1) if s_count > 0 else 0.0
            return {
                "goals": ts.goals,
                "shots": s_count,
                "shots_on_target": sot_count,
                "shot_accuracy": shot_acc,
                "passes": ts.passes_completed,
                "passes_completed": ts.passes_completed,
                "passes_attempted": ts.passes_attempted,
                "pass_accuracy": round(ts.pass_accuracy, 1),
                "corners": ts.corners,
                "fouls": ts.fouls,
                "yellow_cards": ts.yellow_cards,
                "red_cards": ts.red_cards,
                "xg": round(ts.xg, 2),
                "distance_km": round(ts.total_distance / 1000, 2),
                "average_speed": round(ts.average_team_speed, 1),
                "tackles": def_info["tackles"],
                "interceptions": def_info["interceptions"],
                "clearances": def_info["clearances"],
            }

        team_stats = {
            "home": real_team_stat_block(home_ts, def_stats["home"]),
            "away": real_team_stat_block(away_ts, def_stats["away"]),
        }

        from ai_engine.heatmap import render_heatmap_png

        home_positions, away_positions = _load_pitch_positions_by_team(getattr(match, "files", None))
        home_heatmap = render_heatmap_png(home_positions, attack_direction="right")
        away_heatmap = render_heatmap_png(away_positions, attack_direction="left")

        shots = detected_shots
        passes = _load_real_passes(match, files_obj, shots=shots)
        timeline = [{"minute": 0, "type": "kickoff", "description": "Kickoff"}]
        for g in match_goals:
            scorer_label = f"#{g.scorer.jersey_number} {g.scorer.player_name}" if g.scorer else "Unknown scorer"
            og_label = " (OG)" if g.is_own_goal else ""
            timeline.append({
                "minute": g.minute if g.minute is not None else "",
                "type": "goal",
                "description": f"Goal{og_label} — {scorer_label} ({g.team.short_name})",
            })
        for s in shots:
            if s.get("outcome", "").lower() != "goal":
                timeline.append({
                    "minute": s["minute"],
                    "type": "shot",
                    "description": f"Shot ({s['outcome']}) — {s['player']} (xG {s['xg']:.2f})",
                })
        timeline.append({"minute": 90, "type": "fulltime", "description": "Full Time"})
        timeline.sort(key=lambda e: e["minute"] if isinstance(e["minute"], int) else 999)

    else:
        # --- Original Phase 5 dummy generation, unchanged ---
        home_heatmap = away_heatmap = None
        home_possession = rng.randint(38, 62)
        away_possession = 100 - home_possession

        def team_stat_block():
            att = rng.randint(350, 650)
            comp = int(att * rng.uniform(0.75, 0.90))
            s_count = rng.randint(8, 18)
            sot_count = rng.randint(3, min(9, s_count))
            return {
                "shots": s_count,
                "shots_on_target": sot_count,
                "shot_accuracy": round(100.0 * sot_count / s_count, 1) if s_count > 0 else 0.0,
                "passes": comp,
                "passes_completed": comp,
                "passes_attempted": att,
                "pass_accuracy": round(100.0 * comp / att, 1),
                "corners": rng.randint(2, 9),
                "fouls": rng.randint(6, 14),
                "yellow_cards": rng.randint(0, 4),
                "red_cards": rng.choice([0, 0, 0, 0, 1]),
                "xg": round(rng.uniform(0.8, 2.9), 2),
                "distance_km": round(rng.uniform(105, 118), 1),
                "average_speed": round(rng.uniform(6.5, 8.5), 1),
                "tackles": rng.randint(12, 28),
                "interceptions": rng.randint(8, 20),
                "clearances": rng.randint(10, 25),
            }

        team_stats = {"home": team_stat_block(), "away": team_stat_block()}

        shot_outcomes = ["Goal", "Saved", "Blocked", "Off Target", "Woodwork"]
        shot_pool = [(p, "home") for p in home_players] + [(p, "away") for p in away_players]
        shots = []
        for _ in range(rng.randint(10, 18)):
            player, side = rng.choice(shot_pool)
            if side == "home":
                px = round(rng.uniform(18.0, 48.0), 1)
                py = round(rng.uniform(-20.0, 20.0), 1)
                tgx, tgy = 52.5, 0.0
            else:
                px = round(rng.uniform(-48.0, -18.0), 1)
                py = round(rng.uniform(-20.0, 20.0), 1)
                tgx, tgy = -52.5, 0.0
            dist = round(((tgx - px)**2 + (tgy - py)**2)**0.5, 1)
            speed = round(rng.uniform(16.0, 31.0), 1)
            shots.append({
                "frame_idx": rng.randint(25, 750),
                "minute": rng.randint(1, 90),
                "player": player["name"],
                "side": side,
                "xg": round(rng.uniform(0.02, 0.75), 2),
                "outcome": rng.choice(shot_outcomes),
                "pitch_x": px,
                "pitch_y": py,
                "target_goal_x": tgx,
                "target_goal_y": tgy,
                "distance_m": dist,
                "speed_mps": speed,
            })
        shots.sort(key=lambda s: s["minute"])

        passes = []

        timeline = [{"minute": 0, "type": "kickoff", "description": "Kickoff"}]
        event_types = ["goal", "yellow_card", "red_card", "substitution"]
        for _ in range(rng.randint(6, 10)):
            etype = rng.choice(event_types)
            side = rng.choice(["home", "away"])
            team = match.home_team if side == "home" else match.away_team
            player = rng.choice(home_players if side == "home" else away_players)
            label = {
                "goal": "Goal",
                "yellow_card": "Yellow card",
                "red_card": "Red card",
                "substitution": "Substitution",
            }[etype]
            timeline.append({
                "minute": rng.randint(1, 90),
                "type": etype,
                "description": f"{label} — {player['name']} ({team.short_name})",
            })
        timeline.append({"minute": 90, "type": "fulltime", "description": "Full Time"})
        timeline.sort(key=lambda e: e["minute"])

    video_obj = getattr(match, "video", None)
    annotated_video_url = None
    original_video_url = None
    if video_obj:
        if video_obj.annotated_video:
            annotated_video_url = video_obj.annotated_video.url
        if video_obj.original_video:
            original_video_url = video_obj.original_video.url

    tactical_data = _compute_tactical_pitch_data(match, files_obj, using_real_stats, home_players, away_players)

    return {
        "match": match,
        "home_players": home_players,
        "away_players": away_players,
        "home_possession": home_possession,
        "away_possession": away_possession,
        "team_stats": team_stats,
        "shots": shots,
        "passes": passes,
        "timeline": timeline,
        "using_real_stats": using_real_stats,
        "home_heatmap": home_heatmap,
        "away_heatmap": away_heatmap,
        "has_calibration": match.calibrations.exists(),
        "match_goals": match_goals,
        "home_goals": [g for g in match_goals if g.team_id == match.home_team_id],
        "away_goals": [g for g in match_goals if g.team_id == match.away_team_id],
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
        "annotated_video_url": annotated_video_url,
        "original_video_url": original_video_url,
        "has_video": bool(annotated_video_url or original_video_url),
        "tactical_data": tactical_data,
    }


@login_required
def player_heatmap(request, public_id, lineup_entry_id):
    """
    Serves ONE player's heatmap as a PNG, on demand — generated fresh
    per request (matplotlib render, not cached/stored), same "generate
    on demand from the CSV, no persisted image" design as the team
    heatmaps. Only ever called for a player who actually has a tracked
    identity (see results.html — the modal only shows a heatmap section
    when distance_is_real is true for that player), but this view
    checks for itself rather than trusting the caller.
    """
    match = get_object_or_404(Match, public_id=public_id)

    ident = TrackPlayerIdentification.objects.filter(
        match=match, lineup_entry_id=lineup_entry_id
    ).select_related("lineup_entry").first()
    if ident is None:
        return HttpResponseBadRequest("This player has no tracked identity yet.")

    files_obj = getattr(match, "files", None)
    positions = _load_pitch_positions_for_track(files_obj, ident.track_id)

    from ai_engine.heatmap import render_heatmap_png
    attack_dir = "right" if ident.lineup_entry.side == MatchLineup.Side.HOME else "left"
    png_b64 = render_heatmap_png(positions, attack_direction=attack_dir)
    if png_b64 is None:
        return HttpResponseBadRequest("Not enough tracked positions for this player yet.")

    return HttpResponse(base64.b64decode(png_b64), content_type="image/png")


@login_required
def match_results(request, public_id):
    match = get_object_or_404(Match, public_id=public_id)

    if match.status != Match.MatchStatus.COMPLETED:
        return redirect("matches:processing", public_id=match.public_id)

    context = build_match_report_context(match)
    context["is_uploader"] = request.user == match.uploaded_by
    context["match_report"] = Report.objects.filter(match=match, report_type=Report.ReportType.MATCH).first()

    # Serialize player data for dynamic client-side comparison and radar generation
    all_players = []
    for p in context.get("home_players", []):
        all_players.append({
            "lineup_id": p.get("lineup_id"),
            "player_id": p.get("player_id"),
            "name": p.get("name"),
            "jersey_number": p.get("jersey_number"),
            "position": p.get("position"),
            "team_side": "HOME",
            "team_name": match.home_team.name,
            "team_color": match.home_kit_color or "#ef4444",
            "minutes_played": p.get("minutes_played", 0),
            "goals": p.get("goals", 0),
            "assists": p.get("assists", 0),
            "shots": p.get("shots", 0),
            "shots_on_target": p.get("shots_on_target", 0),
            "shot_accuracy": p.get("shot_accuracy", 0.0),
            "xg": p.get("xg", 0.0),
            "passes_completed": p.get("passes_completed", 0),
            "passes_attempted": p.get("passes_attempted", 0),
            "pass_accuracy": p.get("pass_accuracy", 0.0),
            "key_passes": p.get("key_passes", 0),
            "dribbles_completed": p.get("dribbles_completed", 0),
            "tackles": p.get("tackles", 0),
            "interceptions": p.get("interceptions", 0),
            "clearances": p.get("clearances", 0),
            "distance_km": p.get("distance_km", 0.0),
            "top_speed": p.get("top_speed", 0.0),
            "average_speed": p.get("average_speed", 0.0),
            "rating": p.get("rating", 6.0),
            "distance_is_real": p.get("distance_is_real", False),
        })

    for p in context.get("away_players", []):
        all_players.append({
            "lineup_id": p.get("lineup_id"),
            "player_id": p.get("player_id"),
            "name": p.get("name"),
            "jersey_number": p.get("jersey_number"),
            "position": p.get("position"),
            "team_side": "AWAY",
            "team_name": match.away_team.name,
            "team_color": match.away_kit_color or "#eab308",
            "minutes_played": p.get("minutes_played", 0),
            "goals": p.get("goals", 0),
            "assists": p.get("assists", 0),
            "shots": p.get("shots", 0),
            "shots_on_target": p.get("shots_on_target", 0),
            "shot_accuracy": p.get("shot_accuracy", 0.0),
            "xg": p.get("xg", 0.0),
            "passes_completed": p.get("passes_completed", 0),
            "passes_attempted": p.get("passes_attempted", 0),
            "pass_accuracy": p.get("pass_accuracy", 0.0),
            "key_passes": p.get("key_passes", 0),
            "dribbles_completed": p.get("dribbles_completed", 0),
            "tackles": p.get("tackles", 0),
            "interceptions": p.get("interceptions", 0),
            "clearances": p.get("clearances", 0),
            "distance_km": p.get("distance_km", 0.0),
            "top_speed": p.get("top_speed", 0.0),
            "average_speed": p.get("average_speed", 0.0),
            "rating": p.get("rating", 6.0),
            "distance_is_real": p.get("distance_is_real", False),
        })

    context["all_players_json"] = json.dumps(all_players, default=str)
    context["home_team_color"] = match.home_kit_color or "#ef4444"
    context["away_team_color"] = match.away_kit_color or "#eab308"
    context["home_team_stats"] = context.get("team_stats", {}).get("home", {})
    context["away_team_stats"] = context.get("team_stats", {}).get("away", {})

    return render(request, "matches/results.html", context)


@login_required
def compare_hub(request, public_id):
    """
    Head-to-head tactical comparison hub for players and teams.
    Provides side-by-side performance radars, heatmaps, pass vectors,
    and key metric breakdowns.
    """
    match = get_object_or_404(Match, public_id=public_id)

    if match.status != Match.MatchStatus.COMPLETED:
        return redirect("matches:processing", public_id=match.public_id)

    context = build_match_report_context(match)
    context["is_uploader"] = request.user == match.uploaded_by
    context["match_report"] = Report.objects.filter(match=match, report_type=Report.ReportType.MATCH).first()

    # Serialize player data for dynamic client-side comparison and radar generation
    all_players = []
    for p in context.get("home_players", []):
        all_players.append({
            "lineup_id": p.get("lineup_id"),
            "player_id": p.get("player_id"),
            "name": p.get("name"),
            "jersey_number": p.get("jersey_number"),
            "position": p.get("position"),
            "team_side": "HOME",
            "team_name": match.home_team.name,
            "team_color": match.home_kit_color or "#ef4444",
            "minutes_played": p.get("minutes_played", 0),
            "goals": p.get("goals", 0),
            "assists": p.get("assists", 0),
            "shots": p.get("shots", 0),
            "shots_on_target": p.get("shots_on_target", 0),
            "shot_accuracy": p.get("shot_accuracy", 0.0),
            "xg": p.get("xg", 0.0),
            "passes_completed": p.get("passes_completed", 0),
            "passes_attempted": p.get("passes_attempted", 0),
            "pass_accuracy": p.get("pass_accuracy", 0.0),
            "key_passes": p.get("key_passes", 0),
            "dribbles_completed": p.get("dribbles_completed", 0),
            "tackles": p.get("tackles", 0),
            "interceptions": p.get("interceptions", 0),
            "clearances": p.get("clearances", 0),
            "distance_km": p.get("distance_km", 0.0),
            "top_speed": p.get("top_speed", 0.0),
            "average_speed": p.get("average_speed", 0.0),
            "rating": p.get("rating", 6.0),
            "distance_is_real": p.get("distance_is_real", False),
        })

    for p in context.get("away_players", []):
        all_players.append({
            "lineup_id": p.get("lineup_id"),
            "player_id": p.get("player_id"),
            "name": p.get("name"),
            "jersey_number": p.get("jersey_number"),
            "position": p.get("position"),
            "team_side": "AWAY",
            "team_name": match.away_team.name,
            "team_color": match.away_kit_color or "#eab308",
            "minutes_played": p.get("minutes_played", 0),
            "goals": p.get("goals", 0),
            "assists": p.get("assists", 0),
            "shots": p.get("shots", 0),
            "shots_on_target": p.get("shots_on_target", 0),
            "shot_accuracy": p.get("shot_accuracy", 0.0),
            "xg": p.get("xg", 0.0),
            "passes_completed": p.get("passes_completed", 0),
            "passes_attempted": p.get("passes_attempted", 0),
            "pass_accuracy": p.get("pass_accuracy", 0.0),
            "key_passes": p.get("key_passes", 0),
            "dribbles_completed": p.get("dribbles_completed", 0),
            "tackles": p.get("tackles", 0),
            "interceptions": p.get("interceptions", 0),
            "clearances": p.get("clearances", 0),
            "distance_km": p.get("distance_km", 0.0),
            "top_speed": p.get("top_speed", 0.0),
            "average_speed": p.get("average_speed", 0.0),
            "rating": p.get("rating", 6.0),
            "distance_is_real": p.get("distance_is_real", False),
        })

    context["all_players_json"] = json.dumps(all_players, default=str)
    context["home_team_color"] = match.home_kit_color or "#ef4444"
    context["away_team_color"] = match.away_kit_color or "#eab308"

    return render(request, "matches/compare.html", context)



from django.http import JsonResponse
from django.views.decorators.http import require_POST


@require_POST
@login_required
def detect_kit_colors(request):
    """
    Accepts a video file upload (same field as the main form), runs a
    quick color extraction against a short window near the start, and
    returns real swatches for the frontend to display.

    Doesn't create any Match/MatchVideo record — this runs BEFORE the
    user has finished the rest of the form, purely to power the color
    picker step. The video is saved to a temp file just long enough to
    run extraction, then deleted.
    """
    video_file = request.FILES.get("video")
    if not video_file:
        return JsonResponse({"error": "No video file provided."}, status=400)

    # Save to a temp file — extraction needs a real file path (uses
    # cv2.VideoCapture), can't work directly off the in-memory upload.
    suffix = os.path.splitext(video_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in video_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        from ai_engine.stage3_team_reid.kit_color_extraction import extract_kit_color_candidates
        result = extract_kit_color_candidates(tmp_path)
    except Exception as e:
        return JsonResponse({"error": f"Color detection failed: {e}"}, status=500)
    finally:
        os.unlink(tmp_path)

    return JsonResponse(result)


@login_required
def download_match_report_pdf(request, public_id):
    """
    Serves the publication-grade match report PDF for this match.
    If the report does not exist yet (or if ?regenerate=1 is passed),
    it generates it on-demand using generate_match_report().
    """
    match = get_object_or_404(Match, public_id=public_id)
    report = Report.objects.filter(match=match, report_type=Report.ReportType.MATCH).first()
    if not report or not report.pdf_file or request.GET.get("regenerate") == "1":
        from apps.reports.generator import generate_match_report
        report = generate_match_report(match)

    if not report or not report.pdf_file:
        raise Http404("Report PDF could not be generated.")

    as_attachment = request.GET.get("download") == "1"
    filename = f"{match.home_team.short_name}_vs_{match.away_team.short_name}_Match_Report.pdf"
    return FileResponse(
        report.pdf_file.open("rb"),
        as_attachment=as_attachment,
        filename=filename,
        content_type="application/pdf",
    )


@login_required
def export_match_json(request, public_id):
    """
    Exports comprehensive structured JSON match analytics, including
    team statistics, possession, tactical coordinates, passing networks,
    convex hull shapes, shots, passes, and lineups.
    """
    match = get_object_or_404(Match, public_id=public_id)
    ctx = build_match_report_context(match)

    payload = {
        "match": {
            "public_id": str(match.public_id),
            "date": match.match_date.isoformat() if hasattr(match.match_date, "isoformat") else str(match.match_date),
            "competition": match.competition or "",
            "stadium": match.stadium or "",
            "home_team": {
                "name": match.home_team.name,
                "short_name": match.home_team.short_name,
                "score": match.home_score,
            },
            "away_team": {
                "name": match.away_team.name,
                "short_name": match.away_team.short_name,
                "score": match.away_score,
            },
            "using_real_stats": ctx.get("using_real_stats", False),
        },
        "possession": {
            "home": ctx.get("home_possession", 50.0),
            "away": ctx.get("away_possession", 50.0),
        },
        "team_statistics": ctx.get("team_stats", {}),
        "tactical_data": ctx.get("tactical_data", {}),
        "shots": ctx.get("shots", []),
        "passes": ctx.get("passes", []),
        "lineups": {
            "home": ctx.get("home_players", []),
            "away": ctx.get("away_players", []),
        },
    }

    response = JsonResponse(payload, json_dumps_params={"indent": 2})
    if request.GET.get("download") == "1":
        response["Content-Disposition"] = f'attachment; filename="{match.home_team.short_name}_vs_{match.away_team.short_name}_analytics.json"'
    return response


@login_required
def export_match_csv(request, public_id, file_type):
    """
    Direct download endpoint for match computer-vision CSV artifacts:
    tracking, passes, shots, events, ball, or player_stats.
    """
    match = get_object_or_404(Match, public_id=public_id)
    files_obj = getattr(match, "files", None)
    if not files_obj:
        raise Http404("No tracking data files exist for this match.")

    mapping = {
        "tracking": (files_obj.player_tracking_csv, "player_tracking.csv"),
        "passes": (files_obj.passes_csv, "passes.csv"),
        "shots": (files_obj.shots_csv, "shots.csv"),
        "events": (files_obj.events_csv, "events.csv"),
        "ball": (files_obj.ball_tracking_csv, "ball_tracking.csv"),
        "player_stats": (files_obj.player_stats_csv, "player_stats.csv"),
    }

    entry = mapping.get(file_type)
    if not entry or not entry[0]:
        raise Http404(f"Requested CSV '{file_type}' has not been generated for this match.")

    csv_file, default_name = entry
    filename = f"{match.home_team.short_name}_vs_{match.away_team.short_name}_{default_name}"
    return FileResponse(
        csv_file.open("rb"),
        as_attachment=True,
        filename=filename,
        content_type="text/csv",
    )


@login_required
def export_match_clip(request, public_id):
    """
    Sub-clip extractor: cuts a high-definition MP4 clip from the match video
    between `start` and `end` seconds using FFmpeg.
    """
    import subprocess
    import re

    match = get_object_or_404(Match, public_id=public_id)
    video_obj = getattr(match, "video", None)
    if not video_obj:
        return JsonResponse({"error": "No video associated with this match."}, status=404)

    mode = request.GET.get("mode", "annotated").strip().lower()
    start_val = request.GET.get("start", "0")
    end_val = request.GET.get("end", "10")
    title = request.GET.get("title", "Match_Highlight").strip()

    try:
        start_sec = max(0.0, float(start_val))
        end_sec = max(start_sec + 0.5, float(end_val))
    except (ValueError, TypeError):
        start_sec = 0.0
        end_sec = 10.0

    # Limit maximum clip length to 90 seconds
    duration = min(90.0, end_sec - start_sec)
    end_sec = start_sec + duration

    # Resolve video source file
    video_file = None
    if mode == "annotated" and video_obj.annotated_video:
        video_file = video_obj.annotated_video
    elif video_obj.original_video:
        video_file = video_obj.original_video
    elif video_obj.annotated_video:
        video_file = video_obj.annotated_video

    if not video_file:
        return JsonResponse({"error": "Video footage file is missing on storage."}, status=404)

    input_path = video_file.path
    if not os.path.exists(input_path):
        return JsonResponse({"error": f"Video source file not found on disk at {input_path}"}, status=404)

    clean_title = re.sub(r"[^\w\-.]", "_", title).strip("_")
    if not clean_title:
        clean_title = "Clip"
    filename = f"{match.home_team.short_name}_vs_{match.away_team.short_name}_{clean_title}_{int(start_sec)}s-{int(end_sec)}s.mp4"

    tmp_dir = tempfile.gettempdir()
    out_file = os.path.join(tmp_dir, f"subclip_{match.public_id}_{int(start_sec * 10)}_{int(end_sec * 10)}.mp4")

    # Run FFmpeg to cut the clip cleanly
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_file
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, timeout=45)
        if res.returncode != 0 or not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            # Fallback to copy stream without re-encode if ultrafast fails
            cmd_copy = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", input_path,
                "-c", "copy",
                "-movflags", "+faststart",
                out_file
            ]
            subprocess.run(cmd_copy, capture_output=True, timeout=30)

        if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            return JsonResponse({"error": "Failed to generate video sub-clip with FFmpeg."}, status=500)

        return FileResponse(
            open(out_file, "rb"),
            as_attachment=True,
            filename=filename,
            content_type="video/mp4"
        )
    except subprocess.TimeoutExpired:
        return JsonResponse({"error": "Video clip generation timed out."}, status=504)
    except Exception as exc:
        return JsonResponse({"error": f"Clip generation error: {str(exc)}"}, status=500)