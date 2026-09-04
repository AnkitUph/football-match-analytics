import base64
import csv
from collections import defaultdict
import io
import json
import random
import tempfile
import os

import cv2

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from apps.analytics.models import TeamStatistics
from apps.reports.models import Report

from apps.matches.models import Match, MatchCalibration, MatchGoal, MatchLineup, MatchVideo, TrackPlayerIdentification
from apps.matches.pitch_landmarks import PITCH_LANDMARKS
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
            "shots": _safe_int(row.get("shots")),
            "xg": _safe_float(row.get("xg")),
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
            # Overrides the dummy figures computed above — but
            # passes_attempted/pass_accuracy and shots_on_target stay
            # dummy on purpose (see this function's docstring: no real
            # counterpart exists for either to pair with).
            passes_completed = real_stats["passes_completed"]
            shots = real_stats["shots"]
            xg = real_stats["xg"]
            distance_confirmed = real_stats["is_confirmed"]
            passes_completed_is_real = True
            shots_is_real = True
            xg_is_real = True
        else:
            distance_km = round(rng.uniform(7.5, 11.8), 2)
            distance_confirmed = False
            passes_completed_is_real = False
            shots_is_real = False
            xg_is_real = False

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

        rows.append({
            "jersey_number": jersey_number,
            "name": name,
            "lineup_entry_id": lineup_entry_id,
            "position": position,
            "minutes_played": rng.choice([90, 90, 90, rng.randint(60, 89)]),
            "goals": goals,
            "goals_is_real": goals_is_real,
            "goals_confirmed": goals_confirmed,
            "assists": rng.choice([0, 0, 0, 1]),
            "shots": shots,
            "shots_is_real": shots_is_real,
            "shots_on_target": shots_on_target,
            "passes_attempted": passes_attempted,
            "passes_completed": passes_completed,
            "pass_accuracy": pass_accuracy,
            "key_passes": rng.randint(0, 4),
            "dribbles_completed": rng.randint(0, 5),
            "tackles": rng.randint(0, 6),
            "interceptions": rng.randint(0, 5),
            "clearances": rng.randint(0, 8) if position in ("DEF", "GK") else rng.randint(0, 2),
            "fouls_committed": rng.randint(0, 4),
            "fouls_suffered": rng.randint(0, 4),
            "yellow_cards": rng.choice([0, 0, 0, 0, 1]),
            "red_cards": 0,
            "offsides": rng.randint(0, 3) if position == "FWD" else 0,
            "distance_km": distance_km,
            "distance_is_real": real_stats is not None,
            "distance_confirmed": distance_confirmed,
            "passes_completed_is_real": passes_completed_is_real,
            "top_speed": round(rng.uniform(24, 34), 1),
            "average_speed": round(rng.uniform(7, 11), 1),
            "xg": xg,
            "xg_is_real": xg_is_real,
            "rating": round(rng.uniform(5.8, 8.9), 1),
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
    Manual per-match calibration page. Only the uploader can calibrate
    their own match (unlike match_results, which anyone logged in can
    view) — a bad-faith or mistaken calibration silently corrupts real
    stats for everyone else looking at this match.
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

    existing_calibration = getattr(match, "calibration", None)
    has_real_stats = TeamStatistics.objects.filter(match=match).exists()

    context = {
        "match": match,
        "default_frame_idx": default_frame_idx,
        "existing_calibration": existing_calibration,
        "has_real_stats": has_real_stats,
        "landmarks": PITCH_LANDMARKS,
        "landmarks_json": json.dumps(PITCH_LANDMARKS),
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
def calibrate_save(request, public_id):
    """
    Validates and saves a MatchCalibration, then kicks off
    compute_pitch_mapping. If this match already has real TeamStatistics
    (i.e. it was calibrated before) and the request isn't explicitly
    confirming an overwrite, responds with needs_confirmation instead of
    saving — the calibration page's JS shows a confirm() dialog and
    resubmits with confirm_overwrite=true.
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

    if not isinstance(raw_points, list) or len(raw_points) != 4:
        return JsonResponse({"error": "Exactly 4 points are required."}, status=400)

    landmarks_by_id = {landmark["id"]: landmark for landmark in PITCH_LANDMARKS}
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
            defaults={"calibration_frame": frame_idx, "points": points},
        )
        transaction.on_commit(lambda: compute_pitch_mapping.delay(match.id))

    return JsonResponse({"ok": True, "redirect_url": reverse("matches:results", args=[match.public_id])})


def _load_track_summaries(files_obj, match=None, cap_per_team=10):
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
        detections_by_track[track_id].append((int(row["frame_idx"]), x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))

    crops_b64 = {tid: None for tid in track_ids}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return crops_b64

    try:
        for track_id in track_ids:
            dets = detections_by_track.get(track_id, [])
            if not dets:
                continue

            # Largest-area candidates first (closer to camera = more
            # pixels), THEN sorted by frame_idx so any seeking within
            # this one track's candidates at least moves forward.
            candidates = sorted(dets, key=lambda d: d[5], reverse=True)[:candidates_per_track]
            candidates.sort(key=lambda d: d[0])

            best_crop, best_sharpness = None, -1.0
            for frame_idx, x1, y1, x2, y2, _area in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                xi1, yi1, xi2, yi2 = int(max(x1, 0)), int(max(y1, 0)), int(x2), int(y2)
                crop = frame[yi1:yi2, xi1:xi2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_crop = crop

            if best_crop is not None:
                ok, buf = cv2.imencode(".jpg", best_crop)
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

    home_lineup = list(match.lineups.filter(side=MatchLineup.Side.HOME).order_by("jersey_number").values("id", "jersey_number", "player_name"))
    away_lineup = list(match.lineups.filter(side=MatchLineup.Side.AWAY).order_by("jersey_number").values("id", "jersey_number", "player_name"))

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

    for t in tracks:
        ident = existing.get(t["track_id"])
        t["assigned_lineup_entry_id"] = ident.lineup_entry_id if ident else None
        t["is_auto_assigned"] = ident.is_auto_assigned if ident else False
        t["lineup_options"] = home_lineup if t["team"] == Team.TEAM_A.value else away_lineup
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
            "shots": _safe_int(row.get("shots")),
            "xg": _safe_float(row.get("xg")),
            "is_confirmed": not ident.is_auto_assigned,
        }

        lineup_entry = ident.lineup_entry
        if lineup_entry.side == MatchLineup.Side.HOME:
            home_by_jersey[lineup_entry.jersey_number] = stats
        elif lineup_entry.side == MatchLineup.Side.AWAY:
            away_by_jersey[lineup_entry.jersey_number] = stats

    return home_by_jersey, away_by_jersey


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

    # Player-level stats always come from the dummy generator for now —
    # real per-player goals/etc need full event detection (not built).
    # Distance, completed passes, shots, and xg are partial exceptions:
    # for a player a human manually identified (see /identify/ page and
    # TrackPlayerIdentification), these come from real pitch-mapped
    # tracking / event detection (see _load_real_stats_by_jersey's
    # docstring for exactly what's real vs. inferred within that set).
    # Jersey-OCR-based stats are also checked as a fallback, in case OCR
    # is ever re-enabled on higher-resolution footage where it's
    # actually viable (see ai_engine/stage3_team_reid/jersey_ocr.py) —
    # manual assignment takes priority when both exist. Everything
    # else on a player's row is still the Phase 5 dummy generator's output.
    files_obj = getattr(match, "files", None)
    home_stats_ocr, away_stats_ocr = _load_real_stats_by_jersey(files_obj)
    home_stats_manual, away_stats_manual = _load_real_stats_by_assignment(match, files_obj)
    home_stats_by_jersey = {**home_stats_ocr, **home_stats_manual}
    away_stats_by_jersey = {**away_stats_ocr, **away_stats_manual}

    # Goals are a separate real-data source from the CV-pipeline CSV
    # fields above — see MatchGoal's docstring and _dummy_player_rows'
    # real_goals_by_lineup_id param. One shared dict works for both
    # sides since lineup_entry_id (== MatchGoal.scorer_id) is already
    # unique per match regardless of side.
    match_goals = list(MatchGoal.objects.filter(match=match).select_related("scorer", "team"))
    if match_goals:
        goals_by_lineup_id = defaultdict(int)
        for g in match_goals:
            if not g.is_own_goal and g.scorer_id is not None:
                goals_by_lineup_id[g.scorer_id] += 1
    else:
        goals_by_lineup_id = None

    home_players = _dummy_player_rows(rng, home_lineup, match.home_team, home_stats_by_jersey, goals_by_lineup_id)
    away_players = _dummy_player_rows(rng, away_lineup, match.away_team, away_stats_by_jersey, goals_by_lineup_id)

    # --- Team-level stats: real if this match has them, dummy otherwise ---
    # TeamStatistics only gets populated once a calibration exists for
    # this match's footage (Stage 5 dependency) — a fresh upload won't
    # have real rows yet, and will fall through to the dummy block
    # below, same as every match did before Stage 7 existed.
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

        def real_team_stat_block(ts):
            if ts is None:
                return {"shots": 0, "shots_on_target": 0, "passes": 0, "pass_accuracy": 0,
                        "corners": 0, "fouls": 0, "yellow_cards": 0, "red_cards": 0,
                        "xg": 0, "distance_km": 0}
            return {
                "shots": ts.shots,
                "shots_on_target": ts.shots_on_target,
                "passes": ts.passes_completed,
                "pass_accuracy": round(ts.pass_accuracy, 1),
                "corners": ts.corners,
                "fouls": ts.fouls,
                "yellow_cards": ts.yellow_cards,
                "red_cards": ts.red_cards,
                "xg": round(ts.xg, 2),
                "distance_km": round(ts.total_distance / 1000, 2),
            }

        team_stats = {"home": real_team_stat_block(home_ts), "away": real_team_stat_block(away_ts)}

        # Real heatmaps, generated on-demand from the saved CSV's
        # pitch_x/pitch_y columns — no persisted image, no new model
        # fields (see ai_engine/heatmap.py). None if this match's CSV
        # predates the pitch columns, or has no in-calibration rows for
        # a team yet (e.g. that team barely appears in the box-view
        # segment) — the template falls back to the placeholder pitch.
        from ai_engine.heatmap import render_heatmap_png

        home_positions, away_positions = _load_pitch_positions_by_team(getattr(match, "files", None))
        home_heatmap = render_heatmap_png(home_positions)
        away_heatmap = render_heatmap_png(away_positions)

        # No per-shot event data exists yet (that needs the events_csv/
        # shots_csv model work discussed separately) — shots stays empty
        # rather than dummy-filled when we ARE showing real team stats,
        # so the page doesn't mix real aggregate numbers with fabricated
        # shot-by-shot detail that contradicts them. Goals in the
        # timeline ARE real, though — unlike shots, they're human-
        # entered (MatchGoal), not auto-detected, so there's no
        # fabrication risk in showing them here.
        shots = []
        timeline = [{"minute": 0, "type": "kickoff", "description": "Kickoff"}]
        for g in match_goals:
            scorer_label = g.scorer.player_name if g.scorer else "Unknown scorer"
            og_label = " (OG)" if g.is_own_goal else ""
            timeline.append({
                "minute": g.minute if g.minute is not None else "",
                "type": "goal",
                "description": f"Goal{og_label} — {scorer_label} ({g.team.short_name})",
            })
        timeline.append({"minute": 90, "type": "fulltime", "description": "Full Time"})
        timeline.sort(key=lambda e: e["minute"] if isinstance(e["minute"], int) else 999)

    else:
        # --- Original Phase 5 dummy generation, unchanged ---
        home_heatmap = away_heatmap = None
        home_possession = rng.randint(38, 62)
        away_possession = 100 - home_possession

        def team_stat_block():
            return {
                "shots": rng.randint(8, 18),
                "shots_on_target": rng.randint(3, 9),
                "passes": rng.randint(300, 600),
                "pass_accuracy": rng.randint(75, 90),
                "corners": rng.randint(2, 9),
                "fouls": rng.randint(6, 14),
                "yellow_cards": rng.randint(0, 4),
                "red_cards": rng.choice([0, 0, 0, 0, 1]),
                "xg": round(rng.uniform(0.8, 2.9), 2),
                "distance_km": round(rng.uniform(105, 118), 1),
            }

        team_stats = {"home": team_stat_block(), "away": team_stat_block()}

        shot_outcomes = ["Goal", "Saved", "Blocked", "Off Target", "Woodwork"]
        shot_pool = [(p, "home") for p in home_players] + [(p, "away") for p in away_players]
        shots = []
        for _ in range(rng.randint(10, 18)):
            player, side = rng.choice(shot_pool)
            shots.append({
                "minute": rng.randint(1, 90),
                "player": player["name"],
                "side": side,
                "xg": round(rng.uniform(0.02, 0.75), 2),
                "outcome": rng.choice(shot_outcomes),
            })
        shots.sort(key=lambda s: s["minute"])

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

    return {
        "match": match,
        "home_players": home_players,
        "away_players": away_players,
        "home_possession": home_possession,
        "away_possession": away_possession,
        "team_stats": team_stats,
        "shots": shots,
        "timeline": timeline,
        "using_real_stats": using_real_stats,
        "home_heatmap": home_heatmap,
        "away_heatmap": away_heatmap,
        "has_calibration": hasattr(match, "calibration"),
        "match_goals": match_goals,
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
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
    png_b64 = render_heatmap_png(positions)
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
    return render(request, "matches/results.html", context)



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