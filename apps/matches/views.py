import random

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from apps.matches.models import Match, MatchLineup, MatchVideo
from apps.matches.tasks import process_match
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


def _dummy_player_rows(rng, lineup_qs, team):
    """
    Phase 5: placeholder stats only, to verify the results UI before the
    real CV pipeline (Phase 6) fills these in. Uses real lineup entries
    if they exist (Phase 3 optional lineup), otherwise generic rows.
    rng is seeded per-match so the same match shows the same numbers on
    every reload rather than reshuffling each time.

    Field names deliberately mirror apps.analytics.models.PlayerStatistics
    so swapping this out for real queries later is a drop-in replacement.
    """
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

        passes_attempted = rng.randint(15, 75)
        passes_completed = int(passes_attempted * rng.uniform(0.65, 0.95))
        pass_accuracy = round((passes_completed / passes_attempted) * 100, 1) if passes_attempted else 0

        rows.append({
            "jersey_number": jersey_number,
            "name": name,
            "position": position,
            "minutes_played": rng.choice([90, 90, 90, rng.randint(60, 89)]),
            "goals": rng.choice([0, 0, 0, 0, 1, 1, 2]) if position == "FWD" else rng.choice([0, 0, 0, 0, 1]),
            "assists": rng.choice([0, 0, 0, 1]),
            "shots": rng.randint(0, 6),
            "shots_on_target": rng.randint(0, 3),
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
            "distance_km": round(rng.uniform(7.5, 11.8), 2),
            "top_speed": round(rng.uniform(24, 34), 1),
            "average_speed": round(rng.uniform(7, 11), 1),
            "xg": round(rng.uniform(0, 1.2), 2),
            "rating": round(rng.uniform(5.8, 8.9), 1),
        })
    return rows


@login_required
def match_results(request, public_id):
    match = get_object_or_404(Match, public_id=public_id)

    # Don't show stats for a match that isn't actually done - send the
    # user to the processing page instead (which itself redirects to
    # results the moment the task finishes).
    if match.status != Match.MatchStatus.COMPLETED:
        return redirect("matches:processing", public_id=match.public_id)

    # Deterministic per-match "randomness" - same match always shows the
    # same dummy numbers instead of reshuffling on every page load.
    rng = random.Random(str(match.public_id))

    home_lineup = match.lineups.filter(side=MatchLineup.Side.HOME).order_by("jersey_number")
    away_lineup = match.lineups.filter(side=MatchLineup.Side.AWAY).order_by("jersey_number")

    home_players = _dummy_player_rows(rng, home_lineup, match.home_team)
    away_players = _dummy_player_rows(rng, away_lineup, match.away_team)

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

    context = {
        "match": match,
        "home_players": home_players,
        "away_players": away_players,
        "home_possession": home_possession,
        "away_possession": away_possession,
        "team_stats": team_stats,
        "shots": shots,
        "timeline": timeline,
    }
    return render(request, "matches/results.html", context)