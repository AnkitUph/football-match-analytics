from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.shortcuts import redirect, render

from apps.matches.models import Match, MatchLineup, MatchVideo
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

        # Phase 4 will kick off the background task here, e.g.:
        #   process_match.delay(match.id)

        messages.success(
            request,
            "Match uploaded successfully. It's queued and will begin processing shortly.",
        )
        return redirect("dashboard:index")

    context = {
        "teams": teams,
        "team_types": Team.TeamType.choices,
        "positions": Player.Position.choices,
        "lineup_row_range": range(LINEUP_ROWS_PER_SIDE),
    }
    return render(request, "matches/upload.html", context)