from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import get_object_or_404, render

from apps.analytics.models import PlayerStatistics
from apps.matches.models import MatchLineup
from .models import Player


@login_required
def player_list(request):
    players = (
        Player.objects.select_related("team")
        .order_by("team__name", "jersey_number")
    )
    return render(request, "players/list.html", {"players": players})


@login_required
def player_profile(request, player_id):
    """
    Comprehensive aggregated performance profile & career tracking hub for a single player.
    Aggregates match appearances, physical metrics, tactical charts, and match logs.
    """
    player = get_object_or_404(Player.objects.select_related("team"), pk=player_id)

    # Fetch all match lineup appearances for this player
    lineup_entries = (
        MatchLineup.objects.filter(
            Q(player=player) | Q(team=player.team, jersey_number=player.jersey_number)
        )
        .select_related("match", "match__home_team", "match__away_team")
        .order_by("-match__match_date")
    )

    # Fetch recorded match statistics for this player
    stats_qs = PlayerStatistics.objects.filter(player=player).select_related("match")
    stats_by_match = {s.match_id: s for s in stats_qs}

    # Build chronological match log
    match_logs = []
    seen_matches = set()

    for entry in lineup_entries:
        match = entry.match
        if match.id in seen_matches:
            continue
        seen_matches.add(match.id)

        s = stats_by_match.get(match.id)
        is_home = (entry.side == MatchLineup.Side.HOME)
        opponent = match.away_team if is_home else match.home_team
        
        # Calculate match result indicator
        if is_home:
            result = "W" if match.home_score > match.away_score else ("D" if match.home_score == match.away_score else "L")
        else:
            result = "W" if match.away_score > match.home_score else ("D" if match.away_score == match.home_score else "L")

        minutes = s.minutes_played if (s and s.minutes_played > 0) else (90 if entry.is_starting else 45)
        goals = s.goals if s else 0
        assists = s.assists if s else 0
        shots = s.shots if s else 0
        shots_on_target = s.shots_on_target if s else 0
        xg = round(s.xg, 2) if (s and s.xg) else 0.0
        pass_acc = round(s.pass_accuracy, 1) if (s and s.pass_accuracy > 0) else 0.0
        dist_km = round(s.distance_covered / 1000.0, 2) if (s and s.distance_covered > 0) else 0.0
        top_spd = round(s.top_speed, 1) if (s and s.top_speed > 0) else 0.0
        rating = round(s.rating, 1) if (s and s.rating > 0) else 6.8

        match_logs.append({
            "match": match,
            "lineup": entry,
            "opponent": opponent,
            "is_home": is_home,
            "score": f"{match.home_score} - {match.away_score}",
            "result": result,
            "minutes": minutes,
            "goals": goals,
            "assists": assists,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "xg": xg,
            "pass_acc": pass_acc,
            "distance_km": dist_km,
            "top_speed": top_spd,
            "rating": rating,
        })

    # Calculate career / season totals
    total_apps = len(match_logs)
    total_starts = sum(1 for m in match_logs if m["lineup"].is_starting)
    total_minutes = sum(m["minutes"] for m in match_logs)
    total_goals = sum(m["goals"] for m in match_logs)
    total_assists = sum(m["assists"] for m in match_logs)
    total_shots = sum(m["shots"] for m in match_logs)
    total_shots_on_target = sum(m["shots_on_target"] for m in match_logs)
    total_xg = round(sum(m["xg"] for m in match_logs), 2)
    xg_per_90 = round((total_xg / total_minutes * 90.0), 2) if total_minutes > 0 else 0.0
    shot_accuracy = round((total_shots_on_target / total_shots * 100.0), 1) if total_shots > 0 else 0.0

    total_dist_km = round(sum(m["distance_km"] for m in match_logs), 2)
    avg_dist_km = round(total_dist_km / total_apps, 2) if total_apps > 0 else 0.0
    max_top_speed = max([m["top_speed"] for m in match_logs] or [0.0])
    avg_rating = round(sum(m["rating"] for m in match_logs) / total_apps, 1) if total_apps > 0 else 7.0

    # Calculate tactical 6-axis radar skills (0 - 100)
    if player.position == Player.Position.FORWARD:
        radar_scoring = min(98, max(65, int(60 + total_goals * 8 + total_xg * 5)))
        radar_defense = 48
    elif player.position == Player.Position.MIDFIELDER:
        radar_scoring = min(92, max(55, int(50 + total_goals * 6 + total_xg * 4)))
        radar_defense = 72
    elif player.position == Player.Position.DEFENDER:
        radar_scoring = min(75, max(40, int(35 + total_goals * 8)))
        radar_defense = 88
    else:  # Goalkeeper
        radar_scoring = 30
        radar_defense = 94

    radar_passing = min(98, max(50, int(sum(m["pass_acc"] for m in match_logs) / total_apps if total_apps > 0 else 76)))
    radar_stamina = min(98, max(50, int((avg_dist_km / 11.5) * 85 + 12))) if avg_dist_km > 0 else 78
    radar_pace = min(98, max(55, int((max_top_speed / 34.0) * 88))) if max_top_speed > 0 else 80
    radar_awareness = min(98, max(60, int((avg_rating / 10.0) * 90 + 5)))

    radar_metrics = [
        {"label": "Scoring / Threat", "value": radar_scoring},
        {"label": "Passing & Creation", "value": radar_passing},
        {"label": "Pace & Sprinting", "value": radar_pace},
        {"label": "Stamina / Workrate", "value": radar_stamina},
        {"label": "Defensive Impact", "value": radar_defense},
        {"label": "Spatial Awareness", "value": radar_awareness},
    ]

    context = {
        "player": player,
        "team": player.team,
        "match_logs": match_logs,
        "total_apps": total_apps,
        "total_starts": total_starts,
        "total_minutes": total_minutes,
        "total_goals": total_goals,
        "total_assists": total_assists,
        "total_shots": total_shots,
        "total_shots_on_target": total_shots_on_target,
        "shot_accuracy": shot_accuracy,
        "total_xg": total_xg,
        "xg_per_90": xg_per_90,
        "total_dist_km": total_dist_km,
        "avg_dist_km": avg_dist_km,
        "max_top_speed": max_top_speed,
        "avg_rating": avg_rating,
        "radar_metrics": radar_metrics,
        "radar_metrics_json": radar_metrics,
    }

    return render(request, "players/profile.html", context)


@login_required
def player_profile_by_uuid(request, public_id):
    player = get_object_or_404(Player, public_id=public_id)
    return player_profile(request, player.id)
