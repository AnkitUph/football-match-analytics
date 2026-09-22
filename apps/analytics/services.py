import csv
import io
import logging
from collections import defaultdict

from apps.analytics.models import PlayerStatistics
from apps.matches.models import MatchGoal, TrackPlayerIdentification
from apps.players.models import Player

logger = logging.getLogger(__name__)


def sync_player_statistics(match) -> int:
    """
    Populates or updates apps.analytics.models.PlayerStatistics for all
    lineup players in a match.

    Pulls tracked metrics (distance, completed passes, shots, xG) from
    MatchFiles.player_stats_csv joined via TrackPlayerIdentification, and
    goals from MatchGoal. Auto-links a persistent Player record for any
    denormalized MatchLineup entry that lacks one.
    """
    files = getattr(match, "files", None)
    stats_by_track = {}
    if files and files.player_stats_csv:
        try:
            with files.player_stats_csv.open("rb") as f:
                reader = csv.DictReader(io.StringIO(f.read().decode("utf-8")))
                for row in reader:
                    try:
                        tid = int(row["track_id"])
                        stats_by_track[tid] = {
                            "distance_covered": float(row.get("distance_m", 0.0) or 0.0),
                            "passes_completed": int(row.get("passes_completed", 0) or 0),
                            "passes_attempted": int(row.get("passes_attempted", 0) or 0),
                            "pass_accuracy": float(row.get("pass_accuracy", 0.0) or 0.0),
                            "shots": int(row.get("shots", 0) or 0),
                            "shots_on_target": int(row.get("shots_on_target", 0) or 0),
                            "xg": float(row.get("xg", 0.0) or 0.0),
                            "top_speed": float(row.get("top_speed", 0.0) or 0.0),
                            "average_speed": float(row.get("average_speed", 0.0) or 0.0),
                            "tackles": int(row.get("tackles", 0) or 0),
                            "interceptions": int(row.get("interceptions", 0) or 0),
                            "clearances": int(row.get("clearances", 0) or 0),
                            "dribbles_completed": int(row.get("dribbles_completed", 0) or 0),
                            "key_passes": int(row.get("key_passes", 0) or 0),
                            "rating": float(row.get("rating", 6.0) or 6.0),
                        }
                    except (ValueError, KeyError):
                        continue
        except Exception:
            logger.exception("Failed to read player_stats_csv for match=%s", match.id)

    goals_by_lineup_id = defaultdict(int)
    for g in MatchGoal.objects.filter(match=match, is_own_goal=False):
        if g.scorer_id is not None:
            goals_by_lineup_id[g.scorer_id] += 1

    track_by_lineup_id = dict(
        TrackPlayerIdentification.objects.filter(match=match).values_list("lineup_entry_id", "track_id")
    )

    synced_count = 0
    for lineup in match.lineups.select_related("player", "team"):
        if lineup.player is None:
            player, _ = Player.objects.get_or_create(
                team=lineup.team,
                jersey_number=lineup.jersey_number,
                defaults={"name": lineup.player_name, "position": lineup.position},
            )
            lineup.player = player
            lineup.save(update_fields=["player"])
        else:
            player = lineup.player

        track_id = track_by_lineup_id.get(lineup.id)
        p_stats = stats_by_track.get(track_id, {})

        PlayerStatistics.objects.update_or_create(
            match=match,
            player=player,
            defaults={
                "distance_covered": p_stats.get("distance_covered", 0.0),
                "passes_completed": p_stats.get("passes_completed", 0),
                "passes_attempted": p_stats.get("passes_attempted", 0),
                "pass_accuracy": p_stats.get("pass_accuracy", 0.0),
                "shots": p_stats.get("shots", 0),
                "shots_on_target": p_stats.get("shots_on_target", 0),
                "xg": p_stats.get("xg", 0.0),
                "top_speed": p_stats.get("top_speed", 0.0),
                "average_speed": p_stats.get("average_speed", 0.0),
                "tackles": p_stats.get("tackles", 0),
                "interceptions": p_stats.get("interceptions", 0),
                "clearances": p_stats.get("clearances", 0),
                "dribbles_completed": p_stats.get("dribbles_completed", 0),
                "key_passes": p_stats.get("key_passes", 0),
                "rating": p_stats.get("rating", 6.0),
                "goals": goals_by_lineup_id.get(lineup.id, 0),
                "minutes_played": match.duration or 90,
            },
        )
        synced_count += 1

    return synced_count
