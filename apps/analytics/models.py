from django.db import models

from apps.matches.models import Match
from apps.players.models import Player
from apps.teams.models import Team


class PlayerStatistics(models.Model):

    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="player_statistics"
    )

    player = models.ForeignKey(
        Player,
        on_delete=models.CASCADE,
        related_name="statistics"
    )

    minutes_played = models.PositiveSmallIntegerField(default=0)

    goals = models.PositiveSmallIntegerField(default=0)

    assists = models.PositiveSmallIntegerField(default=0)

    shots = models.PositiveSmallIntegerField(default=0)

    shots_on_target = models.PositiveSmallIntegerField(default=0)

    passes_attempted = models.PositiveSmallIntegerField(default=0)

    passes_completed = models.PositiveSmallIntegerField(default=0)

    pass_accuracy = models.FloatField(default=0)

    key_passes = models.PositiveSmallIntegerField(default=0)

    dribbles_completed = models.PositiveSmallIntegerField(default=0)

    tackles = models.PositiveSmallIntegerField(default=0)

    interceptions = models.PositiveSmallIntegerField(default=0)

    clearances = models.PositiveSmallIntegerField(default=0)

    fouls_committed = models.PositiveSmallIntegerField(default=0)

    fouls_suffered = models.PositiveSmallIntegerField(default=0)

    yellow_cards = models.PositiveSmallIntegerField(default=0)

    red_cards = models.PositiveSmallIntegerField(default=0)

    offsides = models.PositiveSmallIntegerField(default=0)

    distance_covered = models.FloatField(
        default=0,
        help_text="Distance in meters"
    )

    top_speed = models.FloatField(
        default=0,
        help_text="km/h"
    )

    average_speed = models.FloatField(
        default=0,
        help_text="km/h"
    )

    xg = models.FloatField(default=0)

    rating = models.FloatField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "player_statistics"
        unique_together = ("match", "player")
        ordering = ["-rating"]
        indexes = [
            models.Index(fields=["match"]),
            models.Index(fields=["player"]),
        ]

    def __str__(self):
        return f"{self.player.name} - {self.match}"


class TeamStatistics(models.Model):

    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="team_statistics"
    )

    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="statistics"
    )

    possession = models.FloatField(default=0)

    goals = models.PositiveSmallIntegerField(default=0)

    shots = models.PositiveSmallIntegerField(default=0)

    shots_on_target = models.PositiveSmallIntegerField(default=0)

    passes_attempted = models.PositiveSmallIntegerField(default=0)

    passes_completed = models.PositiveSmallIntegerField(default=0)

    pass_accuracy = models.FloatField(default=0)

    corners = models.PositiveSmallIntegerField(default=0)

    offsides = models.PositiveSmallIntegerField(default=0)

    fouls = models.PositiveSmallIntegerField(default=0)

    yellow_cards = models.PositiveSmallIntegerField(default=0)

    red_cards = models.PositiveSmallIntegerField(default=0)

    xg = models.FloatField(default=0)

    total_distance = models.FloatField(
        default=0,
        help_text="Meters"
    )

    average_team_speed = models.FloatField(
        default=0,
        help_text="km/h"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "team_statistics"
        unique_together = ("match", "team")
        ordering = ["match"]
        indexes = [
            models.Index(fields=["match"]),
            models.Index(fields=["team"]),
        ]

    def __str__(self):
        return f"{self.team.name} - {self.match}"