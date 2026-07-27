import uuid

from django.db import models
from apps.teams.models import Team


class Player(models.Model):

    class Position(models.TextChoices):
        GOALKEEPER = "GK", "Goalkeeper"
        DEFENDER = "DEF", "Defender"
        MIDFIELDER = "MID", "Midfielder"
        FORWARD = "FWD", "Forward"

    class PreferredFoot(models.TextChoices):
        LEFT = "LEFT", "Left"
        RIGHT = "RIGHT", "Right"
        BOTH = "BOTH", "Both"

    public_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True
    )

    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="players"
    )

    name = models.CharField(
        max_length=100
    )

    jersey_number = models.PositiveSmallIntegerField()

    position = models.CharField(
        max_length=5,
        choices=Position.choices
    )

    preferred_foot = models.CharField(
        max_length=5,
        choices=PreferredFoot.choices,
        default=PreferredFoot.RIGHT
    )

    height = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        help_text="Height in centimeters"
    )

    weight = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        help_text="Weight in kilograms"
    )

    photo = models.ImageField(
        upload_to="players/photos/",
        blank=True,
        null=True
    )

    is_active = models.BooleanField(
        default=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        db_table = "players"
        ordering = ["team", "jersey_number"]
        unique_together = ("team", "jersey_number")
        indexes = [
            models.Index(fields=["team"]),
            models.Index(fields=["name"]),
            models.Index(fields=["position"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.team.short_name} #{self.jersey_number})"