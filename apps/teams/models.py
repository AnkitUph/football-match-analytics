import uuid

from django.db import models


class Team(models.Model):
    class TeamType(models.TextChoices):
        CLUB = "CLUB", "Club"
        NATIONAL = "NATIONAL", "National Team"

    public_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True
    )

    name = models.CharField(
        max_length=100,
        unique=True
    )

    short_name = models.CharField(
        max_length=10,
        unique=True
    )

    team_type = models.CharField(
        max_length=20,
        choices=TeamType.choices,
        default=TeamType.CLUB
    )

    country = models.CharField(
        max_length=100
    )

    league = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    logo = models.ImageField(
        upload_to="teams/logos/",
        blank=True,
        null=True
    )

    primary_color = models.CharField(
        max_length=7,
        help_text="Hex color code (e.g. #FF0000)"
    )

    secondary_color = models.CharField(
        max_length=7,
        blank=True,
        null=True,
        help_text="Hex color code (e.g. #FFFFFF)"
    )

    founded_year = models.PositiveIntegerField(
        blank=True,
        null=True
    )

    stadium = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    coach = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        db_table = "teams"
        ordering = ["name"]
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["country"]),
            models.Index(fields=["league"]),
        ]

    def __str__(self):
        return self.name