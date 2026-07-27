import uuid

from django.conf import settings
from django.db import models

from apps.teams.models import Team


class Match(models.Model):

    class MatchStatus(models.TextChoices):
        PENDING = "PENDING", "Pending"
        PROCESSING = "PROCESSING", "Processing"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"

    public_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True
    )

    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="matches"
    )

    home_team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="home_matches"
    )

    away_team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="away_matches"
    )

    match_date = models.DateField()

    stadium = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    competition = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    duration = models.PositiveIntegerField(
        help_text="Duration in minutes",
        default=90
    )

    status = models.CharField(
        max_length=20,
        choices=MatchStatus.choices,
        default=MatchStatus.PENDING
    )

    processing_progress = models.PositiveSmallIntegerField(
        default=0
    )

    home_score = models.PositiveSmallIntegerField(
        default=0
    )

    away_score = models.PositiveSmallIntegerField(
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        db_table = "matches"
        ordering = ["-match_date"]
        indexes = [
            models.Index(fields=["match_date"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"{self.home_team.short_name} vs {self.away_team.short_name}"



class MatchVideo(models.Model):

    match = models.OneToOneField(
        Match,
        on_delete=models.CASCADE,
        related_name="video"
    )

    original_video = models.FileField(
        upload_to="matches/videos/"
    )

    annotated_video = models.FileField(
        upload_to="matches/annotated/",
        blank=True,
        null=True
    )

    thumbnail = models.ImageField(
        upload_to="matches/thumbnails/",
        blank=True,
        null=True
    )

    video_duration = models.FloatField(
        blank=True,
        null=True,
        help_text="Duration in seconds"
    )

    fps = models.FloatField(
        blank=True,
        null=True
    )

    resolution_width = models.PositiveIntegerField(
        blank=True,
        null=True
    )

    resolution_height = models.PositiveIntegerField(
        blank=True,
        null=True
    )

    uploaded_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        db_table = "match_videos"

    def __str__(self):
        return f"Video - {self.match}"

class MatchFiles(models.Model):

    match = models.OneToOneField(
        Match,
        on_delete=models.CASCADE,
        related_name="files"
    )

    player_tracking_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    ball_tracking_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    passes_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    shots_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    events_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    player_stats_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    team_stats_csv = models.FileField(
        upload_to="matches/csv/",
        blank=True,
        null=True
    )

    match_report = models.FileField(
        upload_to="matches/reports/",
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
        db_table = "match_files"

    def __str__(self):
        return f"Files - {self.match}"