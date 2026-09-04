import uuid

from django.conf import settings
from django.db import models

from apps.players.models import Player
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

    home_kit_color = models.CharField(
        max_length=7,
        blank=True,
        null=True,
        help_text="Hex color worn by home team outfield players this match"
    )

    home_gk_kit_color = models.CharField(
        max_length=7,
        blank=True,
        null=True,
        help_text="Hex color worn by home team goalkeeper this match"
    )

    away_kit_color = models.CharField(
        max_length=7,
        blank=True,
        null=True,
        help_text="Hex color worn by away team outfield players this match"
    )

    away_gk_kit_color = models.CharField(
        max_length=7,
        blank=True,
        null=True,
        help_text="Hex color worn by away team goalkeeper this match"
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


class MatchLineup(models.Model):
    """
    One row per player who appeared for one side in one match.

    Deliberately denormalized (player_name + jersey_number stored
    directly) rather than requiring every player to exist as a permanent
    Player record first — local/one-off matches often involve players who
    were never registered anywhere. `player` is an OPTIONAL link back to
    the persistent roster for teams that do have one; the app auto-links
    it when a Player with a matching team + jersey number already exists.
    """

    class Side(models.TextChoices):
        HOME = "HOME", "Home"
        AWAY = "AWAY", "Away"

    public_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True
    )

    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="lineups"
    )

    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="match_lineups"
    )

    side = models.CharField(
        max_length=4,
        choices=Side.choices
    )

    player = models.ForeignKey(
        Player,
        on_delete=models.SET_NULL,
        related_name="match_lineups",
        blank=True,
        null=True,
        help_text="Linked automatically if a roster Player with this jersey number already exists on the team"
    )

    player_name = models.CharField(
        max_length=100
    )

    jersey_number = models.PositiveSmallIntegerField()

    position = models.CharField(
        max_length=5,
        choices=Player.Position.choices
    )

    is_starting = models.BooleanField(
        default=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        db_table = "match_lineups"
        ordering = ["match", "side", "jersey_number"]
        unique_together = ("match", "side", "jersey_number")
        indexes = [
            models.Index(fields=["match"]),
            models.Index(fields=["team"]),
        ]

    def __str__(self):
        return f"#{self.jersey_number} {self.player_name} ({self.side}) - {self.match}"


class MatchCalibration(models.Model):
    """
    Manual per-match camera calibration: exactly 4 pixel<->pitch point
    pairs a human picked on one frame, used to bootstrap Stage 5's
    homography for this match's specific footage/camera setup.

    Optional. A match with no MatchCalibration still gets full Stage 1-4
    tracking (player_tracking_csv/ball_tracking_csv with blank
    pitch_x/pitch_y) — see apps/matches/tasks.py:process_match. Stage 5-6
    (pitch mapping, possession/pass/shot detection, real TeamStatistics,
    real heatmaps) only runs once this exists — see
    apps/matches/tasks.py:compute_pitch_mapping.

    Points are picked from a fixed preset list of real-world landmarks
    with known coordinates (apps/matches/pitch_landmarks.py) rather than
    the user typing meters by hand — avoids calibration errors from bad
    manual coordinate entry.
    """

    match = models.OneToOneField(
        Match,
        on_delete=models.CASCADE,
        related_name="calibration"
    )

    calibration_frame = models.PositiveIntegerField(
        help_text="Frame index (0-based) the 4 points below were picked on"
    )

    points = models.JSONField(
        help_text=(
            "Exactly 4 dicts: "
            "{'landmark_id', 'pixel_x', 'pixel_y', 'pitch_x', 'pitch_y'}"
        )
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        db_table = "match_calibrations"

    def __str__(self):
        return f"Calibration - {self.match}"

    def image_points(self):
        return [(p["pixel_x"], p["pixel_y"]) for p in self.points]

    def pitch_points(self):
        return [(p["pitch_x"], p["pitch_y"]) for p in self.points]


class TrackPlayerIdentification(models.Model):
    """
    Mapping from one merged tracklet (a track_id from Stage 5's within-
    shot stitching — see apps/matches/tasks.py:compute_pitch_mapping —
    persisted in player_tracking_csv/player_stats_csv) to a real
    MatchLineup entry. Either set by a human on the /identify/ page, or
    auto-guessed as a fallback (is_auto_assigned=True) so every tracked
    player shows SOME name/number rather than staying anonymous — see
    apps/matches/tasks.py:_auto_assign_unidentified_tracks. A guess may
    well be wrong; that's expected and fine (this project doesn't need
    production-grade accuracy here) as long as it's visibly flagged as
    unconfirmed until a human checks it via /identify/.

    Exists because jersey OCR (Stage 3c) was validated as non-viable on
    typical broadcast-resolution wide-shot footage — see
    ai_engine/stage3_team_reid/jersey_ocr.py's module docstring for the
    finding.

    unique_together on (match, lineup_entry) means one real player can't
    be double-assigned to two different tracked identities — a track_id
    CAN be left unassigned (anonymous), but a lineup player can only ever
    point at one track.
    """

    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="track_identifications"
    )

    track_id = models.PositiveIntegerField(
        help_text="Stitched tracklet's track_id, as it appears in player_tracking_csv/player_stats_csv for this match"
    )

    lineup_entry = models.ForeignKey(
        MatchLineup,
        on_delete=models.CASCADE,
        related_name="track_identifications"
    )

    is_auto_assigned = models.BooleanField(
        default=False,
        help_text="True = system guess, not yet confirmed by a human. False = a person explicitly chose this on /identify/."
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        db_table = "track_player_identifications"
        unique_together = [("match", "track_id"), ("match", "lineup_entry")]

    def __str__(self):
        marker = " (guess)" if self.is_auto_assigned else ""
        return f"track {self.track_id} -> {self.lineup_entry}{marker} ({self.match})"


class MatchGoal(models.Model):
    """
    One manually recorded goal. Deliberately NOT auto-detected — see
    project handoff notes: automatic goal detection was scoped out on
    purpose, since ball-tracking confidence is already documented as
    being least reliable during fast shots/saves, exactly when a
    goal-detector would need it most. This follows the same
    "human confirms, nothing auto-trusted" philosophy already used for
    calibration and player identification, just with no automatic
    suggestion step at all for this one — a human enters every row via
    the results page's "Manage Goals" panel.

    `team` is the side CREDITED on the scoreboard for this goal — for an
    own goal that's the BENEFITING team, not the scorer's own team.
    `scorer` is the actual MatchLineup player who put the ball in the
    net (their own net, if is_own_goal=True) — optional, since a human
    may want to log that a goal happened before confirming exactly who
    scored it.

    match.home_score / match.away_score are DERIVED from these rows —
    recomputed and saved by the add/delete views (see
    apps.matches.views._recompute_match_score) every time this table
    changes for a match. This table is the single source of truth for
    the scoreline, not the two integer fields on Match.

    A player's personal "Goals" stat (apps.matches.views._dummy_player_rows)
    only counts is_own_goal=False rows against that player as scorer —
    matches standard football statistics convention that an own goal is
    not credited to the scorer's own tally.
    """

    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="goals"
    )

    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="match_goals",
        help_text="The team credited on the scoreboard for this goal"
    )

    scorer = models.ForeignKey(
        MatchLineup,
        on_delete=models.SET_NULL,
        related_name="goals_scored",
        blank=True,
        null=True,
        help_text="Optional — a goal can be logged before the scorer is confirmed"
    )

    is_own_goal = models.BooleanField(
        default=False,
        help_text="If True, scorer (if set) belongs to the side OPPOSING `team`"
    )

    minute = models.PositiveSmallIntegerField(
        blank=True,
        null=True,
        help_text="Match minute, if known — informational only, not used in any scoring logic"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        db_table = "match_goals"
        ordering = ["match", "minute", "created_at"]
        indexes = [
            models.Index(fields=["match"]),
        ]

    def __str__(self):
        og = " (OG)" if self.is_own_goal else ""
        scorer_label = self.scorer.player_name if self.scorer else "Unknown scorer"
        return f"Goal{og}: {scorer_label} - {self.team.short_name} ({self.match})"