import uuid

from django.db import models

from apps.matches.models import Match


class Report(models.Model):

    class ReportType(models.TextChoices):
        MATCH = "MATCH", "Match Report"
        PLAYER = "PLAYER", "Player Report"
        TEAM = "TEAM", "Team Report"

    public_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True
    )

    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="reports"
    )

    report_type = models.CharField(
        max_length=20,
        choices=ReportType.choices,
        default=ReportType.MATCH
    )

    title = models.CharField(
        max_length=200
    )

    pdf_file = models.FileField(
        upload_to="reports/pdfs/"
    )

    generated_by_ai = models.BooleanField(
        default=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        db_table = "reports"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["match"]),
            models.Index(fields=["report_type"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.match})"