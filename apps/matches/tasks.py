import time

from celery import shared_task


@shared_task(bind=True)
def process_match(self, match_id):
    """
    Phase 4: simulated background processing.

    Moves a Match through PENDING -> PROCESSING -> COMPLETED (or FAILED on
    error), updating processing_progress along the way so the dashboard/
    results page can show real progress.

    This is intentionally fake work (time.sleep) standing in for the real
    pipeline. Phase 6 replaces the loop body with actual YOLO detection,
    ByteTrack, ball tracking, team classification, event detection, etc. -
    the status-transition contract (PENDING/PROCESSING/COMPLETED/FAILED)
    stays exactly the same, so nothing else in the app needs to change
    when that swap happens.
    """
    # Local import: avoids circular/app-loading issues at Celery worker
    # startup, since tasks.py is imported before Django apps are always
    # fully ready.
    from apps.matches.models import Match

    try:
        match = Match.objects.get(pk=match_id)
    except Match.DoesNotExist:
        return

    match.status = Match.MatchStatus.PROCESSING
    match.processing_progress = 0
    match.save(update_fields=["status", "processing_progress", "updated_at"])

    try:
        for percent in (25, 50, 75, 100):
            time.sleep(5)  # stand-in for real processing time
            match.processing_progress = percent
            match.save(update_fields=["processing_progress", "updated_at"])

        match.status = Match.MatchStatus.COMPLETED
        match.save(update_fields=["status", "updated_at"])

    except Exception:
        match.status = Match.MatchStatus.FAILED
        match.save(update_fields=["status", "updated_at"])
        raise