from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from apps.matches.models import Match, MatchVideo
from apps.teams.models import Team


@login_required
def upload_match(request):
    teams = Team.objects.all().order_by("name")

    if request.method == "POST":
        home_team_id = request.POST.get("home_team")
        away_team_id = request.POST.get("away_team")
        match_date = request.POST.get("match_date")
        competition = request.POST.get("competition", "").strip()
        stadium = request.POST.get("stadium", "").strip()
        video_file = request.FILES.get("video")

        # Required fields
        if not home_team_id or not away_team_id or not match_date or not video_file:
            messages.error(request, "Team A, Team B, match date, and video are required.")
            return redirect("matches:upload")

        if home_team_id == away_team_id:
            messages.error(request, "Team A and Team B must be different.")
            return redirect("matches:upload")

        try:
            home_team = Team.objects.get(pk=home_team_id)
            away_team = Team.objects.get(pk=away_team_id)
        except Team.DoesNotExist:
            messages.error(request, "Selected team could not be found.")
            return redirect("matches:upload")

        # Basic file-type guard — real validation (codec/duration/etc.)
        # happens later in the AI pipeline (Phase 6), this just stops
        # obviously wrong uploads.
        allowed_extensions = (".mp4", ".mov", ".avi", ".mkv")
        if not video_file.name.lower().endswith(allowed_extensions):
            messages.error(request, "Please upload a video file (mp4, mov, avi, or mkv).")
            return redirect("matches:upload")

        match = Match.objects.create(
            uploaded_by=request.user,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            stadium=stadium or None,
            competition=competition or None,
            status=Match.MatchStatus.PENDING,
        )

        MatchVideo.objects.create(
            match=match,
            original_video=video_file,
        )

        # Phase 4 will kick off the background processing task here, e.g.:
        #   process_match.delay(match.id)

        messages.success(
            request,
            "Match uploaded successfully. It's queued and will begin processing shortly.",
        )
        return redirect("dashboard:index")

    context = {"teams": teams}
    return render(request, "matches/upload.html", context)