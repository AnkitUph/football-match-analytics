from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from apps.matches.models import Match
from apps.players.models import Player
from apps.teams.models import Team


@login_required
def index(request):
    matches = Match.objects.filter(uploaded_by=request.user)

    context = {
        # Phase 2 spec: match status breakdown
        "total_matches": matches.count(),
        "pending_matches": matches.filter(
            status=Match.MatchStatus.PENDING
        ).count(),
        "processing_matches": matches.filter(
            status=Match.MatchStatus.PROCESSING
        ).count(),
        "completed_matches": matches.filter(
            status=Match.MatchStatus.COMPLETED
        ).count(),
        "failed_matches": matches.filter(
            status=Match.MatchStatus.FAILED
        ).count(),

        # Extra counters you added
        "total_teams": Team.objects.count(),
        "total_players": Player.objects.count(),

        # Recent matches list
        "recent_matches": matches.order_by("-created_at")[:5],
    }
    return render(request, "dashboard/index.html", context)