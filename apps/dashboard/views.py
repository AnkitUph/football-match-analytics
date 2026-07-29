from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from apps.matches.models import Match
from apps.teams.models import Team
from apps.players.models import Player


@login_required
def index(request):
    recent_matches = Match.objects.order_by("-created_at")[:5]

    context = {
        "total_matches": Match.objects.count(),
        "total_teams": Team.objects.count(),
        "total_players": Player.objects.count(),
        "recent_matches": recent_matches,
    }

    return render(request, "dashboard/index.html", context)