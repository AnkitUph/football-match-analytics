from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .models import Team


@login_required
def create_team(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        short_name = request.POST.get("short_name", "").strip().upper()
        country = request.POST.get("country", "").strip()
        league = request.POST.get("league", "").strip()
        team_type = request.POST.get("team_type", Team.TeamType.CLUB)
        primary_color = request.POST.get("primary_color", "").strip()
        secondary_color = request.POST.get("secondary_color", "").strip()
        logo = request.FILES.get("logo")

        if not name or not short_name or not primary_color:
            messages.error(request, "Team name, short name, and primary color are required.")
            return redirect("teams:create")

        if Team.objects.filter(name__iexact=name).exists():
            messages.error(request, "A team with this name already exists.")
            return redirect("teams:create")

        if Team.objects.filter(short_name__iexact=short_name).exists():
            messages.error(request, "A team with this short name already exists.")
            return redirect("teams:create")

        Team.objects.create(
            name=name,
            short_name=short_name,
            country=country,
            league=league or None,
            team_type=team_type,
            primary_color=primary_color,
            secondary_color=secondary_color or None,
            logo=logo,
        )

        messages.success(request, f"{name} added successfully.")
        return redirect("teams:create")

    teams = Team.objects.all().order_by("name")
    return render(
        request,
        "teams/create.html",
        {"teams": teams, "team_types": Team.TeamType.choices},
    )