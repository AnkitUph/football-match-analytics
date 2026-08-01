
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
 
from .models import Player
 
 
@login_required
def player_list(request):
    players = Player.objects.select_related("team").order_by("team__name", "jersey_number")
    return render(request, "players/list.html", {"players": players})
 
