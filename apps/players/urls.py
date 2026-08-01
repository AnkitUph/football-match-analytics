from django.urls import path

from . import views

app_name = "players"

urlpatterns = [
    path("", views.player_list, name="list"),
]