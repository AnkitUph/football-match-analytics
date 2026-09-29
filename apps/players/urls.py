from django.urls import path

from . import views

app_name = "players"

urlpatterns = [
    path("", views.player_list, name="list"),
    path("<int:player_id>/", views.player_profile, name="profile"),
    path("<uuid:public_id>/", views.player_profile_by_uuid, name="profile_uuid"),
]