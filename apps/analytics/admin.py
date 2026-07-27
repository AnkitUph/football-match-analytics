from django.contrib import admin

from .models import PlayerStatistics, TeamStatistics


@admin.register(PlayerStatistics)
class PlayerStatisticsAdmin(admin.ModelAdmin):
    list_display = (
        "player",
        "match",
        "goals",
        "assists",
        "rating",
        "distance_covered",
    )

    list_filter = (
        "match",
        "player__team",
    )

    search_fields = (
        "player__name",
    )


@admin.register(TeamStatistics)
class TeamStatisticsAdmin(admin.ModelAdmin):
    list_display = (
        "team",
        "match",
        "possession",
        "goals",
        "xg",
    )

    list_filter = (
        "match",
        "team",
    )

    search_fields = (
        "team__name",
    )