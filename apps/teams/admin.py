from django.contrib import admin

from .models import Team


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "short_name",
        "country",
        "league",
        "team_type",
    )

    list_filter = (
        "country",
        "league",
        "team_type",
    )

    search_fields = (
        "name",
        "short_name",
    )

    ordering = ("name",)