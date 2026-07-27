from django.contrib import admin

from .models import Player


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "team",
        "jersey_number",
        "position",
        "preferred_foot",
        "is_active",
    )

    list_filter = (
        "team",
        "position",
        "preferred_foot",
        "is_active",
    )

    search_fields = (
        "name",
    )

    ordering = (
        "team",
        "jersey_number",
    )