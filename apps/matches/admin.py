from django.contrib import admin

from .models import Match, MatchVideo, MatchFiles


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = (
        "home_team",
        "away_team",
        "match_date",
        "competition",
        "status",
    )

    list_filter = (
        "status",
        "competition",
    )

    search_fields = (
        "home_team__name",
        "away_team__name",
    )

    ordering = ("-match_date",)


@admin.register(MatchVideo)
class MatchVideoAdmin(admin.ModelAdmin):
    list_display = (
        "match",
        "fps",
        "video_duration",
        "uploaded_at",
    )


@admin.register(MatchFiles)
class MatchFilesAdmin(admin.ModelAdmin):
    list_display = (
        "match",
        "created_at",
    )