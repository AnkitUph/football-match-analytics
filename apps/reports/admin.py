from django.contrib import admin

from .models import Report


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "match",
        "report_type",
        "generated_by_ai",
        "created_at",
    )

    list_filter = (
        "report_type",
        "generated_by_ai",
    )

    search_fields = (
        "title",
    )

    ordering = ("-created_at",)