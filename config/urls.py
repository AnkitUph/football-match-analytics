from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.dashboard.urls")),
    path("accounts/", include("apps.accounts.urls")),
    path("matches/", include("apps.matches.urls")),
    path("players/", include("apps.players.urls")),
    path("teams/", include("apps.teams.urls")),
    #path("analytics/", include("apps.analytics.urls")),
    #path("reports/", include("apps.reports.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)