from django.urls import path

from . import views

app_name = "matches"

urlpatterns = [
    path("upload/", views.upload_match, name="upload"),
    path("<uuid:public_id>/processing/", views.match_processing, name="processing"),
    path("<uuid:public_id>/status/", views.match_status_api, name="status_api"),
    path("<uuid:public_id>/results/", views.match_results, name="results"),
]