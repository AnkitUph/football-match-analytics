from django.urls import path

from . import views

app_name = "matches"

urlpatterns = [
    path("upload/", views.upload_match, name="upload"),
]