from django.urls import path

from . import views

app_name = "matches"

urlpatterns = [
    path("upload/", views.upload_match, name="upload"),
    path("<uuid:public_id>/processing/", views.match_processing, name="processing"),
    path("<uuid:public_id>/status/", views.match_status_api, name="status_api"),
    path("<uuid:public_id>/results/", views.match_results, name="results"),
    path("<uuid:public_id>/compare/", views.compare_hub, name="compare_hub"),
    path("<uuid:public_id>/calibrate/", views.calibrate_match, name="calibrate"),
    path("<uuid:public_id>/calibrate/frame/", views.calibrate_frame, name="calibrate_frame"),
    path("<uuid:public_id>/calibrate/suggest/", views.calibrate_suggest, name="calibrate_suggest"),
    path("<uuid:public_id>/calibrate/save/", views.calibrate_save, name="calibrate_save"),
    path("<uuid:public_id>/calibrate/<int:calibration_id>/delete/", views.delete_calibration, name="delete_calibration"),
    path("<uuid:public_id>/identify/", views.identify_players, name="identify_players"),
    path("<uuid:public_id>/identify/save/", views.identify_players_save, name="identify_players_save"),
    path("<uuid:public_id>/goals/add/", views.add_goal, name="add_goal"),
    path("<uuid:public_id>/goals/<int:goal_id>/delete/", views.delete_goal, name="delete_goal"),
    path("<uuid:public_id>/player-heatmap/<int:lineup_entry_id>/", views.player_heatmap, name="player_heatmap"),
    path("detect-kit-colors/", views.detect_kit_colors, name="detect_kit_colors"),
    path("<uuid:public_id>/report/pdf/", views.download_match_report_pdf, name="download_report_pdf"),
    path("<uuid:public_id>/export/json/", views.export_match_json, name="export_json"),
    path("<uuid:public_id>/export/csv/<str:file_type>/", views.export_match_csv, name="export_csv"),
    path("<uuid:public_id>/export/clip/", views.export_match_clip, name="export_clip"),
]