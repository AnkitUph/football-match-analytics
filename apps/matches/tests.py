from django.test import TestCase, RequestFactory, Client
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.matches.models import Match, MatchVideo
from apps.matches.views import build_match_report_context, match_results
from apps.teams.models import Team

User = get_user_model()


class MatchVideoReplayTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="replay_test", email="replay_test@example.com", password="password123")
        self.home_team = Team.objects.create(name="Home FC", short_name="HOM", primary_color="#2563eb")
        self.away_team = Team.objects.create(name="Away FC", short_name="AWY", primary_color="#dc2626")
        self.match = Match.objects.create(
            uploaded_by=self.user,
            home_team=self.home_team,
            away_team=self.away_team,
            match_date="2026-09-22",
            status=Match.MatchStatus.COMPLETED,
        )
        self.factory = RequestFactory()

    def test_build_match_report_context_without_video(self):
        ctx = build_match_report_context(self.match)
        self.assertIsNone(ctx["annotated_video_url"])
        self.assertIsNone(ctx["original_video_url"])
        self.assertFalse(ctx["has_video"])

    def test_build_match_report_context_with_video(self):
        video_obj = MatchVideo.objects.create(
            match=self.match,
            original_video=SimpleUploadedFile("raw.mp4", b"dummy raw video content", content_type="video/mp4"),
            annotated_video=SimpleUploadedFile("annotated.mp4", b"dummy annotated video", content_type="video/mp4"),
        )
        ctx = build_match_report_context(self.match)
        self.assertTrue(ctx["has_video"])
        self.assertIsNotNone(ctx["annotated_video_url"])
        self.assertIsNotNone(ctx["original_video_url"])
        self.assertIn("annotated", ctx["annotated_video_url"])
        self.assertIn("raw", ctx["original_video_url"])

    def test_results_page_renders_video_replay_widget(self):
        MatchVideo.objects.create(
            match=self.match,
            original_video=SimpleUploadedFile("raw.mp4", b"dummy raw video content", content_type="video/mp4"),
            annotated_video=SimpleUploadedFile("annotated.mp4", b"dummy annotated video", content_type="video/mp4"),
        )
        request = self.factory.get(f"/matches/{self.match.public_id}/results/")
        request.user = self.user

        response = match_results(request, self.match.public_id)
        self.assertEqual(response.status_code, 200)

        content = response.content.decode("utf-8")
        self.assertIn("side-replay-col", content)
        self.assertIn("replay-card", content)
        self.assertIn("replay-video", content)
        self.assertIn("switchVideoMode", content)
        self.assertIn("seekVideoToFrame", content)
        self.assertIn("btn-mode-annotated", content)
        self.assertIn("btn-mode-raw", content)

    def test_results_page_renders_graceful_placeholder_when_no_video(self):
        request = self.factory.get(f"/matches/{self.match.public_id}/results/")
        request.user = self.user

        response = match_results(request, self.match.public_id)
        self.assertEqual(response.status_code, 200)

        content = response.content.decode("utf-8")
        self.assertIn("side-replay-col", content)
        self.assertIn("Video footage unavailable", content)


class MatchRealTrackingDataTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tracking_test", email="track_test@example.com", password="password123")
        self.home_team = Team.objects.create(name="Real Home", short_name="RHM", primary_color="#10b981")
        self.away_team = Team.objects.create(name="Real Away", short_name="RAW", primary_color="#6366f1")
        self.match = Match.objects.create(
            uploaded_by=self.user,
            home_team=self.home_team,
            away_team=self.away_team,
            match_date="2026-09-22",
            status=Match.MatchStatus.COMPLETED,
        )
        self.factory = RequestFactory()

    def test_real_tracking_and_event_analytics(self):
        from apps.matches.models import MatchFiles, MatchLineup, TrackPlayerIdentification
        from apps.analytics.models import TeamStatistics

        # Lineup
        p1 = MatchLineup.objects.create(
            match=self.match, team=self.home_team, side=MatchLineup.Side.HOME,
            jersey_number=10, player_name="Striker Home", position="FWD"
        )
        p2 = MatchLineup.objects.create(
            match=self.match, team=self.away_team, side=MatchLineup.Side.AWAY,
            jersey_number=4, player_name="Defender Away", position="DEF"
        )

        TrackPlayerIdentification.objects.create(
            match=self.match, track_id=101, lineup_entry=p1, is_auto_assigned=False
        )
        TrackPlayerIdentification.objects.create(
            match=self.match, track_id=202, lineup_entry=p2, is_auto_assigned=True
        )

        # Team Statistics
        TeamStatistics.objects.create(
            match=self.match, team=self.home_team,
            possession=56.5, shots=8, shots_on_target=4,
            passes_completed=45, passes_attempted=50, pass_accuracy=90.0,
            corners=3, total_distance=12000.0, average_team_speed=7.8, xg=1.45
        )
        TeamStatistics.objects.create(
            match=self.match, team=self.away_team,
            possession=43.5, shots=5, shots_on_target=2,
            passes_completed=32, passes_attempted=40, pass_accuracy=80.0,
            corners=1, total_distance=11500.0, average_team_speed=7.4, xg=0.65
        )

        # Passes CSV & Player Stats CSV
        passes_data = (
            "frame_idx,start_frame,minute,passer_track_id,receiver_track_id,passer_team,receiver_team,start_x,start_y,end_x,end_y,distance_m,speed_mps,is_completed\n"
            "50,25,1,101,102,team_a,team_a,10.0,5.0,22.0,8.0,12.4,8.5,True\n"
            "120,95,2,202,,team_b,, -15.0, -10.0, 5.0, 0.0,22.4,12.1,False\n"
        )
        player_stats_data = (
            "track_id,team,jersey_number,jersey_conf,distance_m,frames_tracked,passes_completed,passes_attempted,pass_accuracy,shots,shots_on_target,xg,top_speed,average_speed,tackles,interceptions,clearances,dribbles_completed,key_passes,rating\n"
            "101,team_a,10,,3500.0,500,20,22,90.9,3,2,0.65,28.5,8.2,3,2,1,2,1,7.8\n"
            "202,team_b,4,,4100.0,600,15,18,83.3,1,0,0.08,27.1,7.5,5,4,6,0,0,7.2\n"
        )

        files = MatchFiles.objects.create(
            match=self.match,
            passes_csv=SimpleUploadedFile("passes.csv", passes_data.encode("utf-8")),
            player_stats_csv=SimpleUploadedFile("player_stats.csv", player_stats_data.encode("utf-8")),
        )

        ctx = build_match_report_context(self.match)
        self.assertTrue(ctx["using_real_stats"])
        self.assertEqual(len(ctx["passes"]), 2)
        self.assertEqual(ctx["passes"][0]["passer"], "#10 Striker Home")
        self.assertTrue(ctx["passes"][0]["is_completed"])
        self.assertEqual(ctx["passes"][1]["passer"], "#4 Defender Away")
        self.assertFalse(ctx["passes"][1]["is_completed"])

        # Check defensive totals
        self.assertEqual(ctx["team_stats"]["home"]["tackles"], 3)
        self.assertEqual(ctx["team_stats"]["home"]["interceptions"], 2)
        self.assertEqual(ctx["team_stats"]["home"]["clearances"], 1)
        self.assertEqual(ctx["team_stats"]["home"]["shot_accuracy"], 50.0)

        self.assertEqual(ctx["team_stats"]["away"]["tackles"], 5)
        self.assertEqual(ctx["team_stats"]["away"]["interceptions"], 4)
        self.assertEqual(ctx["team_stats"]["away"]["clearances"], 6)
        self.assertEqual(ctx["team_stats"]["away"]["shot_accuracy"], 40.0)

        # Check web response
        request = self.factory.get(f"/matches/{self.match.public_id}/results/")
        request.user = self.user
        response = match_results(request, self.match.public_id)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")

        self.assertIn("Attacking &amp; Finishing", content)
        self.assertIn("Passing &amp; Possession", content)
        self.assertIn("Defending &amp; Ball Recovery", content)
        self.assertIn("Tackles Won", content)
        self.assertIn("Interceptions", content)
        self.assertIn("Clearances", content)
        self.assertIn("badge-tracked", content)
        self.assertIn("badge-unverified", content)


class TacticalPitchVisualizationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tactical_user", email="tactical@example.com", password="password123")
        self.home_team = Team.objects.create(name="Tactical Home", short_name="THM", primary_color="#10b981")
        self.away_team = Team.objects.create(name="Tactical Away", short_name="TAW", primary_color="#6366f1")
        self.match = Match.objects.create(
            uploaded_by=self.user,
            home_team=self.home_team,
            away_team=self.away_team,
            match_date="2026-09-22",
            status=Match.MatchStatus.COMPLETED,
        )
        self.factory = RequestFactory()

    def test_tactical_data_fallback_when_no_tracking(self):
        ctx = build_match_report_context(self.match)
        tactical = ctx.get("tactical_data")
        self.assertIsNotNone(tactical)
        self.assertIn("home", tactical)
        self.assertIn("away", tactical)

        home_t = tactical["home"]
        self.assertEqual(len(home_t["nodes"]), 11)
        self.assertIn("shape", home_t)
        self.assertGreater(home_t["shape"]["area_sqm"], 0)
        self.assertGreater(len(home_t["shape"]["hull_vertices"]), 2)
        self.assertIn("centroid", home_t["shape"])

        # Check template rendering
        request = self.factory.get(f"/matches/{self.match.public_id}/results/")
        request.user = self.user
        response = match_results(request, self.match.public_id)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")

        self.assertIn('id="tactical-data"', content)
        self.assertIn("btn-mode-shots", content)
        self.assertIn("btn-mode-passes", content)
        self.assertIn("btn-mode-shape", content)
        self.assertIn("tactical-hull-group", content)
        self.assertIn("tactical-network-links", content)
        self.assertIn("tactical-network-nodes", content)
        self.assertIn("hud-shape-metrics", content)

    def test_tactical_data_with_tracking_and_passes(self):
        from apps.matches.models import MatchFiles, MatchLineup, TrackPlayerIdentification

        # Create players for home and away
        p_home = []
        for i in range(1, 12):
            pos = "GK" if i == 1 else ("DEF" if i <= 5 else ("MID" if i <= 8 else "FWD"))
            p = MatchLineup.objects.create(
                match=self.match, team=self.home_team, side=MatchLineup.Side.HOME,
                jersey_number=i, player_name=f"Home Player {i}", position=pos
            )
            p_home.append(p)
            TrackPlayerIdentification.objects.create(
                match=self.match, track_id=100 + i, lineup_entry=p, is_auto_assigned=False
            )

        p_away = []
        for i in range(1, 12):
            pos = "GK" if i == 1 else ("DEF" if i <= 5 else ("MID" if i <= 8 else "FWD"))
            p = MatchLineup.objects.create(
                match=self.match, team=self.away_team, side=MatchLineup.Side.AWAY,
                jersey_number=i, player_name=f"Away Player {i}", position=pos
            )
            p_away.append(p)
            TrackPlayerIdentification.objects.create(
                match=self.match, track_id=200 + i, lineup_entry=p, is_auto_assigned=False
            )

        # Player tracking CSV with pitch positions
        tracking_rows = ["frame_idx,track_id,team,pitch_x,pitch_y,speed,distance"]
        # Generate some positions for home players (>= 5 frames per player)
        for i in range(1, 12):
            t_id = 100 + i
            # Spread across home side
            base_x = -40.0 + (i * 3.5)
            base_y = -20.0 + ((i % 5) * 10.0)
            for f_idx in [10, 20, 30, 40, 50, 60]:
                tracking_rows.append(f"{f_idx},{t_id},team_a,{base_x + (f_idx * 0.05)},{base_y + (f_idx * 0.02)},5.0,10.0")

        # Passes between home player 8 and home player 10
        passes_data = (
            "frame_idx,start_frame,minute,passer_track_id,receiver_track_id,passer_team,receiver_team,start_x,start_y,end_x,end_y,distance_m,speed_mps,is_completed\n"
            "50,25,1,108,110,team_a,team_a,-10.0,5.0,15.0,8.0,25.0,9.5,True\n"
            "80,60,2,108,110,team_a,team_a,-5.0,2.0,18.0,6.0,23.0,10.0,True\n"
        )

        MatchFiles.objects.create(
            match=self.match,
            player_tracking_csv=SimpleUploadedFile("tracking.csv", "\n".join(tracking_rows).encode("utf-8")),
            passes_csv=SimpleUploadedFile("passes.csv", passes_data.encode("utf-8")),
        )

        ctx = build_match_report_context(self.match)
        tactical = ctx.get("tactical_data")
        self.assertIsNotNone(tactical)
        home_t = tactical["home"]

        # Check nodes are tracked
        tracked_nodes = [n for n in home_t["nodes"] if n["is_tracked"]]
        self.assertEqual(len(tracked_nodes), 11)

        # Check links
        self.assertEqual(len(home_t["links"]), 1)
        link = home_t["links"][0]
        self.assertEqual(link["count"], 2)
        self.assertEqual(link["completed"], 2)

        # Check shape metrics
        self.assertGreater(home_t["shape"]["area_sqm"], 0)
        self.assertGreater(home_t["shape"]["length_m"], 0)
        self.assertGreater(home_t["shape"]["width_m"], 0)


class MatchExportEndpointsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="export_user", email="export@example.com", password="password123")
        self.home_team = Team.objects.create(name="Exp Home", short_name="EXH", primary_color="#10b981")
        self.away_team = Team.objects.create(name="Exp Away", short_name="EXA", primary_color="#6366f1")
        self.match = Match.objects.create(
            uploaded_by=self.user,
            home_team=self.home_team,
            away_team=self.away_team,
            match_date="2026-09-22",
            status=Match.MatchStatus.COMPLETED,
        )
        self.client = Client()
        self.client.force_login(self.user)

    def test_download_match_report_pdf(self):
        url = f"/matches/{self.match.public_id}/report/pdf/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("inline", response["Content-Disposition"])
        content = b"".join(response.streaming_content)
        self.assertTrue(content.startswith(b"%PDF"))

    def test_export_match_json(self):
        url = f"/matches/{self.match.public_id}/export/json/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("match", data)
        self.assertIn("team_statistics", data)
        self.assertIn("tactical_data", data)
        self.assertIn("shots", data)
        self.assertIn("passes", data)
        self.assertIn("lineups", data)

    def test_export_match_csv(self):
        from apps.matches.models import MatchFiles
        passes_data = "frame_idx,start_frame,minute,passer_track_id,receiver_track_id,passer_team,receiver_team,start_x,start_y,end_x,end_y,distance_m,speed_mps,is_completed\n"
        MatchFiles.objects.create(
            match=self.match,
            passes_csv=SimpleUploadedFile("passes.csv", passes_data.encode("utf-8")),
        )
        url = f"/matches/{self.match.public_id}/export/csv/passes/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        content = b"".join(response.streaming_content)
        self.assertIn(b"passer_track_id", content)

    def test_results_page_contains_export_dropdown(self):
        url = f"/matches/{self.match.public_id}/results/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("btn-export-toggle", content)
        self.assertIn("export-dropdown-menu", content)
        self.assertIn("/report/pdf/", content)
        self.assertIn("/export/json/", content)
        self.assertIn("/export/csv/", content)

    def test_results_page_contains_fotmob_heatmap_elements(self):
        url = f"/matches/{self.match.public_id}/results/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("fotmob-heat-legend", content)
        self.assertIn("Low Activity", content)
        self.assertIn("High Intensity", content)

    def test_player_heatmap_view(self):
        from apps.matches.models import MatchFiles, MatchLineup, TrackPlayerIdentification
        from django.core.files.uploadedfile import SimpleUploadedFile
        import base64

        tracking_csv = (
            "frame_idx,timestamp_ms,track_id,team,jersey_number,pitch_x,pitch_y,speed_mps\n"
            "0,0,1,home,10,12.5,5.2,4.1\n"
            "1,40,1,home,10,13.0,5.5,4.3\n"
            "2,80,1,home,10,13.5,6.0,4.2\n"
        )
        MatchFiles.objects.create(
            match=self.match,
            player_tracking_csv=SimpleUploadedFile("tracking.csv", tracking_csv.encode("utf-8")),
        )
        lineup_entry = MatchLineup.objects.create(
            match=self.match,
            team=self.home_team,
            side=MatchLineup.Side.HOME,
            player_name="Test Striker",
            jersey_number=10,
            position="FWD",
            is_starting=True,
        )
        TrackPlayerIdentification.objects.create(
            match=self.match,
            track_id=1,
            lineup_entry=lineup_entry,
        )

        url = f"/matches/{self.match.public_id}/player-heatmap/{lineup_entry.id}/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/png")
        self.assertTrue(response.content.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_render_heatmap_png_fotmob(self):
        from ai_engine.heatmap import render_heatmap_png
        # Empty positions
        self.assertIsNone(render_heatmap_png([]))

        # Single position
        png_single = render_heatmap_png([(10.0, 5.0)], attack_direction="right")
        self.assertIsNotNone(png_single)

        # Multi-positions
        positions = [(i * 2.0 - 20, (i % 5) * 4.0 - 10) for i in range(50)]
        png_multi = render_heatmap_png(positions, attack_direction="left")
        self.assertIsNotNone(png_multi)



