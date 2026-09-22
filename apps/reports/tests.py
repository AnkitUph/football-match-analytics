from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile

from apps.matches.models import Match, MatchFiles, MatchLineup, TrackPlayerIdentification
from apps.reports.models import Report
from apps.reports.generator import generate_match_report
from apps.teams.models import Team

User = get_user_model()


class MatchReportPDFTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="report_user", email="report@example.com", password="password123")
        self.home_team = Team.objects.create(name="Report Home", short_name="RHM", primary_color="#203A43")
        self.away_team = Team.objects.create(name="Report Away", short_name="RAW", primary_color="#4FAE79")
        self.match = Match.objects.create(
            uploaded_by=self.user,
            home_team=self.home_team,
            away_team=self.away_team,
            match_date="2026-09-22",
            status=Match.MatchStatus.COMPLETED,
        )

    def test_generate_match_report_uncalibrated(self):
        # Even without calibration or tracking data, report generation succeeds with fallback
        report = generate_match_report(self.match)
        self.assertIsNotNone(report)
        self.assertEqual(report.match, self.match)
        self.assertEqual(report.report_type, Report.ReportType.MATCH)
        self.assertTrue(report.pdf_file)

        content = report.pdf_file.read()
        self.assertTrue(content.startswith(b"%PDF"))
        self.assertGreater(len(content), 10000)

    def test_generate_match_report_calibrated(self):
        # Create lineup entries
        p_home = MatchLineup.objects.create(
            match=self.match, team=self.home_team, side=MatchLineup.Side.HOME,
            jersey_number=10, player_name="Captain Home", position="FWD"
        )
        p_away = MatchLineup.objects.create(
            match=self.match, team=self.away_team, side=MatchLineup.Side.AWAY,
            jersey_number=4, player_name="Defender Away", position="DEF"
        )
        TrackPlayerIdentification.objects.create(match=self.match, track_id=10, lineup_entry=p_home, is_auto_assigned=False)
        TrackPlayerIdentification.objects.create(match=self.match, track_id=20, lineup_entry=p_away, is_auto_assigned=True)

        # Tracking CSV & Passes CSV
        tracking_rows = ["frame_idx,track_id,team,pitch_x,pitch_y,speed,distance"]
        for f in range(1, 10):
            tracking_rows.append(f"{f},10,team_a,15.0,2.0,6.0,10.0")
            tracking_rows.append(f"{f},20,team_b,-15.0,-2.0,5.0,8.0")

        passes_data = (
            "frame_idx,start_frame,minute,passer_track_id,receiver_track_id,passer_team,receiver_team,start_x,start_y,end_x,end_y,distance_m,speed_mps,is_completed\n"
            "50,25,1,10,10,team_a,team_a,5.0,2.0,15.0,8.0,12.0,7.5,True\n"
        )

        MatchFiles.objects.create(
            match=self.match,
            player_tracking_csv=SimpleUploadedFile("tracking.csv", "\n".join(tracking_rows).encode("utf-8")),
            passes_csv=SimpleUploadedFile("passes.csv", passes_data.encode("utf-8")),
        )

        report = generate_match_report(self.match)
        self.assertIsNotNone(report)
        content = report.pdf_file.read()
        self.assertTrue(content.startswith(b"%PDF"))
        self.assertGreater(len(content), 20000)
