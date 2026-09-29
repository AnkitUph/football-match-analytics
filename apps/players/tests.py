from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analytics.models import PlayerStatistics
from apps.matches.models import Match, MatchLineup
from apps.players.models import Player
from apps.teams.models import Team

User = get_user_model()


class PlayerProfileViewsTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="analyst_test",
            email="analyst@example.com",
            password="testpassword123",
        )
        self.client.force_login(self.user)

        self.team_home = Team.objects.create(
            name="FC Barcelona",
            short_name="BAR",
            primary_color="#004D98",
        )
        self.team_away = Team.objects.create(
            name="Real Madrid",
            short_name="RMA",
            primary_color="#FFFFFF",
        )

        self.player = Player.objects.create(
            team=self.team_home,
            name="Lionel Messi",
            jersey_number=10,
            position=Player.Position.FORWARD,
            preferred_foot=Player.PreferredFoot.LEFT,
            height=170,
            weight=72,
        )

        self.match = Match.objects.create(
            uploaded_by=self.user,
            home_team=self.team_home,
            away_team=self.team_away,
            match_date="2026-05-15",
            home_score=3,
            away_score=1,
            status=Match.MatchStatus.COMPLETED,
        )

        self.lineup = MatchLineup.objects.create(
            match=self.match,
            team=self.team_home,
            side=MatchLineup.Side.HOME,
            player=self.player,
            player_name=self.player.name,
            jersey_number=self.player.jersey_number,
            position=self.player.position,
            is_starting=True,
        )

        self.stats = PlayerStatistics.objects.create(
            match=self.match,
            player=self.player,
            minutes_played=90,
            goals=2,
            assists=1,
            shots=4,
            shots_on_target=3,
            passes_attempted=45,
            passes_completed=40,
            pass_accuracy=88.9,
            distance_covered=10500,
            top_speed=32.4,
            xg=1.45,
            rating=9.2,
        )

    def test_player_list_view(self):
        url = reverse("players:list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Lionel Messi")
        self.assertContains(res, "FC Barcelona")

    def test_player_profile_view_by_id(self):
        url = reverse("players:profile", kwargs={"player_id": self.player.id})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Lionel Messi")
        self.assertContains(res, "FC Barcelona")
        self.assertContains(res, "Tactical Attributes Matrix")
        self.assertEqual(res.context["total_apps"], 1)
        self.assertEqual(res.context["total_goals"], 2)
        self.assertEqual(res.context["total_assists"], 1)

    def test_player_profile_view_by_uuid(self):
        url = reverse("players:profile_uuid", kwargs={"public_id": self.player.public_id})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Lionel Messi")
