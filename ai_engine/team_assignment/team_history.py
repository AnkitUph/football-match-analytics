"""
Team assignment history.

Stores the latest/stable HOME/AWAY assignment for
each player/global identity.

Designed so the result can later be persisted
to Django/MySQL without changing the AI logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TeamHistoryEntry:

    player_id: int

    assignments: List[str] = field(
        default_factory=list
    )

    stable_team: str = "UNKNOWN"

    confidence: float = 0.0

    last_frame: int = -1


class TeamHistory:

    HOME = "HOME"
    AWAY = "AWAY"
    UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        minimum_votes: int = 3,
    ) -> None:

        self.minimum_votes = int(
            minimum_votes
        )

        self.entries: Dict[
            int,
            TeamHistoryEntry,
        ] = {}

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        player_id: int,
        team: str,
        confidence: float = 0.0,
        frame_index: int = -1,
    ) -> TeamHistoryEntry:

        player_id = int(player_id)

        if player_id not in self.entries:

            self.entries[player_id] = (
                TeamHistoryEntry(
                    player_id=player_id
                )
            )

        entry = self.entries[player_id]

        if team not in {
            self.HOME,
            self.AWAY,
            self.UNKNOWN,
        }:
            team = self.UNKNOWN

        entry.assignments.append(team)

        entry.last_frame = int(
            frame_index
        )

        entry.confidence = float(
            confidence
        )

        self._recalculate(
            entry
        )

        return entry

    # =========================================================
    # RECALCULATE
    # =========================================================

    def _recalculate(
        self,
        entry: TeamHistoryEntry,
    ) -> None:

        valid = [
            x
            for x in entry.assignments
            if x != self.UNKNOWN
        ]

        if not valid:

            entry.stable_team = self.UNKNOWN

            return

        home_votes = valid.count(
            self.HOME
        )

        away_votes = valid.count(
            self.AWAY
        )

        total = (
            home_votes
            + away_votes
        )

        if total < self.minimum_votes:

            entry.stable_team = self.UNKNOWN

            return

        if home_votes > away_votes:

            entry.stable_team = self.HOME

        elif away_votes > home_votes:

            entry.stable_team = self.AWAY

        else:

            entry.stable_team = self.UNKNOWN

    # =========================================================
    # GETTERS
    # =========================================================

    def get_team(
        self,
        player_id: int,
    ) -> str:

        entry = self.entries.get(
            int(player_id)
        )

        if entry is None:
            return self.UNKNOWN

        return entry.stable_team

    def get_entry(
        self,
        player_id: int,
    ) -> Optional[TeamHistoryEntry]:

        return self.entries.get(
            int(player_id)
        )

    def statistics(
        self,
    ) -> Dict[str, int]:

        stats = {
            self.HOME: 0,
            self.AWAY: 0,
            self.UNKNOWN: 0,
        }

        for entry in self.entries.values():

            stats[
                entry.stable_team
            ] += 1

        stats["total"] = len(
            self.entries
        )

        return stats

    # =========================================================
    # DATABASE-FRIENDLY EXPORT
    # =========================================================

    def to_records(
        self,
    ) -> List[dict]:

        records = []

        for entry in self.entries.values():

            records.append(
                {
                    "player_id": entry.player_id,
                    "team": entry.stable_team,
                    "confidence": entry.confidence,
                    "last_frame": entry.last_frame,
                    "assignment_count": len(
                        entry.assignments
                    ),
                }
            )

        return records