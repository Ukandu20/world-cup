from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .historical_tournaments import HistoricalTournamentData, load_historical_tournament


ANCHOR_DATE = date(1998, 6, 10)
DEFAULT_HOLDOUT_YEARS = (2014, 2018, 2022)


@dataclass(frozen=True)
class ValidationFold:
    holdout_year: int
    anchor_date: date
    cutoff_date: date
    tournament: HistoricalTournamentData

    @classmethod
    def build(cls, holdout_year: int, data: dict) -> "ValidationFold":
        tournament = load_historical_tournament(int(holdout_year), data)
        return cls(
            holdout_year=int(holdout_year),
            anchor_date=ANCHOR_DATE,
            cutoff_date=tournament.tournament_start_date - timedelta(days=1),
            tournament=tournament,
        )


def build_all_folds(data: dict, holdout_years: tuple[int, ...] = DEFAULT_HOLDOUT_YEARS) -> list[ValidationFold]:
    return [ValidationFold.build(year, data) for year in holdout_years]
