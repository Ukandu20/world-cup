from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .shared import build_historical_group_code_lookup


@dataclass(frozen=True)
class TournamentFormat:
    n_groups: int
    teams_per_group: int
    third_place_qualifiers: int
    total_teams: int
    stage_sequence: list[str]


FORMAT_MAP = {
    16: TournamentFormat(4, 4, 0, 16, ["group", "qf", "sf", "third_place", "final"]),
    24: TournamentFormat(6, 4, 0, 24, ["group", "r16", "qf", "sf", "third_place", "final"]),
    32: TournamentFormat(8, 4, 0, 32, ["group", "r16", "qf", "sf", "third_place", "final"]),
    48: TournamentFormat(12, 4, 8, 48, ["group", "r32", "r16", "qf", "sf", "third_place", "final"]),
}

FORMAT_BY_YEAR = {
    **{year: FORMAT_MAP[16] for year in [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978]},
    **{year: FORMAT_MAP[24] for year in [1982, 1986, 1990, 1994]},
    **{year: FORMAT_MAP[32] for year in [1998, 2002, 2006, 2010, 2014, 2018, 2022]},
    2026: FORMAT_MAP[48],
}


def get_format(year: int) -> TournamentFormat:
    if int(year) not in FORMAT_BY_YEAR:
        raise ValueError(f"No format defined for year {year}")
    return FORMAT_BY_YEAR[int(year)]


def extract_group_code(stage: str) -> str | None:
    if pd.isna(stage):
        return None
    match = re.search(r"group\s*([A-L])\b", str(stage), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _stage_team_ids(results: pd.DataFrame, pattern: str) -> set[str]:
    mask = results["stage"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
    return set(results.loc[mask, "team_id"].dropna().astype(str))


def get_actual_outcomes(results: pd.DataFrame) -> dict[str, object]:
    normalized_stage = results["stage"].astype(str).str.strip().str.lower()
    final = results[normalized_stage.eq("final")].copy()
    final["result_key"] = final["result"].astype(str).str.lower()
    winners = final[final["result_key"].isin({"w", "win"})]["team_id"].dropna().astype(str)
    champion = winners.iloc[0] if not winners.empty else None
    if champion is None and "shootout_winner" in final.columns:
        shootout_winners = final["shootout_winner"].dropna().astype(str)
        if not shootout_winners.empty:
            winner_name = shootout_winners.iloc[0]
            winner_rows = final[final["team"].astype(str).eq(winner_name)]
            if not winner_rows.empty:
                champion = str(winner_rows.iloc[0]["team_id"])

    return {
        "r32_teams": _stage_team_ids(results, r"round of 32|r32"),
        "r16_teams": _stage_team_ids(results, r"round of 16|r16"),
        "qf_teams": _stage_team_ids(results, r"quarter"),
        "sf_teams": _stage_team_ids(results, r"semi"),
        "final_teams": set(final["team_id"].dropna().astype(str)),
        "champion": champion,
    }


@dataclass
class HistoricalTournamentData:
    year: int
    tournament_id: str
    fmt: TournamentFormat
    teams: pd.DataFrame
    schedule: pd.DataFrame
    results: pd.DataFrame
    tournament_start_date: date
    tournament_end_date: date
    actual_outcomes: dict[str, object]

    @property
    def actual_r16_teams(self) -> set[str]:
        return set(self.actual_outcomes.get("r16_teams", set()))

    @property
    def actual_sf_teams(self) -> set[str]:
        return set(self.actual_outcomes.get("sf_teams", set()))

    @property
    def actual_champion(self) -> str | None:
        champion = self.actual_outcomes.get("champion")
        return None if champion is None else str(champion)


def _apply_group_codes(results: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_level = (
        schedule.rename(columns={"home_team_id": "home_team_code", "away_team_id": "away_team_code"})
        .loc[:, ["match_number", "date", "stage", "home_team", "away_team", "home_team_code", "away_team_code"]]
        .copy()
    )
    match_level["home_score"] = 0
    match_level["away_score"] = 0
    group_lookup = build_historical_group_code_lookup(match_level)
    schedule = schedule.copy()
    results = results.copy()
    schedule["group_code"] = schedule["home_team"].astype(str).map(group_lookup).where(
        schedule["stage"].astype(str).str.contains("group", case=False, na=False),
        None,
    )
    results["group_code"] = results["team"].astype(str).map(group_lookup).where(
        results["stage"].astype(str).str.contains("group", case=False, na=False),
        None,
    )
    return results, schedule


def load_historical_tournament(year: int, data: dict[str, pd.DataFrame]) -> HistoricalTournamentData:
    year = int(year)
    results = data["results"]
    schedule = data["schedule"]
    teams = data["teams"]

    r = results[pd.to_numeric(results["edition"], errors="coerce").eq(year)].copy()
    t = teams[pd.to_numeric(teams["year"], errors="coerce").eq(year)].copy()
    if r.empty:
        raise ValueError(f"No results data found for edition {year}")
    tournament_id = str(r["tournament_id"].iloc[0])
    s = schedule[schedule["tournament_id"].astype(str).eq(tournament_id)].copy()
    if s.empty:
        raise ValueError(f"No schedule data found for tournament_id {tournament_id}")

    r, s = _apply_group_codes(r, s)
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    s["date"] = pd.to_datetime(s["date"], errors="coerce")

    return HistoricalTournamentData(
        year=year,
        tournament_id=tournament_id,
        fmt=get_format(year),
        teams=t,
        schedule=s,
        results=r,
        tournament_start_date=pd.Timestamp(r["date"].min()).date(),
        tournament_end_date=pd.Timestamp(r["date"].max()).date(),
        actual_outcomes=get_actual_outcomes(r),
    )


def verify_loaders(data: dict[str, pd.DataFrame]) -> None:
    checks = [
        (2022, "ARG", {"FRA", "ARG", "MAR", "HRV"}, {"FRA", "ARG"}),
        (2018, "FRA", {"FRA", "HRV", "BEL", "ENG"}, {"FRA", "HRV"}),
        (2014, "GER", {"GER", "ARG", "NED", "BRA"}, {"GER", "ARG"}),
    ]
    for year, champion, sf_teams, final_teams in checks:
        td = load_historical_tournament(year, data)
        if td.actual_champion != champion:
            raise AssertionError(f"{year}: expected champion {champion}, got {td.actual_champion}")
        if not sf_teams.issubset(td.actual_sf_teams):
            raise AssertionError(f"{year}: SF teams mismatch. Expected {sf_teams}, got {td.actual_sf_teams}")
        actual_final_teams = set(td.actual_outcomes.get("final_teams", set()))
        if not final_teams.issubset(actual_final_teams):
            raise AssertionError(f"{year}: final teams mismatch. Expected {final_teams}, got {actual_final_teams}")
        if len(td.teams) != td.fmt.total_teams:
            raise AssertionError(f"{year}: expected {td.fmt.total_teams} teams, got {len(td.teams)}")
