from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WORLD_CUP_DIR = ROOT / "INT-World Cup" / "world_cup"
OUTPUT_DIR = WORLD_CUP_DIR / "2026"

RESULTS_PATH = DATA_DIR / "results.csv"
GOALSCORERS_PATH = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_PATH = DATA_DIR / "shootouts.csv"
FORMER_NAMES_PATH = DATA_DIR / "former_names.csv"
HISTORICAL_SUMMARY_PATH = WORLD_CUP_DIR / "FIFA_World_Cup_Results_All_Time_20260210_051012.csv"

FIFA_TEAMS_PAGE_URL = (
    "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/teams"
)
FIFA_FIXTURES_PAGE_URL = (
    "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures?country=&wtw-filter=ALL"
)
FIFA_FIXTURES_API_URL = (
    "https://api.fifa.com/api/v3/calendar/matches"
    "?language=en&IdCompetition=17&from=2026-06-11&to=2026-07-20&count=200"
)
FIFA_RANKING_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men?dateId=id11230"
FIFA_RANKING_API_URL = (
    "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
    "?rankingScheduleId={ranking_schedule_id}&count=300&language=en"
)
ELO_URL = (
    "https://www.international-football.net/elo-ratings-table"
    "?year={year}&month={month:02d}&day={day:02d}"
)

WORLD_CUP_COMPETITION_ID = "17"
WORLD_CUP_SEASON_ID = "285023"
TOURNAMENT_START_DATE = date(2026, 6, 11)
TOURNAMENT_END_DATE = date(2026, 7, 19)
FREEZE_TARGET_DATE = date(2026, 6, 10)
DEFAULT_BUILD_DATE = date(2026, 4, 3)
DEFAULT_FIFA_RANKING_SCHEDULE_ID = "FRS_Male_Football_20260119"
DEFAULT_FIFA_RANKING_SNAPSHOT_DATE = date(2026, 4, 1)
ROUND_ORDER = {
    "First Stage": ("GS", 1),
    "Round of 32": ("R32", 2),
    "Round of 16": ("R16", 3),
    "Quarter-final": ("QF", 4),
    "Semi-final": ("SF", 5),
    "Play-off for third place": ("3P", 6),
    "Final": ("F", 7),
}
MATCH_STATUS_MAP = {1: "scheduled"}
HOST_CODES = {"CAN", "MEX", "USA"}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

TOURNAMENT_TO_CANONICAL = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
    "Czechia": "Czechia",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
}

CANONICAL_ALIASES = {
    "Bosnia and Herzegovina": {"Bosnia-Herzegovina"},
    "Cape Verde": {"Cabo Verde"},
    "Czechia": {"Czech Republic"},
    "DR Congo": {"Congo DR", "Dem. Rep. of Congo"},
    "Iran": {"IR Iran"},
    "Ivory Coast": {"Côte d'Ivoire"},
    "South Korea": {"Korea Republic"},
    "Turkey": {"Türkiye"},
    "United States": {"USA"},
    "Curacao": {"Curaçao"},
}

ELO_NAME_ALIASES = {
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina"],
    "Cape Verde": ["Cape Verde"],
    "Curacao": ["Curaçao", "Curacao"],
    "Czechia": ["Czech Republic", "Czechia"],
    "DR Congo": ["Dem. Rep. of Congo", "DR Congo", "Congo DR"],
    "Iran": ["Iran", "IR Iran"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire"],
    "South Korea": ["South Korea", "Korea Republic"],
    "Turkey": ["Turkey", "Türkiye"],
    "United States": ["United States", "USA"],
}


@dataclass(frozen=True)
class QualifiedTeam:
    team_id: str
    fifa_code: str
    tournament_name: str
    canonical_name: str
    group_code: str


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-zA-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FIFA World Cup 2026 dataset.")
    parser.add_argument("--build-date", default=DEFAULT_BUILD_DATE.isoformat())
    parser.add_argument(
        "--fifa-ranking-schedule-id",
        default=DEFAULT_FIFA_RANKING_SCHEDULE_ID,
    )
    parser.add_argument(
        "--fifa-ranking-snapshot-date",
        default=DEFAULT_FIFA_RANKING_SNAPSHOT_DATE.isoformat(),
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def request_json(url: str) -> dict:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def request_text(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in fieldnames:
                value = row.get(field, "")
                if value is None:
                    serialized[field] = ""
                elif isinstance(value, bool):
                    serialized[field] = "TRUE" if value else "FALSE"
                else:
                    serialized[field] = str(value)
            writer.writerow(serialized)


def first_description(items: object) -> str:
    if isinstance(items, list) and items:
        return items[0].get("Description", "")
    return ""


def canonical_name_for_tournament(name: str) -> str:
    return TOURNAMENT_TO_CANONICAL.get(name, name)


def build_alias_maps(
    qualified_teams: dict[str, QualifiedTeam], former_names_rows: list[dict[str, str]]
) -> tuple[dict[str, str], list[tuple[str, date, date, str]]]:
    alias_map: dict[str, str] = {}
    dated_former_aliases: list[tuple[str, date, date, str]] = []

    for team in qualified_teams.values():
        alias_map[normalize_key(team.tournament_name)] = team.canonical_name
        alias_map[normalize_key(team.canonical_name)] = team.canonical_name
        for alias in CANONICAL_ALIASES.get(team.canonical_name, set()):
            alias_map[normalize_key(alias)] = team.canonical_name

    qualified_names = {team.canonical_name for team in qualified_teams.values()}
    for row in former_names_rows:
        current = canonical_name_for_tournament(row["current"])
        if current not in qualified_names:
            continue
        dated_former_aliases.append(
            (
                normalize_key(row["former"]),
                parse_iso_date(row["start_date"]),
                parse_iso_date(row["end_date"]),
                current,
            )
        )
    return alias_map, dated_former_aliases


def canonicalize_name(
    raw_name: str,
    match_date: date | None,
    alias_map: dict[str, str],
    dated_former_aliases: list[tuple[str, date, date, str]],
) -> str:
    normalized = normalize_key(raw_name)
    if normalized in alias_map:
        return alias_map[normalized]
    if match_date is not None:
        for former_key, start_date, end_date, canonical_name in dated_former_aliases:
            if normalized == former_key and start_date <= match_date <= end_date:
                return canonical_name
    return raw_name


def fetch_qualified_fixtures() -> list[dict]:
    payload = request_json(FIFA_FIXTURES_API_URL)
    fixtures = payload.get("Results", [])
    if len(fixtures) != 104:
        raise ValueError(f"Expected 104 fixtures from FIFA API, found {len(fixtures)}")
    return fixtures


def build_qualified_teams(fixtures: list[dict]) -> dict[str, QualifiedTeam]:
    teams: dict[str, QualifiedTeam] = {}
    for match in fixtures:
        if first_description(match.get("StageName")) != "First Stage":
            continue
        group_code = first_description(match.get("GroupName")).replace("Group ", "")
        for side in ("Home", "Away"):
            team_data = match.get(side)
            if not team_data:
                continue
            tournament_name = first_description(team_data.get("TeamName"))
            fifa_code = team_data.get("IdCountry") or ""
            teams[fifa_code] = QualifiedTeam(
                team_id=fifa_code,
                fifa_code=fifa_code,
                tournament_name=tournament_name,
                canonical_name=canonical_name_for_tournament(tournament_name),
                group_code=group_code,
            )
    if len(teams) != 48:
        raise ValueError(f"Expected 48 qualified teams, found {len(teams)}")
    return teams


def compute_world_cup_appearances(
    results_rows: list[dict[str, str]],
    qualified_teams: dict[str, QualifiedTeam],
    alias_map: dict[str, str],
    dated_former_aliases: list[tuple[str, date, date, str]],
) -> dict[str, int]:
    qualified_names = {team.canonical_name for team in qualified_teams.values()}
    appearances: dict[str, set[str]] = {name: set() for name in qualified_names}

    for row in results_rows:
        if row["tournament"] != "FIFA World Cup":
            continue
        match_date = parse_iso_date(row["date"])
        for side in ("home_team", "away_team"):
            canonical_name = canonicalize_name(
                row[side], match_date, alias_map, dated_former_aliases
            )
            if canonical_name in qualified_names:
                appearances[canonical_name].add(row["date"][:4])
    return {name: len(years) + 1 for name, years in appearances.items()}


def fetch_fifa_rankings(ranking_schedule_id: str) -> dict[str, dict]:
    payload = request_json(FIFA_RANKING_API_URL.format(ranking_schedule_id=ranking_schedule_id))
    rankings_by_code: dict[str, dict] = {}
    for row in payload.get("Results", []):
        if row.get("IdCountry"):
            rankings_by_code[row["IdCountry"]] = row
    return rankings_by_code


def fetch_elo_rankings(snapshot_date: date) -> dict[str, dict[str, object]]:
    html = request_text(
        ELO_URL.format(
            year=snapshot_date.year,
            month=snapshot_date.month,
            day=snapshot_date.day,
        )
    )
    soup = BeautifulSoup(html, "html.parser")
    ratings: dict[str, dict[str, object]] = {}
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) < 4:
                continue
            rank_token = cells[0].rstrip(" .")
            if not rank_token.isdigit():
                continue
            ratings[cells[2]] = {
                "elo_rank": int(rank_token),
                "elo_rating": int(cells[3]),
                "source_name": cells[2],
            }
    return ratings


def map_team_to_elo(team: QualifiedTeam, elo_rows: dict[str, dict[str, object]]) -> dict[str, object]:
    candidate_names = []
    if team.canonical_name in ELO_NAME_ALIASES:
        candidate_names.extend(ELO_NAME_ALIASES[team.canonical_name])
    candidate_names.extend([team.canonical_name, team.tournament_name])
    for candidate in candidate_names:
        if candidate in elo_rows:
            return elo_rows[candidate]
    raise KeyError(f"No Elo row found for {team.canonical_name} ({team.fifa_code})")


def build_groups_rows(qualified_teams: dict[str, QualifiedTeam]) -> list[dict[str, object]]:
    rows = []
    for team in sorted(qualified_teams.values(), key=lambda item: (item.group_code, item.tournament_name)):
        rows.append(
            {
                "edition_year": "2026",
                "group_code": team.group_code,
                "team_id": team.team_id,
                "team_name": team.tournament_name,
                "canonical_name": team.canonical_name,
                "seed_position": "",
            }
        )
    return rows


def build_rounds_rows(fixtures: list[dict]) -> list[dict[str, object]]:
    counts = Counter(first_description(match.get("StageName")) for match in fixtures)
    rows = []
    for round_name, (round_code, round_order) in ROUND_ORDER.items():
        rows.append(
            {
                "round_code": round_code,
                "round_name": round_name,
                "round_order": round_order,
                "match_count": counts.get(round_name, 0),
            }
        )
    return rows


def build_venues_rows(fixtures: list[dict], source_as_of: str) -> list[dict[str, object]]:
    venues: dict[str, dict[str, object]] = {}
    for match in fixtures:
        stadium = match.get("Stadium") or {}
        stadium_id = stadium.get("IdStadium")
        if not stadium_id:
            continue
        venues[stadium_id] = {
            "venue_id": stadium_id,
            "official_venue_name": first_description(stadium.get("Name")),
            "host_city": first_description(stadium.get("CityName")),
            "host_country": stadium.get("IdCountry", ""),
            "capacity": stadium.get("Capacity") or "",
            "source_url": FIFA_FIXTURES_PAGE_URL,
            "source_as_of": source_as_of,
        }
    return sorted(venues.values(), key=lambda row: row["official_venue_name"])


def derive_fixture_status(match: dict) -> str:
    if match.get("HomeTeamScore") is not None or match.get("AwayTeamScore") is not None:
        return "completed"
    return MATCH_STATUS_MAP.get(match.get("MatchStatus"), "unknown")


def build_fixtures_rows(
    fixtures: list[dict],
    qualified_teams: dict[str, QualifiedTeam],
    alias_map: dict[str, str],
    dated_former_aliases: list[tuple[str, date, date, str]],
    source_as_of: str,
) -> list[dict[str, object]]:
    rows = []
    for match in sorted(fixtures, key=lambda item: (item["Date"], item["MatchNumber"])):
        stage_name = first_description(match.get("StageName"))
        round_code, _ = ROUND_ORDER[stage_name]
        group_name = first_description(match.get("GroupName"))
        group_code = group_name.replace("Group ", "") if group_name else ""

        stadium = match.get("Stadium") or {}
        home = match.get("Home")
        away = match.get("Away")
        home_name = first_description(home.get("TeamName")) if home else ""
        away_name = first_description(away.get("TeamName")) if away else ""

        rows.append(
            {
                "match_id": match.get("IdMatch", ""),
                "match_number": match.get("MatchNumber", ""),
                "edition_year": "2026",
                "competition_id": match.get("IdCompetition", ""),
                "season_id": match.get("IdSeason", ""),
                "round_code": round_code,
                "round_name": stage_name,
                "group_code": group_code,
                "kickoff_datetime_utc": match.get("Date", ""),
                "kickoff_datetime_local": match.get("LocalDate", ""),
                "kickoff_date_local": (match.get("LocalDate") or "")[:10],
                "kickoff_time_local": (match.get("LocalDate") or "")[11:16],
                "venue_id": stadium.get("IdStadium", ""),
                "venue_name": first_description(stadium.get("Name")),
                "home_slot_label": home_name or match.get("PlaceHolderA", ""),
                "away_slot_label": away_name or match.get("PlaceHolderB", ""),
                "home_team_id": (home.get("IdCountry", "") if home else "") if home and home.get("IdCountry") in qualified_teams else "",
                "away_team_id": (away.get("IdCountry", "") if away else "") if away and away.get("IdCountry") in qualified_teams else "",
                "home_tournament_name": home_name,
                "away_tournament_name": away_name,
                "home_canonical_name": canonicalize_name(home_name, None, alias_map, dated_former_aliases) if home_name else "",
                "away_canonical_name": canonicalize_name(away_name, None, alias_map, dated_former_aliases) if away_name else "",
                "status": derive_fixture_status(match),
                "status_code": match.get("MatchStatus", ""),
                "home_score": match.get("HomeTeamScore", ""),
                "away_score": match.get("AwayTeamScore", ""),
                "went_to_extra_time": "FALSE",
                "penalties_home": match.get("HomeTeamPenaltyScore", ""),
                "penalties_away": match.get("AwayTeamPenaltyScore", ""),
                "winner_team_id": match.get("Winner", ""),
                "placeholder_a": match.get("PlaceHolderA", ""),
                "placeholder_b": match.get("PlaceHolderB", ""),
                "result_type": match.get("ResultType", ""),
                "last_verified_at": source_as_of,
                "source_url": FIFA_FIXTURES_PAGE_URL,
            }
        )
    return rows


def build_fifa_ranking_rows(
    qualified_teams: dict[str, QualifiedTeam],
    fifa_rankings: dict[str, dict],
    ranking_snapshot_date: date,
    source_as_of: str,
) -> list[dict[str, object]]:
    rows = []
    is_final_freeze = ranking_snapshot_date >= FREEZE_TARGET_DATE
    for team in sorted(qualified_teams.values(), key=lambda item: item.tournament_name):
        ranking = fifa_rankings.get(team.fifa_code)
        if not ranking:
            raise KeyError(f"Missing FIFA ranking row for {team.fifa_code}")
        rows.append(
            {
                "team_id": team.team_id,
                "canonical_name": team.canonical_name,
                "tournament_name": team.tournament_name,
                "fifa_code": team.fifa_code,
                "confederation": ranking.get("ConfederationName", ""),
                "snapshot_date": ranking_snapshot_date.isoformat(),
                "freeze_target_date": FREEZE_TARGET_DATE.isoformat(),
                "is_final_freeze": is_final_freeze,
                "snapshot_status": "final" if is_final_freeze else "provisional",
                "rank": ranking.get("Rank", ""),
                "previous_rank": ranking.get("PrevRank", ""),
                "points": round(float(ranking.get("TotalPoints", 0.0)), 6),
                "previous_points": round(float(ranking.get("PrevPoints", 0.0)), 6),
                "ranking_movement": ranking.get("RankingMovement", ""),
                "source_url": FIFA_RANKING_PAGE_URL,
                "source_as_of": source_as_of,
            }
        )
    return rows


def build_elo_rows(
    qualified_teams: dict[str, QualifiedTeam],
    elo_rankings: dict[str, dict[str, object]],
    snapshot_date: date,
    source_as_of: str,
) -> list[dict[str, object]]:
    rows = []
    is_final_freeze = snapshot_date >= FREEZE_TARGET_DATE
    source_url = ELO_URL.format(year=snapshot_date.year, month=snapshot_date.month, day=snapshot_date.day)
    for team in sorted(qualified_teams.values(), key=lambda item: item.tournament_name):
        elo_row = map_team_to_elo(team, elo_rankings)
        rows.append(
            {
                "team_id": team.team_id,
                "canonical_name": team.canonical_name,
                "tournament_name": team.tournament_name,
                "fifa_code": team.fifa_code,
                "snapshot_date": snapshot_date.isoformat(),
                "freeze_target_date": FREEZE_TARGET_DATE.isoformat(),
                "is_final_freeze": is_final_freeze,
                "snapshot_status": "final" if is_final_freeze else "provisional",
                "elo_rank": elo_row["elo_rank"],
                "elo_rating": elo_row["elo_rating"],
                "elo_source_name": elo_row["source_name"],
                "source_url": source_url,
                "source_as_of": source_as_of,
            }
        )
    return rows


def build_teams_rows(
    qualified_teams: dict[str, QualifiedTeam],
    fifa_rankings: dict[str, dict],
    appearances: dict[str, int],
    source_as_of: str,
) -> list[dict[str, object]]:
    rows = []
    for team in sorted(qualified_teams.values(), key=lambda item: (item.group_code, item.tournament_name)):
        ranking = fifa_rankings.get(team.fifa_code, {})
        rows.append(
            {
                "team_id": team.team_id,
                "canonical_name": team.canonical_name,
                "tournament_name": team.tournament_name,
                "fifa_code": team.fifa_code,
                "confederation": ranking.get("ConfederationName", ""),
                "group_code": team.group_code,
                "is_host": team.fifa_code in HOST_CODES,
                "qualification_path": "Host nation" if team.fifa_code in HOST_CODES else "Qualified",
                "world_cup_participations": appearances.get(team.canonical_name, 1),
                "squad_status": "pending",
                "source_url": FIFA_TEAMS_PAGE_URL,
                "source_as_of": source_as_of,
            }
        )
    return rows


def aggregate_goalscorers(goalscorers_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, int]]:
    aggregates: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in goalscorers_rows:
        match_key = (row["date"], row["home_team"], row["away_team"])
        team = row["team"]
        aggregates[match_key][f"{team}__goals"] += 1
        if row["penalty"] == "TRUE":
            aggregates[match_key][f"{team}__penalties"] += 1
        if row["own_goal"] == "TRUE":
            aggregates[match_key][f"{team}__own_goals"] += 1
    return aggregates


def aggregate_shootouts(shootouts_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    aggregates = {}
    for row in shootouts_rows:
        aggregates[(row["date"], row["home_team"], row["away_team"])] = {
            "winner": row["winner"],
            "first_shooter": row["first_shooter"],
        }
    return aggregates


def build_lead_in_rows(
    results_rows: list[dict[str, str]],
    goalscorer_aggregates: dict[tuple[str, str, str], dict[str, int]],
    shootout_aggregates: dict[tuple[str, str, str], dict[str, str]],
    qualified_teams: dict[str, QualifiedTeam],
    alias_map: dict[str, str],
    dated_former_aliases: list[tuple[str, date, date, str]],
) -> list[dict[str, object]]:
    canonical_to_code = {team.canonical_name: team.fifa_code for team in qualified_teams.values()}
    qualified_names = set(canonical_to_code)
    rows = []
    lead_in_id = 1

    for result in results_rows:
        match_date = parse_iso_date(result["date"])
        if match_date > FREEZE_TARGET_DATE:
            continue

        home_canonical = canonicalize_name(result["home_team"], match_date, alias_map, dated_former_aliases)
        away_canonical = canonicalize_name(result["away_team"], match_date, alias_map, dated_former_aliases)
        if home_canonical not in qualified_names and away_canonical not in qualified_names:
            continue

        match_key = (result["date"], result["home_team"], result["away_team"])
        goal_data = goalscorer_aggregates.get(match_key, {})
        shootout_data = shootout_aggregates.get(match_key, {})

        perspectives = [
            ("home", result["home_team"], home_canonical, result["away_team"], away_canonical, int(result["home_score"]), int(result["away_score"])),
            ("away", result["away_team"], away_canonical, result["home_team"], home_canonical, int(result["away_score"]), int(result["home_score"])),
        ]

        for perspective, source_team_name, canonical_team_name, source_opponent_name, canonical_opponent_name, team_score, opponent_score in perspectives:
            if canonical_team_name not in qualified_names:
                continue
            if team_score > opponent_score:
                result_label = "win"
            elif team_score < opponent_score:
                result_label = "loss"
            else:
                result_label = "draw"

            rows.append(
                {
                    "lead_in_id": f"lead_in_{lead_in_id:06d}",
                    "match_key": "|".join(match_key),
                    "date": result["date"],
                    "qualified_team_id": canonical_to_code[canonical_team_name],
                    "qualified_team_name": canonical_team_name,
                    "source_team_name": source_team_name,
                    "opponent_team_id": canonical_to_code.get(canonical_opponent_name, ""),
                    "opponent_name": canonical_opponent_name,
                    "source_opponent_name": source_opponent_name,
                    "perspective": perspective,
                    "is_home_perspective": perspective == "home",
                    "home_team": result["home_team"],
                    "away_team": result["away_team"],
                    "home_team_canonical": home_canonical,
                    "away_team_canonical": away_canonical,
                    "team_score": team_score,
                    "opponent_score": opponent_score,
                    "goal_difference": team_score - opponent_score,
                    "result": result_label,
                    "tournament": result["tournament"],
                    "city": result["city"],
                    "country": result["country"],
                    "neutral": result["neutral"] == "TRUE",
                    "days_before_tournament": (TOURNAMENT_START_DATE - match_date).days,
                    "goalscorer_events_for_team": goal_data.get(f"{source_team_name}__goals", 0),
                    "goalscorer_events_for_opponent": goal_data.get(f"{source_opponent_name}__goals", 0),
                    "penalty_goal_events_for_team": goal_data.get(f"{source_team_name}__penalties", 0),
                    "penalty_goal_events_for_opponent": goal_data.get(f"{source_opponent_name}__penalties", 0),
                    "own_goal_events_for_team": goal_data.get(f"{source_team_name}__own_goals", 0),
                    "own_goal_events_for_opponent": goal_data.get(f"{source_opponent_name}__own_goals", 0),
                    "decided_by_shootout": bool(shootout_data),
                    "shootout_winner": shootout_data.get("winner", ""),
                    "shootout_first_shooter": shootout_data.get("first_shooter", ""),
                }
            )
            lead_in_id += 1
    return rows


def build_edition_metadata(
    build_date: date,
    ranking_snapshot_date: date,
    elo_snapshot_date: date,
    fixtures_rows: list[dict[str, object]],
    teams_rows: list[dict[str, object]],
    groups_rows: list[dict[str, object]],
    venues_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "edition_year": "2026",
            "competition_id": WORLD_CUP_COMPETITION_ID,
            "season_id": WORLD_CUP_SEASON_ID,
            "competition_name": "FIFA World Cup 2026™",
            "hosts": "Canada / Mexico / USA",
            "host_country_codes": "CAN|MEX|USA",
            "tournament_start_date": TOURNAMENT_START_DATE.isoformat(),
            "tournament_end_date": TOURNAMENT_END_DATE.isoformat(),
            "pre_tournament_freeze_target_date": FREEZE_TARGET_DATE.isoformat(),
            "build_date": build_date.isoformat(),
            "teams_count": len(teams_rows),
            "groups_count": len({row["group_code"] for row in groups_rows}),
            "venues_count": len(venues_rows),
            "fixtures_count": len(fixtures_rows),
            "format_description": "48 teams, 12 groups of 4, Round of 32 onward",
            "fifa_ranking_snapshot_date": ranking_snapshot_date.isoformat(),
            "fifa_ranking_next_update_date": FREEZE_TARGET_DATE.isoformat(),
            "elo_snapshot_date": elo_snapshot_date.isoformat(),
            "squads_available": "FALSE",
            "squad_status": "pending",
            "data_status": "pre_tournament_provisional" if build_date < FREEZE_TARGET_DATE else "pre_tournament_final_freeze",
            "source_urls": "|".join([FIFA_TEAMS_PAGE_URL, FIFA_FIXTURES_PAGE_URL, FIFA_RANKING_PAGE_URL]),
            "source_as_of": build_date.isoformat(),
        }
    ]


def build_editions_summary(build_date: date) -> list[dict[str, object]]:
    historical_rows = load_csv(HISTORICAL_SUMMARY_PATH)
    summary_rows = []
    for row in sorted(historical_rows, key=lambda item: int(item["Year"])):
        summary_rows.append(
            {
                "edition_year": row["Year"],
                "host": row["Host"],
                "winner": row["Winner"],
                "runner_up": row["Runner_Up"],
                "third_place": row["Third_Place"],
                "fourth_place": row["Fourth_Place"],
                "final_score": row["Final_Score"],
                "venue": row["Venue"],
                "goals_scored": row["Goals_Scored"],
                "matches_played": row["Matches_Played"],
                "attendance": row["Attendance"],
                "top_scorer": row["Top_Scorer"],
                "winner_total_titles": row["Winner_Total_Titles"],
                "host_won": row["Host_Won"],
                "status": "completed",
                "source_url": str(HISTORICAL_SUMMARY_PATH.relative_to(ROOT)).replace("\\", "/"),
                "source_as_of": build_date.isoformat(),
            }
        )
    summary_rows.append(
        {
            "edition_year": "2026",
            "host": "Canada / Mexico / USA",
            "winner": "",
            "runner_up": "",
            "third_place": "",
            "fourth_place": "",
            "final_score": "",
            "venue": "",
            "goals_scored": "",
            "matches_played": "104",
            "attendance": "",
            "top_scorer": "",
            "winner_total_titles": "",
            "host_won": "",
            "status": "scheduled",
            "source_url": FIFA_FIXTURES_PAGE_URL,
            "source_as_of": build_date.isoformat(),
        }
    )
    return summary_rows


def validate_outputs(
    fixtures_rows: list[dict[str, object]],
    teams_rows: list[dict[str, object]],
    groups_rows: list[dict[str, object]],
    venues_rows: list[dict[str, object]],
    fifa_rows: list[dict[str, object]],
    elo_rows: list[dict[str, object]],
    lead_in_rows: list[dict[str, object]],
) -> None:
    if len(fixtures_rows) != 104:
        raise ValueError(f"Expected 104 fixtures, found {len(fixtures_rows)}")
    if len(teams_rows) != 48:
        raise ValueError(f"Expected 48 teams, found {len(teams_rows)}")
    if len({row['group_code'] for row in groups_rows}) != 12:
        raise ValueError("Expected 12 groups")
    if len(venues_rows) != 16:
        raise ValueError(f"Expected 16 venues, found {len(venues_rows)}")
    if len(fifa_rows) != 48:
        raise ValueError("Expected FIFA rankings for all 48 teams")
    if len(elo_rows) != 48:
        raise ValueError("Expected Elo ratings for all 48 teams")
    if any(int(row["days_before_tournament"]) < 1 for row in lead_in_rows):
        raise ValueError("Lead-in rows must all predate tournament kickoff")
    if any(count != 4 for count in Counter(row["group_code"] for row in groups_rows).values()):
        raise ValueError("Each group must contain exactly four teams")


def write_outputs(output_dir: Path, datasets: dict[str, tuple[list[dict[str, object]], list[str]]]) -> None:
    for filename, (rows, fieldnames) in datasets.items():
        write_csv(output_dir / filename, rows, fieldnames)


def build_manifest(output_dir: Path, build_date: date) -> None:
    manifest = {
        "build_date": build_date.isoformat(),
        "output_dir": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "files": sorted(path.name for path in output_dir.glob("*.csv")),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_date = parse_iso_date(args.build_date)
    ranking_snapshot_date = parse_iso_date(args.fifa_ranking_snapshot_date)
    output_dir = Path(args.output_dir)

    fixtures = fetch_qualified_fixtures()
    qualified_teams = build_qualified_teams(fixtures)
    former_names_rows = load_csv(FORMER_NAMES_PATH)
    alias_map, dated_former_aliases = build_alias_maps(qualified_teams, former_names_rows)

    results_rows = load_csv(RESULTS_PATH)
    goalscorers_rows = load_csv(GOALSCORERS_PATH)
    shootouts_rows = load_csv(SHOOTOUTS_PATH)
    appearances = compute_world_cup_appearances(results_rows, qualified_teams, alias_map, dated_former_aliases)

    fifa_rankings = fetch_fifa_rankings(args.fifa_ranking_schedule_id)
    elo_rankings = fetch_elo_rankings(build_date)
    source_as_of = build_date.isoformat()

    teams_rows = build_teams_rows(qualified_teams, fifa_rankings, appearances, source_as_of)
    groups_rows = build_groups_rows(qualified_teams)
    rounds_rows = build_rounds_rows(fixtures)
    venues_rows = build_venues_rows(fixtures, source_as_of)
    fixtures_rows = build_fixtures_rows(fixtures, qualified_teams, alias_map, dated_former_aliases, source_as_of)
    fifa_rows = build_fifa_ranking_rows(qualified_teams, fifa_rankings, ranking_snapshot_date, source_as_of)
    elo_rows = build_elo_rows(qualified_teams, elo_rankings, build_date, source_as_of)
    lead_in_rows = build_lead_in_rows(
        results_rows,
        aggregate_goalscorers(goalscorers_rows),
        aggregate_shootouts(shootouts_rows),
        qualified_teams,
        alias_map,
        dated_former_aliases,
    )
    edition_metadata_rows = build_edition_metadata(
        build_date, ranking_snapshot_date, build_date, fixtures_rows, teams_rows, groups_rows, venues_rows
    )
    editions_summary_rows = build_editions_summary(build_date)

    validate_outputs(fixtures_rows, teams_rows, groups_rows, venues_rows, fifa_rows, elo_rows, lead_in_rows)

    datasets = {
        "teams.csv": (
            teams_rows,
            ["team_id", "canonical_name", "tournament_name", "fifa_code", "confederation", "group_code", "is_host", "qualification_path", "world_cup_participations", "squad_status", "source_url", "source_as_of"],
        ),
        "groups.csv": (
            groups_rows,
            ["edition_year", "group_code", "team_id", "team_name", "canonical_name", "seed_position"],
        ),
        "rounds.csv": (rounds_rows, ["round_code", "round_name", "round_order", "match_count"]),
        "venues.csv": (
            venues_rows,
            ["venue_id", "official_venue_name", "host_city", "host_country", "capacity", "source_url", "source_as_of"],
        ),
        "fixtures.csv": (
            fixtures_rows,
            ["match_id", "match_number", "edition_year", "competition_id", "season_id", "round_code", "round_name", "group_code", "kickoff_datetime_utc", "kickoff_datetime_local", "kickoff_date_local", "kickoff_time_local", "venue_id", "venue_name", "home_slot_label", "away_slot_label", "home_team_id", "away_team_id", "home_tournament_name", "away_tournament_name", "home_canonical_name", "away_canonical_name", "status", "status_code", "home_score", "away_score", "went_to_extra_time", "penalties_home", "penalties_away", "winner_team_id", "placeholder_a", "placeholder_b", "result_type", "last_verified_at", "source_url"],
        ),
        "fifa_rank_snapshots.csv": (
            fifa_rows,
            ["team_id", "canonical_name", "tournament_name", "fifa_code", "confederation", "snapshot_date", "freeze_target_date", "is_final_freeze", "snapshot_status", "rank", "previous_rank", "points", "previous_points", "ranking_movement", "source_url", "source_as_of"],
        ),
        "elo_snapshots.csv": (
            elo_rows,
            ["team_id", "canonical_name", "tournament_name", "fifa_code", "snapshot_date", "freeze_target_date", "is_final_freeze", "snapshot_status", "elo_rank", "elo_rating", "elo_source_name", "source_url", "source_as_of"],
        ),
        "edition_metadata.csv": (
            edition_metadata_rows,
            ["edition_year", "competition_id", "season_id", "competition_name", "hosts", "host_country_codes", "tournament_start_date", "tournament_end_date", "pre_tournament_freeze_target_date", "build_date", "teams_count", "groups_count", "venues_count", "fixtures_count", "format_description", "fifa_ranking_snapshot_date", "fifa_ranking_next_update_date", "elo_snapshot_date", "squads_available", "squad_status", "data_status", "source_urls", "source_as_of"],
        ),
        "editions_summary.csv": (
            editions_summary_rows,
            ["edition_year", "host", "winner", "runner_up", "third_place", "fourth_place", "final_score", "venue", "goals_scored", "matches_played", "attendance", "top_scorer", "winner_total_titles", "host_won", "status", "source_url", "source_as_of"],
        ),
        "squads.csv": (
            [],
            ["edition_year", "team_id", "player_name", "position", "jersey_number", "date_of_birth", "club", "caps", "goals", "is_final_squad", "source_url", "source_as_of"],
        ),
        "team_results_lead_in.csv": (
            lead_in_rows,
            ["lead_in_id", "match_key", "date", "qualified_team_id", "qualified_team_name", "source_team_name", "opponent_team_id", "opponent_name", "source_opponent_name", "perspective", "is_home_perspective", "home_team", "away_team", "home_team_canonical", "away_team_canonical", "team_score", "opponent_score", "goal_difference", "result", "tournament", "city", "country", "neutral", "days_before_tournament", "goalscorer_events_for_team", "goalscorer_events_for_opponent", "penalty_goal_events_for_team", "penalty_goal_events_for_opponent", "own_goal_events_for_team", "own_goal_events_for_opponent", "decided_by_shootout", "shootout_winner", "shootout_first_shooter"],
        ),
    }
    write_outputs(output_dir, datasets)
    build_manifest(output_dir, build_date)
    print(json.dumps({"output_dir": str(output_dir), "teams": len(teams_rows), "groups": len({row["group_code"] for row in groups_rows}), "venues": len(venues_rows), "fixtures": len(fixtures_rows), "fifa_rank_rows": len(fifa_rows), "elo_rows": len(elo_rows), "lead_in_rows": len(lead_in_rows)}, indent=2))


if __name__ == "__main__":
    main()
