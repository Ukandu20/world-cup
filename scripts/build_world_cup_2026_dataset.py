from __future__ import annotations

import argparse
import csv
import json
import re
import time
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
ELO_BASE_URL = "https://www.eloratings.net"

WORLD_CUP_COMPETITION_ID = "17"
WORLD_CUP_SEASON_ID = "285023"
TOURNAMENT_START_DATE = date(2026, 6, 11)
TOURNAMENT_END_DATE = date(2026, 7, 19)
FREEZE_TARGET_DATE = date(2026, 6, 10)
DEFAULT_BUILD_DATE = date(2026, 4, 3)
DEFAULT_FIFA_RANKING_SCHEDULE_ID = "FRS_Male_Football_20260119"
DEFAULT_FIFA_RANKING_SNAPSHOT_DATE = date(2026, 4, 1)
REQUIRED_RATED_LEAD_IN_MATCHES = 20
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
FLAG_ICONS_CODE_BY_FIFA_CODE = {
    "ALG": "dz",
    "ARG": "ar",
    "AUS": "au",
    "AUT": "at",
    "BEL": "be",
    "BIH": "ba",
    "BRA": "br",
    "CAN": "ca",
    "CIV": "ci",
    "COD": "cd",
    "COL": "co",
    "CPV": "cv",
    "CRO": "hr",
    "CUW": "cw",
    "CZE": "cz",
    "ECU": "ec",
    "EGY": "eg",
    "ENG": "gb-eng",
    "ESP": "es",
    "FRA": "fr",
    "GER": "de",
    "GHA": "gh",
    "HAI": "ht",
    "IRN": "ir",
    "IRQ": "iq",
    "JOR": "jo",
    "JPN": "jp",
    "KOR": "kr",
    "KSA": "sa",
    "MAR": "ma",
    "MEX": "mx",
    "NED": "nl",
    "NOR": "no",
    "NZL": "nz",
    "PAN": "pa",
    "PAR": "py",
    "POR": "pt",
    "QAT": "qa",
    "RSA": "za",
    "SCO": "gb-sct",
    "SEN": "sn",
    "SUI": "ch",
    "SWE": "se",
    "TUN": "tn",
    "TUR": "tr",
    "URU": "uy",
    "USA": "us",
    "UZB": "uz",
}

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
    "Cape Verde": ["Cape Verde", "Cape Verde Islands", "Cabo Verde"],
    "Curacao": ["Curaçao", "Curacao"],
    "Czechia": ["Czech Republic", "Czechia"],
    "DR Congo": ["Dem. Rep. of Congo", "DR Congo", "Congo DR", "Dr Congo"],
    "Iran": ["Iran", "IR Iran"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire"],
    "Jordan": ["Jordan"],
    "South Korea": ["South Korea", "Korea Republic"],
    "Turkey": ["Turkey", "Türkiye"],
    "United States": ["United States", "USA"],
    "Uzbekistan": ["Uzbekistan"],
}


@dataclass(frozen=True)
class QualifiedTeam:
    team_id: str
    fifa_code: str
    tournament_name: str
    canonical_name: str
    group_code: str


@dataclass(frozen=True)
class MatchElo:
    team: str
    opponent: str
    team_score: int
    opponent_score: int
    team_elo_start: int
    opponent_elo_start: int
    team_elo_end: int
    opponent_elo_end: int
    team_elo_delta: int


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-zA-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def page_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^A-Za-z0-9 ]+", "", normalized)
    return normalized.replace(" ", "_")


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
    return response.content.decode("utf-8", errors="replace")


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


def decode_elo_delta(value: str) -> int:
    stripped = (
        value.strip()
        .replace("âˆ’", "-")
        .replace("Ã¢ÂˆÂ’", "-")
        .replace("Ä\x88\x92", "-")
        .replace("&minus;", "-")
    )
    if stripped in {"", "-", "âˆ’"}:
        return 0
    return int(stripped)


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

    for row in former_names_rows:
        current = canonical_name_for_tournament(row["current"])
        alias_map.setdefault(normalize_key(current), current)
        alias_map.setdefault(normalize_key(row["former"]), current)
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


def build_elo_team_data() -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    successor_map: dict[str, str] = {}
    for line in request_text(f"{ELO_BASE_URL}/teams.tsv").splitlines():
        if not line.strip():
            continue
        current, successor = line.split("\t")[:2]
        successor_map[current] = successor

    names_by_code: dict[str, list[str]] = {}
    code_by_name: dict[str, str] = {}
    for line in request_text(f"{ELO_BASE_URL}/en.teams.tsv").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0]
        names = parts[1:]
        names_by_code[code] = names
        for name in names:
            code_by_name.setdefault(normalize_key(name), code)
    return successor_map, names_by_code, code_by_name


def map_team_to_elo_code(
    team: QualifiedTeam,
    code_by_name: dict[str, str],
    elo_rankings: dict[str, dict[str, object]],
) -> str:
    candidate_names: list[str] = []
    try:
        candidate_names.append(str(map_team_to_elo(team, elo_rankings)["source_name"]))
    except KeyError:
        pass
    candidate_names.extend(ELO_NAME_ALIASES.get(team.canonical_name, []))
    candidate_names.extend([team.canonical_name, team.tournament_name])
    for candidate in candidate_names:
        code = code_by_name.get(normalize_key(candidate))
        if code:
            return code
    raise KeyError(f"No Elo code found for {team.canonical_name} ({team.fifa_code})")


def build_lead_in_match_elo_lookup(
    qualified_teams: dict[str, QualifiedTeam],
    elo_rankings: dict[str, dict[str, object]],
    alias_map: dict[str, str],
    dated_former_aliases: list[tuple[str, date, date, str]],
) -> tuple[dict[tuple[str, str, str, int, int], MatchElo], dict[tuple[str, str, int, int], MatchElo]]:
    successor_map, names_by_code, code_by_name = build_elo_team_data()
    teams_by_page: dict[str, set[str]] = defaultdict(set)

    for team in qualified_teams.values():
        elo_code = map_team_to_elo_code(team, code_by_name, elo_rankings)
        page_code = successor_map.get(elo_code, elo_code)
        page_slug = page_name(names_by_code[page_code][0])
        teams_by_page[page_slug].add(team.canonical_name)

    lookup: dict[tuple[str, str, str, int, int], MatchElo] = {}
    fallback_lookup: dict[tuple[str, str, int, int], MatchElo] = {}
    for index, (page_slug, teams_on_page) in enumerate(sorted(teams_by_page.items()), start=1):
        text = request_text(f"{ELO_BASE_URL}/{page_slug}.tsv")
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 16:
                continue
            match_date = f"{fields[0]}-{fields[1]}-{fields[2]}"
            try:
                parse_iso_date(match_date)
            except ValueError:
                continue
            code1 = fields[3]
            code2 = fields[4]
            if code1 not in names_by_code or code2 not in names_by_code:
                continue
            team1 = canonicalize_name(names_by_code[code1][0], parse_iso_date(match_date), alias_map, dated_former_aliases)
            team2 = canonicalize_name(names_by_code[code2][0], parse_iso_date(match_date), alias_map, dated_former_aliases)
            score1 = int(fields[5])
            score2 = int(fields[6])
            delta1 = decode_elo_delta(fields[9])
            post1 = int(fields[10])
            post2 = int(fields[11])
            pre1 = post1 - delta1
            pre2 = post2 + delta1

            if team1 in teams_on_page:
                team1_match = MatchElo(
                    team=team1,
                    opponent=team2,
                    team_score=score1,
                    opponent_score=score2,
                    team_elo_start=pre1,
                    opponent_elo_start=pre2,
                    team_elo_end=post1,
                    opponent_elo_end=post2,
                    team_elo_delta=delta1,
                )
                lookup[(team1, match_date, team2, score1, score2)] = team1_match
                fallback_lookup[(team1, match_date, score1, score2)] = team1_match
            if team2 in teams_on_page:
                team2_match = MatchElo(
                    team=team2,
                    opponent=team1,
                    team_score=score2,
                    opponent_score=score1,
                    team_elo_start=pre2,
                    opponent_elo_start=pre1,
                    team_elo_end=post2,
                    opponent_elo_end=post1,
                    team_elo_delta=-delta1,
                )
                lookup[(team2, match_date, team1, score2, score1)] = team2_match
                fallback_lookup[(team2, match_date, score2, score1)] = team2_match
        if index % 10 == 0:
            print(f"Fetched Elo team pages: {index}/{len(teams_by_page)}")
        time.sleep(0.05)
    return lookup, fallback_lookup


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
        flag_icon_code = FLAG_ICONS_CODE_BY_FIFA_CODE.get(team.fifa_code, "")
        rows.append(
            {
                "team_id": team.team_id,
                "canonical_name": team.canonical_name,
                "tournament_name": team.tournament_name,
                "fifa_code": team.fifa_code,
                "flag_icon_code": flag_icon_code,
                "flag_icon_css_class": f"fi fi-{flag_icon_code}" if flag_icon_code else "",
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
    match_elo_lookup: dict[tuple[str, str, str, int, int], MatchElo],
    match_elo_fallback_lookup: dict[tuple[str, str, int, int], MatchElo],
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
            elo_match = match_elo_lookup.get(
                (canonical_team_name, result["date"], canonical_opponent_name, team_score, opponent_score)
            )
            if elo_match is None:
                elo_match = match_elo_fallback_lookup.get(
                    (canonical_team_name, result["date"], team_score, opponent_score)
                )
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
                    "team_elo_start": elo_match.team_elo_start if elo_match else "",
                    "opponent_elo_start": elo_match.opponent_elo_start if elo_match else "",
                    "team_elo_delta": elo_match.team_elo_delta if elo_match else "",
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
    rated_counts = Counter(
        row["qualified_team_id"]
        for row in lead_in_rows
        if row["team_elo_start"] != "" and row["opponent_elo_start"] != "" and row["team_elo_delta"] != ""
    )
    if any(rated_counts.get(str(row["team_id"]), 0) < REQUIRED_RATED_LEAD_IN_MATCHES for row in teams_rows):
        raise ValueError(
            f"Each qualified team must have at least {REQUIRED_RATED_LEAD_IN_MATCHES} lead-in matches with Elo fields"
        )
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
    lead_in_match_elo_lookup, lead_in_match_elo_fallback_lookup = build_lead_in_match_elo_lookup(
        qualified_teams,
        elo_rankings,
        alias_map,
        dated_former_aliases,
    )
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
        lead_in_match_elo_lookup,
        lead_in_match_elo_fallback_lookup,
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
            ["team_id", "canonical_name", "tournament_name", "fifa_code", "flag_icon_code", "flag_icon_css_class", "confederation", "group_code", "is_host", "qualification_path", "world_cup_participations", "squad_status", "source_url", "source_as_of"],
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
            ["lead_in_id", "match_key", "date", "qualified_team_id", "qualified_team_name", "source_team_name", "opponent_team_id", "opponent_name", "source_opponent_name", "perspective", "is_home_perspective", "home_team", "away_team", "home_team_canonical", "away_team_canonical", "team_score", "opponent_score", "goal_difference", "team_elo_start", "opponent_elo_start", "team_elo_delta", "result", "tournament", "city", "country", "neutral", "days_before_tournament", "goalscorer_events_for_team", "goalscorer_events_for_opponent", "penalty_goal_events_for_team", "penalty_goal_events_for_opponent", "own_goal_events_for_team", "own_goal_events_for_opponent", "decided_by_shootout", "shootout_winner", "shootout_first_shooter"],
        ),
    }
    write_outputs(output_dir, datasets)
    build_manifest(output_dir, build_date)
    print(json.dumps({"output_dir": str(output_dir), "teams": len(teams_rows), "groups": len({row["group_code"] for row in groups_rows}), "venues": len(venues_rows), "fixtures": len(fixtures_rows), "fifa_rank_rows": len(fifa_rows), "elo_rows": len(elo_rows), "lead_in_rows": len(lead_in_rows)}, indent=2))


if __name__ == "__main__":
    main()
