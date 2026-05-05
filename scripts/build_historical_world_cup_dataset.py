from __future__ import annotations

import csv
import hashlib
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
INT_WORLD_CUP_DIR = ROOT / "INT-World Cup" / "world_cup"
WORLDCUP_CSV_DIR = ROOT / "worldcup" / "data-csv"
BY_CONFEDERATION_DIR = INT_WORLD_CUP_DIR / "by_confederation"
ALL_EDITIONS_DIR = INT_WORLD_CUP_DIR / "all_editions"
CACHE_DIR = ROOT / ".cache" / "historical_world_cup_dataset"

RESULTS_PATH = INT_WORLD_CUP_DIR / "results.csv"
FORMER_NAMES_PATH = INT_WORLD_CUP_DIR / "former_names.csv"
HISTORY_PATH = INT_WORLD_CUP_DIR / "fifa_world_cup_history.csv"
SHOOTOUTS_PATH = INT_WORLD_CUP_DIR / "shootouts.csv"

WORLDCUP_TEAMS_PATH = WORLDCUP_CSV_DIR / "teams.csv"
WORLDCUP_PLAYERS_PATH = WORLDCUP_CSV_DIR / "players.csv"
WORLDCUP_SQUADS_PATH = WORLDCUP_CSV_DIR / "squads.csv"
WORLDCUP_TOURNAMENTS_PATH = WORLDCUP_CSV_DIR / "tournaments.csv"

ELO_BASE_URL = "https://www.eloratings.net"
NTF_BASE_URL = "https://www.national-football-teams.com"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PLACEMENT_FIELDS = [
    "tournament_id",
    "year",
    "country",
    "team_id",
    "team_code",
    "confederation",
    "placement",
    "position",
    "matches_played",
    "gs",
    "ga",
    "start_elo",
    "finish_elo",
    "elo_change",
    "next_edition",
    "next_placement",
    "next_position",
]
SCHEDULE_FIELDS = [
    "tournament_id",
    "match_id",
    "match_number",
    "date",
    "stage",
    "status",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "home_team_code",
    "away_team_code",
    "home_team_confederation",
    "away_team_confederation",
    "home_elo_start",
    "away_elo_start",
    "city",
    "country",
    "neutral",
]
RESULT_FIELDS = [
    "tournament_id",
    "match_id",
    "match_number",
    "date",
    "stage",
    "status",
    "team_id",
    "team",
    "team_confederation",
    "opponent_id",
    "opponent",
    "opponent_confederation",
    "is_home",
    "team_score",
    "opponent_score",
    "result",
    "team_elo_start",
    "opponent_elo_start",
    "team_elo_end",
    "opponent_elo_end",
    "team_elo_delta",
    "city",
    "country",
    "neutral",
    "decided_by_shootout",
    "shootout_winner",
]
SQUAD_FIELDS = [
    "team",
    "team_id",
    "team_code",
    "confederation",
    "tournament",
    "tournament_id",
    "year",
    "player_name",
    "player_id",
    "position",
    "pos_code",
    "shirt_number",
    "date_of_birth",
    "age",
    "club",
    "club_country",
    "caps",
    "goals",
]
COUNTRY_RESULTS_FIELDS = [
    "tournament_id",
    "match_id",
    "match_number",
    "date",
    "tournament",
    "status",
    "team",
    "team_id",
    "team_code",
    "team_confederation",
    "opponent",
    "opponent_id",
    "opponent_code",
    "opponent_confederation",
    "is_home",
    "team_score",
    "opponent_score",
    "result",
    "city",
    "country",
    "neutral",
    "team_elo_start",
    "opponent_elo_start",
    "team_elo_end",
    "opponent_elo_end",
    "team_elo_delta",
]
COUNTRY_SCHEDULE_FIELDS = [
    "tournament",
    "tournament_id",
    "year",
    "match_id",
    "match_number",
    "date",
    "stage",
    "status",
    "team",
    "team_id",
    "team_code",
    "team_confederation",
    "opponent",
    "opponent_id",
    "opponent_code",
    "opponent_confederation",
    "is_home",
    "city",
    "country",
    "neutral",
    "team_elo_start",
    "opponent_elo_start",
]
TEAMS_FIELDS = [
    "tournament_id",
    "year",
    "team_id",
    "team",
    "team_code",
    "confederation",
    "tournament_name",
    "placement",
    "position",
    "matches_played",
]
ELO_FIELDS = [
    "tournament_id",
    "year",
    "team_id",
    "team",
    "confederation",
    "elo_start",
    "elo_end",
    "elo_change",
    "elo_status",
]

TOP_FOUR_POSITIONS = {
    "Winner": 1,
    "Runner-up": 2,
    "Third Place": 3,
    "Fourth Place": 4,
}
STAGE_POSITIONS = {
    "Semi-final": 4,
    "Quarter-final": 8,
    "Round of 16": 16,
}
YEAR_STAGE_PLANS = {
    1930: [("Group Stage", 15), ("Semi-final", 2), ("Final", 1)],
    1934: [("Round of 16", 8), ("Quarter-final", 5), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1938: [("Round of 16", 9), ("Quarter-final", 5), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1950: [("Group Stage", 16), ("Final", 6)],
    1954: [("Group Stage", 18), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1958: [("Group Stage", 27), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1962: [("Group Stage", 24), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1966: [("Group Stage", 24), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1970: [("Group Stage", 24), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1974: [("Group Stage", 24), ("Quarter-final", 12), ("Third Place", 1), ("Final", 1)],
    1978: [("Group Stage", 24), ("Quarter-final", 12), ("Third Place", 1), ("Final", 1)],
    1982: [("Group Stage", 36), ("Quarter-final", 12), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1986: [("Group Stage", 36), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1990: [("Group Stage", 36), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1994: [("Group Stage", 36), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    1998: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    2002: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    2006: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    2010: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    2014: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    2018: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
    2022: [("Group Stage", 48), ("Round of 16", 8), ("Quarter-final", 4), ("Semi-final", 2), ("Third Place", 1), ("Final", 1)],
}
FUTURE_SCHEDULE_DEFAULT_STAGE = {
    2026: "Group Stage",
}

PRESERVE_HISTORICAL = {
    "CIS",
    "Czechoslovakia",
    "German DR",
    "Serbia and Montenegro",
    "Soviet Union",
    "West Germany",
    "Yugoslavia",
    "Zaire",
}
DIRECT_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "CuraÃ§ao": "Curacao",
    "CuraÃƒÂ§ao": "Curacao",
    "Czech Republic": "Czechia",
    "CÃƒÂ´te d'Ivoire": "Ivory Coast",
    "CÃ´te d'Ivoire": "Ivory Coast",
    "Dem. Rep. of Congo": "DR Congo",
    "Dr Congo": "DR Congo",
    "East Germany": "German DR",
    "IR Iran": "Iran",
    "Korea": "South Korea",
    "Korea Republic": "South Korea",
    "TÃƒÂ¼rkiye": "Turkey",
    "TÃ¼rkiye": "Turkey",
    "USA": "United States",
    "ZaÃ¯re": "Zaire",
}
OUTPUT_TEAM_ALIASES = {
    "East Germany": "German DR",
}
NTF_PREFERRED_NAMES = {
    "German DR": "East Germany",
    "West Germany": "Germany",
    "Zaire": "Dr Congo",
    "United States": "Usa",
}
CONFEDERATION_FOLDER_NAMES = {
    "AFC": "afc",
    "CAF": "caf",
    "CONCACAF": "concacaf",
    "CONMEBOL": "conmebol",
    "OFC": "ofc",
    "UEFA": "uefa",
}
CANONICAL_TEAM_ID_OVERRIDES = {
    "Germany": "GER",
    "Netherlands": "NED",
    "Saudi Arabia": "KSA",
}
TEAM_REFERENCE_ALIASES = {
    "China": ("China PR",),
    "Zaire": ("DR Congo", "Congo DR"),
}
TEAM_REFERENCE_ALIAS_NAMES = {
    alias for aliases in TEAM_REFERENCE_ALIASES.values() for alias in aliases
}


@dataclass(frozen=True)
class TournamentInfo:
    tournament_id: str
    tournament_name: str
    year: int
    start_date: date
    end_date: date
    teams: int


@dataclass(frozen=True)
class TeamInfo:
    team_name: str
    team_code: str
    confederation_code: str


@dataclass(frozen=True)
class ReverseHistoricalMapping:
    current: str
    former: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class NtfCountry:
    country_id: str
    slug: str
    display_name: str


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


SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-zA-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def slugify_path_component(value: str) -> str:
    normalized = normalize_key(value)
    normalized = normalized.replace(" ", "_")
    return normalized or "unknown"


def canonical_team_id(team_name: str, team_reference: dict[str, TeamInfo]) -> str:
    if team_name in CANONICAL_TEAM_ID_OVERRIDES:
        return CANONICAL_TEAM_ID_OVERRIDES[team_name]
    team_meta = team_reference.get(team_name)
    return team_meta.team_code if team_meta else ""


def team_confederation(team_name: str, team_reference: dict[str, TeamInfo]) -> str:
    team_meta = team_reference.get(team_name)
    return team_meta.confederation_code if team_meta else ""


def canonical_match_id(tournament_id: str, match_number: object) -> str:
    return f"{tournament_id}_{int(match_number):03d}"


def page_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^A-Za-z0-9 ]+", "", normalized)
    return normalized.replace(" ", "_")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized: dict[str, str] = {}
            for field in fieldnames:
                value = row.get(field, "")
                if value is None:
                    serialized[field] = ""
                elif isinstance(value, bool):
                    serialized[field] = "TRUE" if value else "FALSE"
                else:
                    serialized[field] = str(value)
            writer.writerow(serialized)


def request_response(url: str, allow_404: bool = False) -> requests.Response:
    retries = 6
    for attempt in range(1, retries + 1):
        try:
            response = SESSION.get(url, timeout=60)
            if allow_404 and response.status_code == 404:
                return response
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(10, attempt * 0.75))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(min(10, attempt * 0.75))
    raise RuntimeError(f"Unreachable retry loop for {url}")


def cache_path_for_url(url: str, suffix: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}{suffix}"


def request_bytes(url: str) -> bytes:
    suffix = Path(url.split("?", 1)[0]).suffix or ".bin"
    cache_path = cache_path_for_url(url, suffix)
    if cache_path.exists():
        return cache_path.read_bytes()
    content = request_response(url).content
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content


def request_text(url: str) -> str:
    return request_bytes(url).decode("utf-8", errors="replace")


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def sanitize_birth_date(value: str) -> str:
    stripped = value.strip()
    if not stripped or stripped == "not available":
        return ""
    return stripped


def format_player_name(family_name: str, given_name: str) -> str:
    if given_name == "not applicable" or not given_name:
        return family_name
    return f"{given_name} {family_name}"


def compute_age(birth_date: str, tournament_start: date) -> str:
    if not birth_date:
        return ""
    born = parse_iso_date(birth_date)
    years = tournament_start.year - born.year
    if (tournament_start.month, tournament_start.day) < (born.month, born.day):
        years -= 1
    return str(years)


def parse_int(value: str) -> int:
    stripped = value.strip()
    if stripped in {"", "-"}:
        return 0
    return int(stripped)


def decode_elo_delta(value: str) -> int:
    stripped = (
        value.strip()
        .replace("−", "-")
        .replace("â", "-")
        .replace("ā\x88\x92", "-")
        .replace("&minus;", "-")
    )
    if stripped in {"", "-", "−"}:
        return 0
    return int(stripped)


def assign_stages(year: int, match_count: int) -> list[str]:
    plan = YEAR_STAGE_PLANS[year]
    stages: list[str] = []
    for stage, count in plan:
        stages.extend([stage] * count)
    if len(stages) != match_count:
        raise ValueError(f"{year}: stage plan covers {len(stages)} matches, expected {match_count}")
    return stages


def infer_schedule_stages(year: int, raw_matches: list[dict[str, object]]) -> list[str]:
    if year in YEAR_STAGE_PLANS:
        expected_count = sum(count for _, count in YEAR_STAGE_PLANS[year])
        if len(raw_matches) == expected_count:
            return assign_stages(year, len(raw_matches))
    default_stage = FUTURE_SCHEDULE_DEFAULT_STAGE.get(year, "")
    return [default_stage] * len(raw_matches)


def build_former_name_map() -> dict[str, str]:
    mapping = {normalize_key(source): target for source, target in DIRECT_ALIASES.items()}
    for row in load_csv(FORMER_NAMES_PATH):
        former = row["former"].strip()
        current = row["current"].strip()
        start_date = row["start_date"].strip()
        end_date = row["end_date"].strip()
        if not former or not current:
            continue
        if not start_date or not end_date:
            continue
        if former in PRESERVE_HISTORICAL:
            continue
        mapping.setdefault(normalize_key(former), current)
    return mapping


def build_reverse_historical_mappings() -> dict[str, list[ReverseHistoricalMapping]]:
    reverse: dict[str, list[ReverseHistoricalMapping]] = defaultdict(list)
    for row in load_csv(FORMER_NAMES_PATH):
        former = row["former"].strip()
        current = row["current"].strip()
        start_date = row["start_date"].strip()
        end_date = row["end_date"].strip()
        if former not in PRESERVE_HISTORICAL:
            continue
        if not start_date or not end_date:
            continue
        reverse[normalize_key(current)].append(
            ReverseHistoricalMapping(
                current=current,
                former=former,
                start_date=parse_iso_date(start_date),
                end_date=parse_iso_date(end_date),
            )
        )
    return reverse


def canonicalize_name(
    name: str,
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
    match_date: str | None = None,
) -> str:
    stripped = OUTPUT_TEAM_ALIASES.get(name.strip(), name.strip())
    if stripped in PRESERVE_HISTORICAL:
        return stripped
    key = normalize_key(stripped)
    if match_date:
        current_match_date = parse_iso_date(match_date)
        for mapping in reverse_historical_mappings.get(key, []):
            if mapping.start_date <= current_match_date <= mapping.end_date:
                return mapping.former
    canonical = former_name_map.get(key, stripped)
    return OUTPUT_TEAM_ALIASES.get(canonical, canonical)


def load_tournaments() -> dict[int, TournamentInfo]:
    tournaments: dict[int, TournamentInfo] = {}
    for row in load_csv(WORLDCUP_TOURNAMENTS_PATH):
        if row["tournament_id"].startswith("WC-") and "Men" in row["tournament_name"]:
            year = int(row["year"])
            tournaments[year] = TournamentInfo(
                tournament_id=row["tournament_id"],
                tournament_name=row["tournament_name"],
                year=year,
                start_date=parse_iso_date(row["start_date"]),
                end_date=parse_iso_date(row["end_date"]),
                teams=int(row["count_teams"]),
            )
    return tournaments


def load_team_reference(
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[str, TeamInfo]:
    team_reference: dict[str, TeamInfo] = {}
    for row in load_csv(WORLDCUP_TEAMS_PATH):
        if row["mens_team"] != "1":
            continue
        team_name = canonicalize_name(row["team_name"], former_name_map, reverse_historical_mappings)
        team_reference[team_name] = TeamInfo(
            team_name=team_name,
            team_code=row["team_code"],
            confederation_code=row["confederation_code"],
        )
        for alias in TEAM_REFERENCE_ALIASES.get(team_name, ()):
            team_reference[alias] = TeamInfo(
                team_name=alias,
                team_code=row["team_code"],
                confederation_code=row["confederation_code"],
            )
    return team_reference


def load_world_cup_matches(
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in load_csv(RESULTS_PATH):
        if row["tournament"] != "FIFA World Cup":
            continue
        is_unplayed = row["home_score"] in {"", "NA"} or row["away_score"] in {"", "NA"}
        grouped[int(row["date"][:4])].append(
            {
                "date": row["date"],
                "home_team": canonicalize_name(
                    row["home_team"], former_name_map, reverse_historical_mappings, row["date"]
                ),
                "away_team": canonicalize_name(
                    row["away_team"], former_name_map, reverse_historical_mappings, row["date"]
                ),
                "home_score": "" if is_unplayed else int(row["home_score"]),
                "away_score": "" if is_unplayed else int(row["away_score"]),
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"],
                "status": "scheduled" if is_unplayed else "played",
            }
        )
    return dict(sorted(grouped.items()))


def build_shootout_lookup(
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for row in load_csv(SHOOTOUTS_PATH):
        key = (
            row["date"],
            canonicalize_name(row["home_team"], former_name_map, reverse_historical_mappings, row["date"]),
            canonicalize_name(row["away_team"], former_name_map, reverse_historical_mappings, row["date"]),
        )
        lookup[key] = canonicalize_name(
            row["winner"], former_name_map, reverse_historical_mappings, row["date"]
        )
    return lookup


def load_history(
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[int, dict[str, object]]:
    history: dict[int, dict[str, object]] = {}
    for row in load_csv(HISTORY_PATH):
        year = int(row["Year"])
        history[year] = {
            "year": year,
            "winner": canonicalize_name(row["Winner"], former_name_map, reverse_historical_mappings),
            "runner_up": canonicalize_name(row["Runner_Up"], former_name_map, reverse_historical_mappings),
            "third_place": canonicalize_name(row["Third_Place"], former_name_map, reverse_historical_mappings),
            "fourth_place": canonicalize_name(row["Fourth_Place"], former_name_map, reverse_historical_mappings),
            "total_goals": int(row["Total_Goals"]),
            "matches_played": int(row["Matches_Played"]),
            "teams": int(row["Teams"]),
        }
    return history


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


def map_team_to_elo_code(team_name: str, code_by_name: dict[str, str]) -> str:
    aliases = [team_name]
    if team_name == "German DR":
        aliases.extend(["East Germany", "German DR"])
    elif team_name == "China PR":
        aliases.extend(["China"])
    elif team_name == "DR Congo":
        aliases.extend(["Dr Congo", "Dem. Rep. of Congo"])
    elif team_name == "Republic of Ireland":
        aliases.extend(["Ireland"])
    elif team_name == "South Korea":
        aliases.extend(["South Korea", "Korea"])
    elif team_name == "Turkey":
        aliases.extend(["Türkiye"])
    elif team_name == "Türkiye":
        aliases.extend(["Turkey", "Türkiye"])
    elif team_name == "United States":
        aliases.extend(["United States", "USA"])
    elif team_name == "Curacao":
        aliases.extend(["Curacao", "Curaçao"])
    elif team_name == "Ivory Coast":
        aliases.extend(["Ivory Coast", "Côte d'Ivoire"])
    elif team_name == "Bosnia and Herzegovina":
        aliases.extend(["Bosnia-Herzegovina"])
    elif team_name == "Turkey":
        aliases.extend(["Türkiye"])
    elif team_name == "Iran":
        aliases.extend(["IR Iran"])
    elif team_name == "Czechia":
        aliases.extend(["Czech Republic"])

    for alias in aliases:
        code = code_by_name.get(normalize_key(alias))
        if code:
            return code
    raise KeyError(f"No Elo code found for {team_name}")


def parse_elo_rating_table(content: str) -> dict[str, int]:
    ratings: dict[str, int] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        ratings[fields[2]] = int(fields[3])
    return ratings


def parse_elo_match_rows(
    content: str,
    names_by_code: dict[str, list[str]],
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[tuple[str, str, str, int, int], MatchElo]:
    lookup: dict[tuple[str, str, str, int, int], MatchElo] = {}
    for line in content.splitlines():
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
        team1_code = fields[3]
        team2_code = fields[4]
        team1 = canonicalize_name(
            names_by_code[team1_code][0], former_name_map, reverse_historical_mappings, match_date
        )
        team2 = canonicalize_name(
            names_by_code[team2_code][0], former_name_map, reverse_historical_mappings, match_date
        )
        score1 = int(fields[5])
        score2 = int(fields[6])
        delta1 = decode_elo_delta(fields[9])
        post1 = int(fields[10])
        post2 = int(fields[11])
        pre1 = post1 - delta1
        pre2 = post2 + delta1
        lookup[(match_date, team1, team2, score1, score2)] = MatchElo(
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
        lookup[(match_date, team2, team1, score2, score1)] = MatchElo(
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
    return lookup


def top_four_placements(summary: dict[str, object]) -> dict[str, tuple[str, int]]:
    return {
        str(summary["winner"]): ("Winner", TOP_FOUR_POSITIONS["Winner"]),
        str(summary["runner_up"]): ("Runner-up", TOP_FOUR_POSITIONS["Runner-up"]),
        str(summary["third_place"]): ("Third Place", TOP_FOUR_POSITIONS["Third Place"]),
        str(summary["fourth_place"]): ("Fourth Place", TOP_FOUR_POSITIONS["Fourth Place"]),
    }


def build_placement_rows(
    year: int,
    tournament_id: str,
    stats: dict[str, dict[str, object]],
    summary: dict[str, object],
    team_reference: dict[str, TeamInfo],
) -> list[dict[str, object]]:
    placements = top_four_placements(summary)
    rows: list[dict[str, object]] = []
    for country in sorted(stats):
        team_stats = stats[country]
        stage_set = set(team_stats["stages"])
        placement, position = placements.get(country, ("", 0))
        if not placement:
            if "Semi-final" in stage_set:
                placement = "Semi-final"
                position = STAGE_POSITIONS[placement]
            elif "Quarter-final" in stage_set:
                placement = "Quarter-final"
                position = STAGE_POSITIONS[placement]
            elif "Round of 16" in stage_set:
                placement = "Round of 16"
                position = STAGE_POSITIONS[placement]
            else:
                placement = "Group Stage"
                position = int(summary["teams"])
        team_meta = team_reference.get(country)
        rows.append(
            {
                "tournament_id": tournament_id,
                "year": year,
                "country": country,
                "team_id": canonical_team_id(country, team_reference),
                "team_code": team_meta.team_code if team_meta else "",
                "confederation": team_meta.confederation_code if team_meta else "",
                "placement": placement,
                "position": position,
                "matches_played": team_stats["matches_played"],
                "gs": team_stats["gs"],
                "ga": team_stats["ga"],
                "start_elo": team_stats.get("start_elo", ""),
                "finish_elo": team_stats.get("finish_elo", ""),
                "elo_change": team_stats.get("elo_change", ""),
                "next_edition": "",
                "next_placement": "",
                "next_position": "",
            }
        )
    rows.sort(key=lambda row: (int(row["position"]), str(row["country"])))
    return rows


def annotate_next_edition_placements(placement_rows_by_year: dict[int, list[dict[str, object]]]) -> None:
    by_country: dict[str, list[dict[str, object]]] = defaultdict(list)
    for year in sorted(placement_rows_by_year):
        for row in placement_rows_by_year[year]:
            by_country[str(row["country"])].append(row)
    for country_rows in by_country.values():
        for index, row in enumerate(country_rows[:-1]):
            next_row = country_rows[index + 1]
            row["next_edition"] = next_row["edition"]
            row["next_placement"] = next_row["placement"]
            row["next_position"] = next_row["position"]


def validate_year(
    year: int,
    schedule_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
    placement_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    if len(result_rows) != int(summary["matches_played"]) * 2:
        raise ValueError(f"{year}: results row count mismatch")
    if len(schedule_rows) != int(summary["matches_played"]):
        raise ValueError(f"{year}: schedule row count mismatch")
    if len(placement_rows) != int(summary["teams"]):
        raise ValueError(f"{year}: placement row count mismatch")
    stage_pairs = [(row["match_number"], row["stage"]) for row in schedule_rows]
    result_stage_pairs = sorted({(row["match_number"], row["stage"]) for row in result_rows})
    if stage_pairs != result_stage_pairs:
        raise ValueError(f"{year}: schedule and results stages differ")
    total_matches_played = sum(int(row["matches_played"]) for row in placement_rows)
    if total_matches_played != int(summary["matches_played"]) * 2:
        raise ValueError(f"{year}: matches_played total mismatch")
    total_gs = sum(int(row["gs"]) for row in placement_rows)
    total_ga = sum(int(row["ga"]) for row in placement_rows)
    if total_gs != int(summary["total_goals"]) or total_ga != int(summary["total_goals"]):
        raise ValueError(f"{year}: goal totals mismatch")


def build_ntf_country_index() -> dict[str, NtfCountry]:
    xml_text = request_text(f"{NTF_BASE_URL}/sitemap_country.xml")
    root = ET.fromstring(xml_text)
    countries: dict[str, NtfCountry] = {}
    for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        url = (loc.text or "").strip()
        match = re.search(r"/country/(\d+)/([^/]+)\.html$", url)
        if not match:
            continue
        country_id, slug = match.groups()
        display = re.sub(r"_\d+$", "", slug).replace("_", " ")
        country = NtfCountry(country_id=country_id, slug=slug, display_name=display)
        for key in {normalize_key(display), normalize_key(slug)}:
            countries[key] = country
    return countries


def resolve_ntf_country(team_name: str, ntf_countries: dict[str, NtfCountry]) -> NtfCountry | None:
    candidates = [team_name]
    preferred = NTF_PREFERRED_NAMES.get(team_name)
    if preferred:
        candidates.insert(0, preferred)
    if team_name == "DR Congo":
        candidates.append("Dr Congo")
    for candidate in candidates:
        found = ntf_countries.get(normalize_key(candidate))
        if found:
            return found
    return None


def fetch_ntf_year_page(country: NtfCountry, year: int) -> str | None:
    year_url = f"{NTF_BASE_URL}/country/{country.country_id}/{year}/{country.slug}.html"
    cache_path = CACHE_DIR / "ntf" / f"{country.country_id}_{year}_{country.slug}.html"
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8")
        return None if cached == "__404__" else cached
    response = request_response(year_url, allow_404=True)
    if response.status_code == 404:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("__404__", encoding="utf-8")
        return None
    html = response.content.decode("utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


def parse_ntf_player_table(html: str) -> dict[tuple[str, str], dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="player")
    if table is None:
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    body = table.find("tbody")
    if body is None:
        return rows
    for tr in body.find_all("tr"):
        name_cell = tr.find("td", class_="name")
        dob_cell = tr.find("td", class_="dob")
        position_cell = tr.find("td", class_="position")
        club_cell = tr.find("td", class_="club")
        stat_cells = tr.find_all("td", class_=re.compile(r"stats"))
        if name_cell is None or dob_cell is None or position_cell is None or club_cell is None or len(stat_cells) < 6:
            continue
        family = name_cell.find("span", itemprop="familyName")
        given = name_cell.find("span", itemprop="givenName")
        if family and given:
            player_name = format_player_name(family.get_text(strip=True), given.get_text(strip=True))
        else:
            player_name = name_cell.get_text(" ", strip=True).replace(",", "").strip()
        birth_date = sanitize_birth_date(dob_cell.get_text(strip=True))
        club_country = ""
        club_flag = tr.find("td", class_="flag")
        if club_flag:
            image = club_flag.find("img")
            if image:
                club_country = image.get("alt", "").strip()
        fifa_matches = parse_int(stat_cells[0].get_text(strip=True))
        non_fifa_matches = parse_int(stat_cells[3].get_text(strip=True))
        fifa_goals = parse_int(stat_cells[2].get_text(strip=True))
        non_fifa_goals = parse_int(stat_cells[5].get_text(strip=True))
        rows[(normalize_key(player_name), birth_date)] = {
            "club": club_cell.get_text(" ", strip=True),
            "club_country": club_country,
            "caps": str(fifa_matches + non_fifa_matches),
            "goals": str(fifa_goals + non_fifa_goals),
            "position": position_cell.get_text(" ", strip=True),
        }
    return rows


def build_ntf_enrichment(
    team_years: set[tuple[str, int]],
    ntf_countries: dict[str, NtfCountry],
) -> dict[tuple[str, int], dict[tuple[str, str], dict[str, str]]]:
    enrichment: dict[tuple[str, int], dict[tuple[str, str], dict[str, str]]] = {}
    ordered_team_years = sorted(team_years)
    for index, (team_name, year) in enumerate(ordered_team_years, start=1):
        country = resolve_ntf_country(team_name, ntf_countries)
        if country is None:
            enrichment[(team_name, year)] = {}
            continue
        html = fetch_ntf_year_page(country, year)
        enrichment[(team_name, year)] = parse_ntf_player_table(html) if html else {}
        if index % 25 == 0:
            print(f"Fetched National Football Teams pages: {index}/{len(ordered_team_years)}")
        time.sleep(0.05)
    return enrichment


def build_squad_rows_by_year(
    tournaments: dict[int, TournamentInfo],
    team_reference: dict[str, TeamInfo],
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
    ntf_enrichment: dict[tuple[str, int], dict[tuple[str, str], dict[str, str]]],
) -> dict[int, list[dict[str, object]]]:
    players_by_id = {row["player_id"]: row for row in load_csv(WORLDCUP_PLAYERS_PATH)}
    squads_by_year: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in load_csv(WORLDCUP_SQUADS_PATH):
        tournament_id = row["tournament_id"]
        if not tournament_id.startswith("WC-"):
            continue
        year = int(tournament_id.split("-")[1])
        tournament = tournaments.get(year)
        if tournament is None:
            continue
        team_name = canonicalize_name(
            row["team_name"], former_name_map, reverse_historical_mappings, tournament.start_date.isoformat()
        )
        player = players_by_id[row["player_id"]]
        player_name = format_player_name(row["family_name"], row["given_name"])
        birth_date = sanitize_birth_date(player["birth_date"])
        enrich_key = (normalize_key(player_name), birth_date)
        enrich = ntf_enrichment.get((team_name, year), {}).get(enrich_key, {})
        shirt_number = row["shirt_number"] if row["shirt_number"] != "0" else ""
        squads_by_year[year].append(
            {
                "team": team_name,
                "team_id": canonical_team_id(team_name, team_reference),
                "team_code": row["team_code"],
                "confederation": team_confederation(team_name, team_reference),
                "tournament": tournament.tournament_name,
                "tournament_id": tournament.tournament_id,
                "year": year,
                "player_name": player_name,
                "player_id": row["player_id"],
                "position": row["position_name"],
                "pos_code": row["position_code"],
                "shirt_number": shirt_number,
                "date_of_birth": birth_date,
                "age": compute_age(birth_date, tournament.start_date),
                "club": enrich.get("club", ""),
                "club_country": enrich.get("club_country", ""),
                "caps": enrich.get("caps", ""),
                "goals": enrich.get("goals", ""),
            }
        )
    return {year: sorted(rows, key=lambda item: (str(item["team"]), str(item["player_name"]))) for year, rows in squads_by_year.items()}


def build_country_match_elo_lookup(
    team_to_elo_code: dict[str, str],
    successor_map: dict[str, str],
    names_by_code: dict[str, list[str]],
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[tuple[str, str, str, int, int], MatchElo]:
    teams_by_page: dict[str, set[str]] = defaultdict(set)
    for team_name, elo_code in team_to_elo_code.items():
        page_code = successor_map.get(elo_code, elo_code)
        page_slug = page_name(names_by_code[page_code][0])
        teams_by_page[page_slug].add(team_name)

    lookup: dict[tuple[str, str, str, int, int], MatchElo] = {}
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
            team1 = canonicalize_name(
                names_by_code[code1][0], former_name_map, reverse_historical_mappings, match_date
            )
            team2 = canonicalize_name(
                names_by_code[code2][0], former_name_map, reverse_historical_mappings, match_date
            )
            score1 = int(fields[5])
            score2 = int(fields[6])
            delta1 = decode_elo_delta(fields[9])
            post1 = int(fields[10])
            post2 = int(fields[11])
            pre1 = post1 - delta1
            pre2 = post2 + delta1

            if team1 in teams_on_page:
                lookup[(team1, match_date, team2, score1, score2)] = MatchElo(
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
            if team2 in teams_on_page:
                lookup[(team2, match_date, team1, score2, score1)] = MatchElo(
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
        if index % 15 == 0:
            print(f"Fetched Elo team pages: {index}/{len(teams_by_page)}")
        time.sleep(0.05)
    return lookup


def build_latest_elo_by_team(
    team_to_elo_code: dict[str, str],
    successor_map: dict[str, str],
    names_by_code: dict[str, list[str]],
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> dict[str, int]:
    teams_by_page: dict[str, set[str]] = defaultdict(set)
    for team_name, elo_code in team_to_elo_code.items():
        page_code = successor_map.get(elo_code, elo_code)
        page_slug = page_name(names_by_code[page_code][0])
        teams_by_page[page_slug].add(team_name)

    latest: dict[str, tuple[str, int]] = {}
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
            team1 = canonicalize_name(
                names_by_code[code1][0], former_name_map, reverse_historical_mappings, match_date
            )
            team2 = canonicalize_name(
                names_by_code[code2][0], former_name_map, reverse_historical_mappings, match_date
            )
            post1 = int(fields[10])
            post2 = int(fields[11])

            if team1 in teams_on_page:
                previous = latest.get(team1)
                if previous is None or match_date > previous[0]:
                    latest[team1] = (match_date, post1)
            if team2 in teams_on_page:
                previous = latest.get(team2)
                if previous is None or match_date > previous[0]:
                    latest[team2] = (match_date, post2)
        if index % 15 == 0:
            print(f"Fetched Elo team pages for latest ratings: {index}/{len(teams_by_page)}")
        time.sleep(0.05)
    return {team_name: rating for team_name, (_, rating) in latest.items()}


def build_outputs_for_year(
    year: int,
    tournament_id: str,
    raw_matches: list[dict[str, object]],
    summary: dict[str, object],
    team_reference: dict[str, TeamInfo],
    shootout_lookup: dict[tuple[str, str, str], str],
    elo_lookup: dict[tuple[str, str, str, int, int], MatchElo],
    start_elo_by_team: dict[str, int],
    end_elo_by_team: dict[str, int],
    latest_elo_by_team: dict[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    stages = infer_schedule_stages(year, raw_matches)
    schedule_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"matches_played": 0, "gs": 0, "ga": 0, "stages": set(), "start_elo": "", "finish_elo": "", "elo_change": ""}
    )

    for match_number, (row, stage) in enumerate(zip(raw_matches, stages, strict=True), start=1):
        match_id = canonical_match_id(tournament_id, match_number)
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        status = str(row.get("status", "played"))
        home_score_value = row["home_score"]
        away_score_value = row["away_score"]
        shootout_key = (str(row["date"]), home_team, away_team)
        shootout_winner = shootout_lookup.get(shootout_key, "")
        decided_by_shootout = bool(shootout_winner)
        home_team_code = team_reference.get(home_team).team_code if home_team in team_reference else ""
        away_team_code = team_reference.get(away_team).team_code if away_team in team_reference else ""
        home_team_id = canonical_team_id(home_team, team_reference)
        away_team_id = canonical_team_id(away_team, team_reference)
        home_confederation = team_confederation(home_team, team_reference)
        away_confederation = team_confederation(away_team, team_reference)
        elo_match = None
        if status == "played":
            home_score = int(home_score_value)
            away_score = int(away_score_value)
            elo_match = elo_lookup.get((str(row["date"]), home_team, away_team, home_score, away_score))
        home_schedule_elo = elo_match.team_elo_start if elo_match else latest_elo_by_team.get(home_team, "")
        away_schedule_elo = elo_match.opponent_elo_start if elo_match else latest_elo_by_team.get(away_team, "")

        schedule_rows.append(
            {
                "tournament_id": tournament_id,
                "match_id": match_id,
                "match_number": match_number,
                "date": row["date"],
                "stage": stage,
                "status": status,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_code": home_team_code,
                "away_team_code": away_team_code,
                "home_team_confederation": home_confederation,
                "away_team_confederation": away_confederation,
                "home_elo_start": home_schedule_elo,
                "away_elo_start": away_schedule_elo,
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"],
            }
        )
        if status != "played":
            continue
        perspectives = [
            (
                home_team,
                home_team_id,
                home_confederation,
                away_team,
                away_team_id,
                away_confederation,
                True,
                home_score,
                away_score,
                elo_match.team_elo_start if elo_match else "",
                elo_match.opponent_elo_start if elo_match else "",
                elo_match.team_elo_end if elo_match else "",
                elo_match.opponent_elo_end if elo_match else "",
                elo_match.team_elo_delta if elo_match else "",
            ),
            (
                away_team,
                away_team_id,
                away_confederation,
                home_team,
                home_team_id,
                home_confederation,
                False,
                away_score,
                home_score,
                elo_match.opponent_elo_start if elo_match else "",
                elo_match.team_elo_start if elo_match else "",
                elo_match.opponent_elo_end if elo_match else "",
                elo_match.team_elo_end if elo_match else "",
                (-elo_match.team_elo_delta if elo_match else ""),
            ),
        ]
        for (
            team_name,
            team_id,
            confederation,
            opponent_name,
            opponent_id,
            opponent_confederation,
            is_home,
            team_score,
            opponent_score,
            team_elo_start,
            opponent_elo_start,
            team_elo_end,
            opponent_elo_end,
            team_elo_delta,
        ) in perspectives:
            result_label = "draw"
            if team_score > opponent_score:
                result_label = "win"
            elif team_score < opponent_score:
                result_label = "loss"
            result_rows.append(
                {
                    "tournament_id": tournament_id,
                    "match_id": match_id,
                    "match_number": match_number,
                    "date": row["date"],
                    "stage": stage,
                    "status": status,
                    "team_id": team_id,
                    "team": team_name,
                    "team_confederation": confederation,
                    "opponent_id": opponent_id,
                    "opponent": opponent_name,
                    "opponent_confederation": opponent_confederation,
                    "is_home": is_home,
                    "team_score": team_score,
                    "opponent_score": opponent_score,
                    "result": result_label,
                    "team_elo_start": team_elo_start,
                    "opponent_elo_start": opponent_elo_start,
                    "team_elo_end": team_elo_end,
                    "opponent_elo_end": opponent_elo_end,
                    "team_elo_delta": team_elo_delta,
                    "city": row["city"],
                    "country": row["country"],
                    "neutral": row["neutral"],
                    "decided_by_shootout": decided_by_shootout,
                    "shootout_winner": shootout_winner,
                }
            )

        for team, gf, ga, start_elo, finish_elo in (
            (home_team, home_score, away_score, elo_match.team_elo_start if elo_match else "", elo_match.team_elo_end if elo_match else ""),
            (away_team, away_score, home_score, elo_match.opponent_elo_start if elo_match else "", elo_match.opponent_elo_end if elo_match else ""),
        ):
            team_stats = stats[team]
            team_stats["matches_played"] = int(team_stats["matches_played"]) + 1
            team_stats["gs"] = int(team_stats["gs"]) + gf
            team_stats["ga"] = int(team_stats["ga"]) + ga
            team_stats["stages"].add(stage)
            if start_elo != "" and team_stats["start_elo"] == "":
                team_stats["start_elo"] = start_elo
            if finish_elo != "":
                team_stats["finish_elo"] = finish_elo

    for team_name, team_stats in stats.items():
        if team_stats["start_elo"] == "":
            team_stats["start_elo"] = start_elo_by_team.get(team_name, "")
        if team_stats["finish_elo"] == "":
            team_stats["finish_elo"] = end_elo_by_team.get(team_name, "")
        if team_stats["start_elo"] != "" and team_stats["finish_elo"] != "":
            team_stats["elo_change"] = int(team_stats["finish_elo"]) - int(team_stats["start_elo"])

    placement_rows = build_placement_rows(year, tournament_id, stats, summary, team_reference)
    return schedule_rows, result_rows, placement_rows


def tournament_identity(year: int, tournaments: dict[int, TournamentInfo]) -> tuple[str, str]:
    tournament = tournaments.get(year)
    if tournament is not None:
        return tournament.tournament_name, tournament.tournament_id
    return f"{year} FIFA Men's World Cup", f"WC-{year}"


def build_teams_rows(
    year: int,
    tournament_id: str,
    tournament_name: str,
    placement_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "tournament_id": tournament_id,
            "year": year,
            "team_id": row["team_id"],
            "team": row["country"],
            "team_code": row["team_code"],
            "confederation": row["confederation"],
            "tournament_name": tournament_name,
            "placement": row["placement"],
            "position": row["position"],
            "matches_played": row["matches_played"],
        }
        for row in placement_rows
    ]


def build_elo_rows_from_placement(placement_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in placement_rows:
        rows.append(
            {
                "tournament_id": row["tournament_id"],
                "year": row["year"],
                "team_id": row["team_id"],
                "team": row["country"],
                "confederation": row["confederation"],
                "elo_start": row["start_elo"],
                "elo_end": row["finish_elo"],
                "elo_change": row["elo_change"],
                "elo_status": "final" if row["start_elo"] != "" or row["finish_elo"] != "" else "missing",
            }
        )
    return rows


def append_country_schedule_rows(
    country_schedule_rows: dict[str, list[dict[str, object]]],
    schedule_rows: list[dict[str, object]],
    year: int,
    tournament_name: str,
    tournament_id: str,
) -> None:
    for row in schedule_rows:
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        country_schedule_rows[home_team].append(
            {
                "tournament": tournament_name,
                "tournament_id": tournament_id,
                "year": year,
                "match_id": row["match_id"],
                "match_number": row["match_number"],
                "date": row["date"],
                "stage": row["stage"],
                "status": row["status"],
                "team": home_team,
                "team_id": row["home_team_id"],
                "team_code": row["home_team_code"],
                "team_confederation": row["home_team_confederation"],
                "opponent": away_team,
                "opponent_id": row["away_team_id"],
                "opponent_code": row["away_team_code"],
                "opponent_confederation": row["away_team_confederation"],
                "is_home": True,
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"] == "TRUE",
                "team_elo_start": row["home_elo_start"],
                "opponent_elo_start": row["away_elo_start"],
            }
        )
        country_schedule_rows[away_team].append(
            {
                "tournament": tournament_name,
                "tournament_id": tournament_id,
                "year": year,
                "match_id": row["match_id"],
                "match_number": row["match_number"],
                "date": row["date"],
                "stage": row["stage"],
                "status": row["status"],
                "team": away_team,
                "team_id": row["away_team_id"],
                "team_code": row["away_team_code"],
                "team_confederation": row["away_team_confederation"],
                "opponent": home_team,
                "opponent_id": row["home_team_id"],
                "opponent_code": row["home_team_code"],
                "opponent_confederation": row["home_team_confederation"],
                "is_home": False,
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"] == "TRUE",
                "team_elo_start": row["away_elo_start"],
                "opponent_elo_start": row["home_elo_start"],
            }
        )


def build_schedule_only_rows_for_year(
    year: int,
    tournament_id: str,
    raw_matches: list[dict[str, object]],
    team_reference: dict[str, TeamInfo],
    latest_elo_by_team: dict[str, int],
) -> list[dict[str, object]]:
    stages = infer_schedule_stages(year, raw_matches)
    schedule_rows: list[dict[str, object]] = []
    for match_number, (row, stage) in enumerate(zip(raw_matches, stages, strict=True), start=1):
        match_id = canonical_match_id(tournament_id, match_number)
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_team_code = team_reference.get(home_team).team_code if home_team in team_reference else ""
        away_team_code = team_reference.get(away_team).team_code if away_team in team_reference else ""
        schedule_rows.append(
            {
                "tournament_id": tournament_id,
                "match_id": match_id,
                "match_number": match_number,
                "date": row["date"],
                "stage": stage,
                "status": str(row.get("status", "scheduled")),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": canonical_team_id(home_team, team_reference),
                "away_team_id": canonical_team_id(away_team, team_reference),
                "home_team_code": home_team_code,
                "away_team_code": away_team_code,
                "home_team_confederation": team_confederation(home_team, team_reference),
                "away_team_confederation": team_confederation(away_team, team_reference),
                "home_elo_start": latest_elo_by_team.get(home_team, ""),
                "away_elo_start": latest_elo_by_team.get(away_team, ""),
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"],
            }
        )
    return schedule_rows


def build_country_exports(
    team_reference: dict[str, TeamInfo],
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
    country_match_elo: dict[tuple[str, str, str, int, int], MatchElo],
    country_schedule_rows: dict[str, list[dict[str, object]]],
    country_squad_rows: dict[str, list[dict[str, object]]],
) -> None:
    rows_by_team: dict[str, list[dict[str, object]]] = defaultdict(list)
    world_cup_match_numbers: dict[tuple[str, str, str], int] = {}
    for year, raw_matches in load_world_cup_matches(former_name_map, reverse_historical_mappings).items():
        for match_number, match in enumerate(raw_matches, start=1):
            world_cup_match_numbers[(str(match["date"]), str(match["home_team"]), str(match["away_team"]))] = match_number

    for row in load_csv(RESULTS_PATH):
        match_date = row["date"]
        if row["home_score"] in {"", "NA"} or row["away_score"] in {"", "NA"}:
            continue
        home_team = canonicalize_name(row["home_team"], former_name_map, reverse_historical_mappings, match_date)
        away_team = canonicalize_name(row["away_team"], former_name_map, reverse_historical_mappings, match_date)
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        is_world_cup = row["tournament"] == "FIFA World Cup"
        tournament_id = f"WC-{match_date[:4]}" if is_world_cup else ""
        match_number = world_cup_match_numbers.get((match_date, home_team, away_team), "")
        match_id = canonical_match_id(tournament_id, match_number) if tournament_id and match_number != "" else ""
        perspectives = [
            (home_team, away_team, home_score, away_score, True),
            (away_team, home_team, away_score, home_score, False),
        ]
        for team_name, opponent_name, team_score, opponent_score, is_home in perspectives:
            if team_name not in team_reference:
                continue
            elo = country_match_elo.get((team_name, match_date, opponent_name, team_score, opponent_score))
            result_label = "draw"
            if team_score > opponent_score:
                result_label = "win"
            elif team_score < opponent_score:
                result_label = "loss"
            rows_by_team[team_name].append(
                {
                    "tournament_id": tournament_id,
                    "match_id": match_id,
                    "match_number": match_number,
                    "date": match_date,
                    "tournament": row["tournament"],
                    "status": "played",
                    "team": team_name,
                    "team_id": canonical_team_id(team_name, team_reference),
                    "team_code": team_reference[team_name].team_code,
                    "team_confederation": team_confederation(team_name, team_reference),
                    "opponent": opponent_name,
                    "opponent_id": canonical_team_id(opponent_name, team_reference),
                    "opponent_code": team_reference.get(opponent_name).team_code if opponent_name in team_reference else "",
                    "opponent_confederation": team_confederation(opponent_name, team_reference),
                    "is_home": is_home,
                    "team_score": team_score,
                    "opponent_score": opponent_score,
                    "result": result_label,
                    "city": row["city"],
                    "country": row["country"],
                    "neutral": row["neutral"] == "TRUE",
                    "team_elo_start": elo.team_elo_start if elo else "",
                    "opponent_elo_start": elo.opponent_elo_start if elo else "",
                    "team_elo_end": elo.team_elo_end if elo else "",
                    "opponent_elo_end": elo.opponent_elo_end if elo else "",
                    "team_elo_delta": elo.team_elo_delta if elo else "",
                }
            )

    for team_name, team_meta in team_reference.items():
        if team_name in TEAM_REFERENCE_ALIAS_NAMES:
            continue
        confederation_folder = CONFEDERATION_FOLDER_NAMES.get(team_meta.confederation_code, team_meta.confederation_code.lower())
        team_folder = BY_CONFEDERATION_DIR / confederation_folder / slugify_path_component(team_name)
        write_csv(
            team_folder / "results.csv",
            sorted(rows_by_team.get(team_name, []), key=lambda item: item["date"]),
            COUNTRY_RESULTS_FIELDS,
        )
        write_csv(
            team_folder / "schedule.csv",
            sorted(country_schedule_rows.get(team_name, []), key=lambda item: (item["year"], item["match_number"])),
            COUNTRY_SCHEDULE_FIELDS,
        )
        write_csv(
            team_folder / "squads.csv",
            sorted(country_squad_rows.get(team_name, []), key=lambda item: (item["year"], item["player_name"])),
            SQUAD_FIELDS,
        )


def main() -> None:
    former_name_map = build_former_name_map()
    reverse_historical_mappings = build_reverse_historical_mappings()
    tournaments = load_tournaments()
    team_reference = load_team_reference(former_name_map, reverse_historical_mappings)
    history = load_history(former_name_map, reverse_historical_mappings)
    matches_by_year = load_world_cup_matches(former_name_map, reverse_historical_mappings)
    shootout_lookup = build_shootout_lookup(former_name_map, reverse_historical_mappings)

    team_years: set[tuple[str, int]] = set()
    for row in load_csv(WORLDCUP_SQUADS_PATH):
        if not row["tournament_id"].startswith("WC-"):
            continue
        year = int(row["tournament_id"].split("-")[1])
        tournament = tournaments.get(year)
        if tournament is None:
            continue
        team_name = canonicalize_name(
            row["team_name"], former_name_map, reverse_historical_mappings, tournament.start_date.isoformat()
        )
        team_years.add((team_name, year))

    print("Building National Football Teams country index")
    ntf_countries = build_ntf_country_index()
    print(f"Fetching squad enrichment pages for {len(team_years)} team-year combinations")
    ntf_enrichment = build_ntf_enrichment(team_years, ntf_countries)
    squad_rows_by_year = build_squad_rows_by_year(
        tournaments, team_reference, former_name_map, reverse_historical_mappings, ntf_enrichment
    )

    print("Loading Elo dictionaries")
    successor_map, names_by_code, code_by_name = build_elo_team_data()
    historical_teams = {team for rows in matches_by_year.values() for match in rows for team in (match["home_team"], match["away_team"])}
    team_to_elo_code = {team_name: map_team_to_elo_code(team_name, code_by_name) for team_name in historical_teams}
    print("Building latest Elo snapshot from live team pages")
    latest_elo_by_team = build_latest_elo_by_team(
        team_to_elo_code=team_to_elo_code,
        successor_map=successor_map,
        names_by_code=names_by_code,
        former_name_map=former_name_map,
        reverse_historical_mappings=reverse_historical_mappings,
    )

    placement_rows_by_year: dict[int, list[dict[str, object]]] = {}
    country_schedule_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    country_squad_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    merged_schedule_rows: list[dict[str, object]] = []
    merged_result_rows: list[dict[str, object]] = []
    merged_teams_rows: list[dict[str, object]] = []
    merged_squad_rows: list[dict[str, object]] = []
    merged_elo_rows: list[dict[str, object]] = []

    for year in sorted(history):
        print(f"Building World Cup {year}")
        tournament_name, tournament_id = tournament_identity(year, tournaments)
        raw_matches = matches_by_year.get(year, [])
        if not raw_matches:
            raise ValueError(f"{year}: no World Cup matches found in results.csv")
        page_base = f"{year}_World_Cup"
        start_ratings_by_code = parse_elo_rating_table(request_text(f"{ELO_BASE_URL}/{page_base}_start.tsv"))
        end_ratings_by_code = parse_elo_rating_table(request_text(f"{ELO_BASE_URL}/{page_base}.tsv"))
        match_elo_lookup = parse_elo_match_rows(
            request_text(f"{ELO_BASE_URL}/{page_base}_results.tsv"),
            names_by_code,
            former_name_map,
            reverse_historical_mappings,
        )
        start_elo_by_team: dict[str, int] = {}
        end_elo_by_team: dict[str, int] = {}
        for team_name in {match["home_team"] for match in raw_matches} | {match["away_team"] for match in raw_matches}:
            code = team_to_elo_code[team_name]
            if code in start_ratings_by_code:
                start_elo_by_team[team_name] = start_ratings_by_code[code]
            if code in end_ratings_by_code:
                end_elo_by_team[team_name] = end_ratings_by_code[code]

        schedule_rows, result_rows, placement_rows = build_outputs_for_year(
            year=year,
            tournament_id=tournament_id,
            raw_matches=raw_matches,
            summary=history[year],
            team_reference=team_reference,
            shootout_lookup=shootout_lookup,
            elo_lookup=match_elo_lookup,
            start_elo_by_team=start_elo_by_team,
            end_elo_by_team=end_elo_by_team,
            latest_elo_by_team=latest_elo_by_team,
        )
        validate_year(year, schedule_rows, result_rows, placement_rows, history[year])

        output_dir = INT_WORLD_CUP_DIR / str(year)
        write_csv(output_dir / "schedule.csv", schedule_rows, SCHEDULE_FIELDS)
        write_csv(output_dir / "results.csv", result_rows, RESULT_FIELDS)
        write_csv(output_dir / "placement.csv", placement_rows, PLACEMENT_FIELDS)
        write_csv(output_dir / "squads.csv", squad_rows_by_year.get(year, []), SQUAD_FIELDS)
        teams_rows = build_teams_rows(year, tournament_id, tournament_name, placement_rows)
        elo_rows = build_elo_rows_from_placement(placement_rows)
        write_csv(output_dir / "teams.csv", teams_rows, TEAMS_FIELDS)
        write_csv(output_dir / "elo.csv", elo_rows, ELO_FIELDS)

        placement_rows_by_year[year] = [{"edition": year, **row} for row in placement_rows]
        for row in squad_rows_by_year.get(year, []):
            country_squad_rows[str(row["team"])].append(row)
        append_country_schedule_rows(country_schedule_rows, schedule_rows, year, tournament_name, tournament_id)
        merged_schedule_rows.extend({"edition": year, **row} for row in schedule_rows)
        merged_result_rows.extend({"edition": year, **row} for row in result_rows)
        merged_teams_rows.extend(teams_rows)
        merged_squad_rows.extend(squad_rows_by_year.get(year, []))
        merged_elo_rows.extend(elo_rows)
        print(f"Wrote {year}: {len(result_rows)} matches, {len(placement_rows)} teams, {len(squad_rows_by_year.get(year, []))} squad rows")
        time.sleep(0.05)

    annotate_next_edition_placements(placement_rows_by_year)
    for year, annotated_rows in placement_rows_by_year.items():
        output_dir = INT_WORLD_CUP_DIR / str(year)
        per_year_rows = [{field: row.get(field, "") for field in PLACEMENT_FIELDS} for row in annotated_rows]
        write_csv(output_dir / "placement.csv", per_year_rows, PLACEMENT_FIELDS)
        teams_rows = build_teams_rows(
            year,
            str(per_year_rows[0]["tournament_id"]) if per_year_rows else f"WC-{year}",
            tournament_identity(year, tournaments)[0],
            per_year_rows,
        )
        elo_rows = build_elo_rows_from_placement(per_year_rows)
        write_csv(output_dir / "teams.csv", teams_rows, TEAMS_FIELDS)
        write_csv(output_dir / "elo.csv", elo_rows, ELO_FIELDS)

    merged_placement_rows = [
        {"edition": year, **{field: row.get(field, "") for field in PLACEMENT_FIELDS}}
        for year, annotated_rows in placement_rows_by_year.items()
        for row in annotated_rows
    ]
    merged_teams_rows = [
        row
        for year, annotated_rows in placement_rows_by_year.items()
        for row in build_teams_rows(
            year,
            str(annotated_rows[0]["tournament_id"]) if annotated_rows else f"WC-{year}",
            tournament_identity(year, tournaments)[0],
            [{field: item.get(field, "") for field in PLACEMENT_FIELDS} for item in annotated_rows],
        )
    ]
    merged_elo_rows = [
        row
        for annotated_rows in placement_rows_by_year.values()
        for row in build_elo_rows_from_placement(
            [{field: item.get(field, "") for field in PLACEMENT_FIELDS} for item in annotated_rows]
        )
    ]
    write_csv(ALL_EDITIONS_DIR / "schedule.csv", merged_schedule_rows, ["edition", *SCHEDULE_FIELDS])
    write_csv(ALL_EDITIONS_DIR / "results.csv", merged_result_rows, ["edition", *RESULT_FIELDS])
    write_csv(ALL_EDITIONS_DIR / "placement.csv", merged_placement_rows, ["edition", *PLACEMENT_FIELDS])
    write_csv(ALL_EDITIONS_DIR / "teams.csv", merged_teams_rows, TEAMS_FIELDS)
    write_csv(ALL_EDITIONS_DIR / "squads.csv", merged_squad_rows, SQUAD_FIELDS)
    write_csv(ALL_EDITIONS_DIR / "elo.csv", merged_elo_rows, ELO_FIELDS)

    print("Building country Elo lookup")
    country_match_elo = build_country_match_elo_lookup(
        team_to_elo_code={team: team_to_elo_code[team] for team in team_reference if team in team_to_elo_code},
        successor_map=successor_map,
        names_by_code=names_by_code,
        former_name_map=former_name_map,
        reverse_historical_mappings=reverse_historical_mappings,
    )

    for year in sorted(year for year in matches_by_year if year not in history):
        raw_matches = matches_by_year[year]
        tournament_name, tournament_id = tournament_identity(year, tournaments)
        schedule_rows = build_schedule_only_rows_for_year(year, tournament_id, raw_matches, team_reference, latest_elo_by_team)
        output_dir = INT_WORLD_CUP_DIR / str(year)
        write_csv(output_dir / "schedule.csv", schedule_rows, SCHEDULE_FIELDS)
        append_country_schedule_rows(country_schedule_rows, schedule_rows, year, tournament_name, tournament_id)
        print(f"Wrote {year} schedule-only export: {len(schedule_rows)} matches")

    print("Writing by_confederation exports")
    build_country_exports(
        team_reference=team_reference,
        former_name_map=former_name_map,
        reverse_historical_mappings=reverse_historical_mappings,
        country_match_elo=country_match_elo,
        country_schedule_rows=country_schedule_rows,
        country_squad_rows=country_squad_rows,
    )
    print("Completed historical World Cup dataset build")


if __name__ == "__main__":
    main()
