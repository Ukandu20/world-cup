from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORLD_CUP_DIR = ROOT / "INT-World Cup" / "world_cup"
MERGED_DIR = WORLD_CUP_DIR / "all_editions"
RESULTS_PATH = WORLD_CUP_DIR / "results.csv"
SHOOTOUTS_PATH = WORLD_CUP_DIR / "shootouts.csv"
FORMER_NAMES_PATH = WORLD_CUP_DIR / "former_names.csv"
HISTORY_PATH = WORLD_CUP_DIR / "fifa_world_cup_history.csv"
FORMATS_PATH = WORLD_CUP_DIR / "world_cup_historical_tournament_formats_1930_2022.csv"

PLACEMENT_FIELDS = [
    "country",
    "placement",
    "position",
    "matches_played",
    "gs",
    "ga",
    "next_edition",
    "next_placement",
    "next_position",
]
SCHEDULE_FIELDS = [
    "match_number",
    "date",
    "stage",
    "home_team",
    "away_team",
    "city",
    "country",
    "neutral",
]
RESULT_FIELDS = [
    "match_number",
    "date",
    "stage",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "city",
    "country",
    "neutral",
    "decided_by_shootout",
    "shootout_winner",
]
MERGED_PLACEMENT_FIELDS = ["edition", *PLACEMENT_FIELDS]
MERGED_RESULT_FIELDS = ["edition", *RESULT_FIELDS]

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

# Preserve dissolved historical entities as distinct teams in outputs.
PRESERVE_HISTORICAL = {
    "CIS",
    "Czechoslovakia",
    "FR Yugoslavia",
    "German DR",
    "Serbia and Montenegro",
    "Soviet Union",
    "Yugoslavia",
}
DIRECT_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curacao",
    "CuraÃ§ao": "Curacao",
    "Czech Republic": "Czechia",
    "CÃ´te d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Dutch East Indies": "Indonesia",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "TÃ¼rkiye": "Turkey",
    "Türkiye": "Turkey",
    "USA": "United States",
    "West Germany": "Germany",
    "Zaïre": "DR Congo",
    "ZaÃ¯re": "DR Congo",
    "Zaire": "DR Congo",
}


@dataclass(frozen=True)
class EditionSummary:
    year: int
    host_country: str
    winner: str
    runner_up: str
    third_place: str
    fourth_place: str
    total_goals: int
    matches_played: int
    teams: int


@dataclass(frozen=True)
class ReverseHistoricalMapping:
    current: str
    former: str
    start_date: date
    end_date: date


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-zA-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


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


def build_former_name_map() -> dict[str, str]:
    mapping = {normalize_key(source): target for source, target in DIRECT_ALIASES.items()}
    for row in load_csv(FORMER_NAMES_PATH):
        former = row["former"].strip()
        current = row["current"].strip()
        if not former or not current:
            continue
        if former in PRESERVE_HISTORICAL:
            continue
        if current in {"Germany", "Russia", "Serbia"} and former in PRESERVE_HISTORICAL:
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
        if current not in {"Russia", "Serbia"}:
            continue
        if not start_date or not end_date:
            continue
        reverse[normalize_key(current)].append(
            ReverseHistoricalMapping(
                current=current,
                former=former,
                start_date=date.fromisoformat(start_date),
                end_date=date.fromisoformat(end_date),
            )
        )
    return reverse


def canonicalize_name(
    name: str,
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]] | None = None,
    match_date: str | None = None,
) -> str:
    stripped = name.strip()
    if stripped in PRESERVE_HISTORICAL:
        return stripped
    key = normalize_key(stripped)
    if match_date and reverse_historical_mappings:
        current_match_date = date.fromisoformat(match_date)
        for mapping in reverse_historical_mappings.get(key, []):
            if mapping.start_date <= current_match_date <= mapping.end_date:
                return mapping.former
    return former_name_map.get(key, stripped)


def load_history(former_name_map: dict[str, str]) -> dict[int, EditionSummary]:
    summaries: dict[int, EditionSummary] = {}
    for row in load_csv(HISTORY_PATH):
        year = int(row["Year"])
        summaries[year] = EditionSummary(
            year=year,
            host_country=row["Host_Country"],
            winner=canonicalize_name(row["Winner"], former_name_map),
            runner_up=canonicalize_name(row["Runner_Up"], former_name_map),
            third_place=canonicalize_name(row["Third_Place"], former_name_map),
            fourth_place=canonicalize_name(row["Fourth_Place"], former_name_map),
            total_goals=int(row["Total_Goals"]),
            matches_played=int(row["Matches_Played"]),
            teams=int(row["Teams"]),
        )
    return summaries


def load_formats() -> dict[int, dict[str, str]]:
    return {int(row["year"]): row for row in load_csv(FORMATS_PATH)}


def load_world_cup_matches() -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in load_csv(RESULTS_PATH):
        if row["tournament"] != "FIFA World Cup":
            continue
        grouped[int(row["date"][:4])].append(row)
    return dict(sorted(grouped.items()))


def build_shootout_lookup(former_name_map: dict[str, str]) -> dict[tuple[str, str, str], str]:
    reverse_historical_mappings = build_reverse_historical_mappings()
    lookup: dict[tuple[str, str, str], str] = {}
    for row in load_csv(SHOOTOUTS_PATH):
        key = (
            row["date"],
            canonicalize_name(
                row["home_team"], former_name_map, reverse_historical_mappings, row["date"]
            ),
            canonicalize_name(
                row["away_team"], former_name_map, reverse_historical_mappings, row["date"]
            ),
        )
        lookup[key] = canonicalize_name(
            row["winner"], former_name_map, reverse_historical_mappings, row["date"]
        )
    return lookup


def assign_stages(year: int, match_count: int) -> list[str]:
    plan = YEAR_STAGE_PLANS.get(year)
    if plan is None:
        raise ValueError(f"No stage plan configured for {year}.")

    stages: list[str] = []
    for stage, count in plan:
        stages.extend([stage] * count)
    if len(stages) != match_count:
        raise ValueError(
            f"Stage plan for {year} covers {len(stages)} matches, expected {match_count}."
        )
    return stages


def top_four_placements(summary: EditionSummary) -> dict[str, tuple[str, int]]:
    return {
        summary.winner: ("Winner", TOP_FOUR_POSITIONS["Winner"]),
        summary.runner_up: ("Runner-up", TOP_FOUR_POSITIONS["Runner-up"]),
        summary.third_place: ("Third Place", TOP_FOUR_POSITIONS["Third Place"]),
        summary.fourth_place: ("Fourth Place", TOP_FOUR_POSITIONS["Fourth Place"]),
    }


def build_outputs_for_year(
    year: int,
    raw_matches: list[dict[str, str]],
    summary: EditionSummary,
    shootout_lookup: dict[tuple[str, str, str], str],
    former_name_map: dict[str, str],
    reverse_historical_mappings: dict[str, list[ReverseHistoricalMapping]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    stages = assign_stages(year, len(raw_matches))
    schedule_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    stats: dict[str, dict[str, int | set[str]]] = defaultdict(
        lambda: {"matches_played": 0, "gs": 0, "ga": 0, "stages": set()}
    )

    for match_number, (row, stage) in enumerate(zip(raw_matches, stages, strict=True), start=1):
        home_team = canonicalize_name(
            row["home_team"], former_name_map, reverse_historical_mappings, row["date"]
        )
        away_team = canonicalize_name(
            row["away_team"], former_name_map, reverse_historical_mappings, row["date"]
        )
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        shootout_key = (row["date"], home_team, away_team)
        shootout_winner = shootout_lookup.get(shootout_key, "")
        decided_by_shootout = bool(shootout_winner)

        schedule_rows.append(
            {
                "match_number": match_number,
                "date": row["date"],
                "stage": stage,
                "home_team": home_team,
                "away_team": away_team,
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"],
            }
        )
        result_rows.append(
            {
                "match_number": match_number,
                "date": row["date"],
                "stage": stage,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "city": row["city"],
                "country": row["country"],
                "neutral": row["neutral"],
                "decided_by_shootout": decided_by_shootout,
                "shootout_winner": shootout_winner,
            }
        )

        for team, gf, ga in ((home_team, home_score, away_score), (away_team, away_score, home_score)):
            team_stats = stats[team]
            team_stats["matches_played"] += 1
            team_stats["gs"] += gf
            team_stats["ga"] += ga
            stage_set = team_stats["stages"]
            assert isinstance(stage_set, set)
            stage_set.add(stage)

    placement_rows = build_placement_rows(stats, summary)
    return schedule_rows, result_rows, placement_rows


def build_placement_rows(
    stats: dict[str, dict[str, int | set[str]]],
    summary: EditionSummary,
) -> list[dict[str, object]]:
    placements = top_four_placements(summary)
    rows: list[dict[str, object]] = []

    for country in sorted(stats):
        team_stats = stats[country]
        stage_set = team_stats["stages"]
        assert isinstance(stage_set, set)

        placement, position = placements.get(country, ("", 0))
        if not placement:
            # Use deepest standardized stage reached for non-top-four teams.
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
                position = summary.teams

        rows.append(
            {
                "country": country,
                "placement": placement,
                "position": position,
                "matches_played": team_stats["matches_played"],
                "gs": team_stats["gs"],
                "ga": team_stats["ga"],
                "next_edition": "",
                "next_placement": "",
                "next_position": "",
            }
        )

    rows.sort(key=lambda row: (int(row["position"]), str(row["country"])))
    return rows


def annotate_next_edition_placements(
    placement_rows_by_year: dict[int, list[dict[str, object]]]
) -> None:
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
    summary: EditionSummary,
) -> None:
    if len(result_rows) != summary.matches_played:
        raise ValueError(
            f"{year}: results row count {len(result_rows)} != official {summary.matches_played}"
        )
    if len(schedule_rows) != summary.matches_played:
        raise ValueError(
            f"{year}: schedule row count {len(schedule_rows)} != official {summary.matches_played}"
        )
    if len(placement_rows) != summary.teams:
        raise ValueError(
            f"{year}: placement row count {len(placement_rows)} != official {summary.teams}"
        )

    stage_pairs = [(row["match_number"], row["stage"]) for row in schedule_rows]
    result_stage_pairs = [(row["match_number"], row["stage"]) for row in result_rows]
    if stage_pairs != result_stage_pairs:
        raise ValueError(f"{year}: schedule and results stages do not line up row-for-row")

    total_matches_played = sum(int(row["matches_played"]) for row in placement_rows)
    if total_matches_played != summary.matches_played * 2:
        raise ValueError(
            f"{year}: summed matches_played {total_matches_played} != {summary.matches_played * 2}"
        )

    total_gs = sum(int(row["gs"]) for row in placement_rows)
    total_ga = sum(int(row["ga"]) for row in placement_rows)
    if total_gs != summary.total_goals or total_ga != summary.total_goals:
        raise ValueError(
            f"{year}: summed goals ({total_gs}/{total_ga}) != official {summary.total_goals}"
        )

    placement_lookup = {str(row["country"]): str(row["placement"]) for row in placement_rows}
    expected = top_four_placements(summary)
    for country, (placement, _) in expected.items():
        if placement_lookup.get(country) != placement:
            raise ValueError(
                f"{year}: expected {country} to be {placement}, got {placement_lookup.get(country)}"
            )


def main() -> None:
    former_name_map = build_former_name_map()
    reverse_historical_mappings = build_reverse_historical_mappings()
    history = load_history(former_name_map)
    formats = load_formats()
    matches_by_year = load_world_cup_matches()
    shootout_lookup = build_shootout_lookup(former_name_map)
    placement_rows_by_year: dict[int, list[dict[str, object]]] = {}
    merged_placement_rows: list[dict[str, object]] = []
    merged_result_rows: list[dict[str, object]] = []

    for year, summary in history.items():
        raw_matches = matches_by_year.get(year, [])
        if not raw_matches:
            raise ValueError(f"{year}: no World Cup matches found in results.csv")
        if year not in formats:
            raise ValueError(f"{year}: missing tournament format row")
        if int(formats[year]["teams"]) != summary.teams:
            raise ValueError(
                f"{year}: teams mismatch between formats ({formats[year]['teams']}) and history ({summary.teams})"
            )

        schedule_rows, result_rows, placement_rows = build_outputs_for_year(
            year=year,
            raw_matches=raw_matches,
            summary=summary,
            shootout_lookup=shootout_lookup,
            former_name_map=former_name_map,
            reverse_historical_mappings=reverse_historical_mappings,
        )
        validate_year(year, schedule_rows, result_rows, placement_rows, summary)

        output_dir = WORLD_CUP_DIR / str(year)
        write_csv(output_dir / "schedule.csv", schedule_rows, SCHEDULE_FIELDS)
        write_csv(output_dir / "results.csv", result_rows, RESULT_FIELDS)
        placement_rows_by_year[year] = []

        merged_result_rows.extend({"edition": year, **row} for row in result_rows)
        for row in placement_rows:
            placement_rows_by_year[year].append({"edition": year, **row})

        print(
            f"Wrote {year}: {len(result_rows)} matches, {len(placement_rows)} teams -> {output_dir}"
        )

    annotate_next_edition_placements(placement_rows_by_year)

    for year, annotated_rows in placement_rows_by_year.items():
        output_dir = WORLD_CUP_DIR / str(year)
        per_year_rows = [{field: row.get(field, "") for field in PLACEMENT_FIELDS} for row in annotated_rows]
        write_csv(output_dir / "placement.csv", per_year_rows, PLACEMENT_FIELDS)
        merged_placement_rows.extend(annotated_rows)

    write_csv(MERGED_DIR / "results.csv", merged_result_rows, MERGED_RESULT_FIELDS)
    write_csv(MERGED_DIR / "placement.csv", merged_placement_rows, MERGED_PLACEMENT_FIELDS)
    print(
        f"Wrote merged outputs: {len(merged_result_rows)} matches, "
        f"{len(merged_placement_rows)} team-edition rows -> {MERGED_DIR}"
    )


if __name__ == "__main__":
    main()
