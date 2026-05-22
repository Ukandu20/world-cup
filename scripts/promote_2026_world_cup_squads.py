from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = ROOT / "INT-World Cup" / "world_cup"
PROCESSED_DIR = ROOT / "data" / "processed" / "world_cup"
STAGING_2026_DIR = STAGING_DIR / "2026"
PROCESSED_2026_DIR = PROCESSED_DIR / "2026"

TOURNAMENT_ID = "WC-2026"
TOURNAMENT_NAME = "2026 FIFA Men's World Cup"
SOURCE_AS_OF = "2026-05-22"

SQUAD_FIELDS_2026 = [
    "tournament_id",
    "edition_year",
    "year",
    "team_id",
    "team",
    "team_code",
    "confederation",
    "group",
    "coach",
    "player_name",
    "position",
    "pos_code",
    "jersey_number",
    "date_of_birth",
    "club",
    "caps",
    "goals",
    "is_captain",
    "is_final_squad",
    "source_url",
    "source_as_of",
]

ALL_EDITIONS_FIELDS = [
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
    "is_captain",
    "group",
    "coach",
    "is_final_squad",
    "source_url",
    "source_as_of",
]

STATUS_FIELDS = [
    "team_name",
    "team_id",
    "team_code",
    "confederation",
    "group",
    "coach",
    "is_final_squad",
    "row_count",
    "announcement_text",
    "source_url",
    "source_as_of",
]

VALIDATION_FIELDS = [
    "team_name",
    "group",
    "is_final_squad",
    "row_count",
    "expected_count",
    "status",
    "message",
]

REQUIRED_SQUAD_FIELDS = [
    "team_id",
    "team",
    "player_name",
    "position",
    "pos_code",
    "date_of_birth",
    "club",
    "is_final_squad",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize(row.get(field, "")) for field in fieldnames})


def serialize(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def normalize_bool(value: str) -> str:
    return "TRUE" if str(value).strip().upper() == "TRUE" else "FALSE"


def normalize_squad_row(row: dict[str, str]) -> dict[str, object]:
    normalized: dict[str, object] = {field: row.get(field, "") for field in SQUAD_FIELDS_2026}
    normalized["tournament_id"] = TOURNAMENT_ID
    normalized["edition_year"] = "2026"
    normalized["year"] = "2026"
    normalized["is_captain"] = normalize_bool(str(normalized["is_captain"]))
    normalized["is_final_squad"] = normalize_bool(str(normalized["is_final_squad"]))
    normalized["source_as_of"] = normalized.get("source_as_of") or SOURCE_AS_OF
    return normalized


def validate_staged_inputs(
    squad_rows: list[dict[str, object]],
    status_rows: list[dict[str, str]],
    processed_teams: list[dict[str, str]],
) -> None:
    if len(squad_rows) != 1112:
        raise ValueError(f"Expected 1,112 staged squad rows, found {len(squad_rows)}")
    if len(status_rows) != 48:
        raise ValueError(f"Expected 48 staged team status rows, found {len(status_rows)}")

    teams_by_id = {row["team_id"]: row for row in processed_teams}
    if len(teams_by_id) != 48:
        raise ValueError(f"Expected 48 processed 2026 teams, found {len(teams_by_id)}")

    duplicate_counts = Counter(
        (str(row["team_id"]), str(row["player_name"]), str(row["date_of_birth"]))
        for row in squad_rows
    )
    duplicates = [key for key, count in duplicate_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Found duplicate staged squad player keys: {duplicates[:5]}")

    for index, row in enumerate(squad_rows, start=2):
        missing = [field for field in REQUIRED_SQUAD_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Staged squads row {index} missing required fields: {', '.join(missing)}")

        team = teams_by_id.get(str(row["team_id"]))
        if not team:
            raise ValueError(f"Staged squads row {index} has unknown team_id: {row['team_id']}")
        if str(row["team"]) != team["team"]:
            raise ValueError(
                f"Staged squads row {index} team mismatch for {row['team_id']}: "
                f"{row['team']} != {team['team']}"
            )

    status_team_ids = {row.get("team_id", "") for row in status_rows}
    missing_status = sorted(set(teams_by_id) - status_team_ids)
    if missing_status:
        raise ValueError(f"Missing squad status rows for team IDs: {missing_status}")


def to_all_editions_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "team": row["team"],
        "team_id": row["team_id"],
        "team_code": row["team_code"],
        "confederation": row["confederation"],
        "tournament": TOURNAMENT_NAME,
        "tournament_id": row["tournament_id"],
        "year": row["year"],
        "player_name": row["player_name"],
        "player_id": "",
        "position": row["position"],
        "pos_code": row["pos_code"],
        "shirt_number": row["jersey_number"],
        "date_of_birth": row["date_of_birth"],
        "age": "",
        "club": row["club"],
        "club_country": "",
        "caps": row["caps"],
        "goals": row["goals"],
        "is_captain": row["is_captain"],
        "group": row["group"],
        "coach": row["coach"],
        "is_final_squad": row["is_final_squad"],
        "source_url": row["source_url"],
        "source_as_of": row["source_as_of"],
    }


def promote_all_editions(squad_rows: list[dict[str, object]]) -> None:
    path = PROCESSED_DIR / "all_editions" / "squads.csv"
    existing = load_csv(path)
    kept = [
        row
        for row in existing
        if row.get("tournament_id") != TOURNAMENT_ID and row.get("year") != "2026"
    ]
    for row in kept:
        for field in ALL_EDITIONS_FIELDS:
            row.setdefault(field, "")
    write_csv(path, kept + [to_all_editions_row(row) for row in squad_rows], ALL_EDITIONS_FIELDS)


def squad_status_for_team(status_row: dict[str, str]) -> str:
    row_count = int(status_row.get("row_count") or "0")
    if row_count == 0:
        return "pending_no_table"
    if normalize_bool(status_row.get("is_final_squad", "")) == "TRUE":
        return "final"
    return "preliminary"


def update_teams(status_rows: list[dict[str, str]]) -> None:
    path = PROCESSED_2026_DIR / "teams.csv"
    teams = load_csv(path)
    status_by_team_id = {row["team_id"]: row for row in status_rows}
    for row in teams:
        status_row = status_by_team_id[row["team_id"]]
        row["squad_status"] = squad_status_for_team(status_row)
        row["source_as_of"] = SOURCE_AS_OF
    write_csv(path, teams, list(teams[0].keys()))


def update_edition_metadata() -> None:
    path = PROCESSED_2026_DIR / "edition_metadata.csv"
    rows = load_csv(path)
    for row in rows:
        if row.get("edition_year") == "2026":
            row["squads_available"] = "TRUE"
            row["squad_status"] = "partial_final_and_preliminary"
            row["source_as_of"] = SOURCE_AS_OF
    write_csv(path, rows, list(rows[0].keys()))


def main() -> None:
    staged_squads = [normalize_squad_row(row) for row in load_csv(STAGING_2026_DIR / "squads.csv")]
    staged_status = load_csv(STAGING_2026_DIR / "squads_teams_status.csv")
    staged_validation = load_csv(STAGING_2026_DIR / "squads_validation_report.csv")
    processed_teams = load_csv(PROCESSED_2026_DIR / "teams.csv")

    validate_staged_inputs(staged_squads, staged_status, processed_teams)

    write_csv(PROCESSED_2026_DIR / "squads.csv", staged_squads, SQUAD_FIELDS_2026)
    write_csv(PROCESSED_2026_DIR / "squads_teams_status.csv", staged_status, STATUS_FIELDS)
    write_csv(PROCESSED_2026_DIR / "squads_validation_report.csv", staged_validation, VALIDATION_FIELDS)
    promote_all_editions(staged_squads)
    update_teams(staged_status)
    update_edition_metadata()

    final_teams = sum(1 for row in staged_status if normalize_bool(row["is_final_squad"]) == "TRUE")
    captains = sum(1 for row in staged_squads if row["is_captain"] == "TRUE")
    print(f"processed_squad_rows={len(staged_squads)}")
    print(f"team_status_rows={len(staged_status)}")
    print(f"final_teams={final_teams}")
    print(f"non_final_or_pending_teams={len(staged_status) - final_teams}")
    print(f"captains={captains}")
    print(f"validation_warnings={len(staged_validation)}")


if __name__ == "__main__":
    main()
