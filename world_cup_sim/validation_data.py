from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .constants import REPO_ROOT, WORLD_CUP_ROOT


RAW_2026_LEAD_IN_PATH = REPO_ROOT / "INT-World Cup" / "world_cup" / "2026" / "team_results_lead_in.csv"
PROCESSED_2026_LEAD_IN_PATH = WORLD_CUP_ROOT / "2026" / "team_results_lead_in.csv"
ALL_EDITIONS_DIR = WORLD_CUP_ROOT / "all_editions"
ALL_EDITIONS_LEAD_IN_PATH = ALL_EDITIONS_DIR / "team_results_lead_in.csv"

MODELING_DATASETS = ("results", "schedule", "teams", "lead_in")
FORBIDDEN_MODELING_COLUMNS = frozenset({"next_edition", "next_placement", "next_position"})


def _normalized_tournament_id_part(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "unknown"


def harmonize_lead_in_frame(lead_in: pd.DataFrame) -> pd.DataFrame:
    """Preserve the canonical lead-in schema and add match-level compatibility columns."""
    out = lead_in.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    tournament = out["tournament"].fillna("Unknown") if "tournament" in out.columns else pd.Series("Unknown", index=out.index)
    year = out["date"].dt.year.astype("Int64").astype(str).replace("<NA>", "unknown")

    if "tournament_id" not in out.columns:
        tournament_part = tournament.map(_normalized_tournament_id_part)
        out["tournament_id"] = year + "_" + tournament_part
    if "match_id" not in out.columns:
        if "match_key" in out.columns:
            out["match_id"] = out["match_key"].astype(str)
        else:
            out["match_id"] = out.get("lead_in_id", pd.Series(out.index, index=out.index)).astype(str)
    if "match_number" not in out.columns:
        out["match_number"] = pd.NA
    if "stage" not in out.columns:
        out["stage"] = tournament.astype(str)
    if "status" not in out.columns:
        out["status"] = "completed"

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def refresh_all_editions_lead_in(
    raw_path: Path = RAW_2026_LEAD_IN_PATH,
    processed_path: Path = PROCESSED_2026_LEAD_IN_PATH,
    output_path: Path = ALL_EDITIONS_LEAD_IN_PATH,
) -> Path:
    """Build the all-editions lead-in file from the canonical 2026 lead-in schema."""
    source_path = raw_path if raw_path.exists() else processed_path
    if not source_path.exists():
        raise FileNotFoundError(f"No canonical lead-in file found at {raw_path} or {processed_path}")

    lead_in = pd.read_csv(source_path)
    lead_in = harmonize_lead_in_frame(lead_in)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lead_in.to_csv(output_path, index=False)
    return output_path


def load_all_data(refresh_lead_in: bool = False) -> dict[str, pd.DataFrame]:
    """Load validation data once and return all DataFrames needed by the pipeline."""
    if refresh_lead_in or not ALL_EDITIONS_LEAD_IN_PATH.exists():
        refresh_all_editions_lead_in()

    data = {
        "results": pd.read_csv(ALL_EDITIONS_DIR / "results.csv", parse_dates=["date"]),
        "schedule": pd.read_csv(ALL_EDITIONS_DIR / "schedule.csv", parse_dates=["date"]),
        "teams": pd.read_csv(ALL_EDITIONS_DIR / "teams.csv"),
        "lead_in": pd.read_csv(ALL_EDITIONS_LEAD_IN_PATH, parse_dates=["date"]),
        "placement": pd.read_csv(ALL_EDITIONS_DIR / "placement.csv"),
    }
    validate_schema_integrity(data)
    return data


def validate_schema_integrity(data: dict[str, pd.DataFrame]) -> None:
    """Validate the modeling data contract while allowing placement reporting columns."""
    required_columns = {
        "results": {"edition", "tournament_id", "team_id", "team_score", "opponent_score", "result", "date"},
        "schedule": {"edition", "tournament_id", "home_team_id", "away_team_id", "date", "stage"},
        "teams": {"year", "team_id", "team", "position"},
        "lead_in": {
            "qualified_team_id",
            "date",
            "team_score",
            "opponent_score",
            "goal_difference",
            "result",
            "team_elo_start",
            "opponent_elo_start",
            "tournament_id",
            "match_id",
            "stage",
            "status",
        },
        "placement": {"edition", "year", "team_id", "position"},
    }
    for name, columns in required_columns.items():
        if name not in data:
            raise ValueError(f"Missing validation dataset: {name}")
        missing = columns - set(data[name].columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    for name in MODELING_DATASETS:
        forbidden = FORBIDDEN_MODELING_COLUMNS.intersection(data[name].columns)
        if forbidden:
            raise ValueError(f"{name} contains future-looking modeling columns: {sorted(forbidden)}")

    lead_in = data["lead_in"].copy()
    if not lead_in.empty:
        lead_in["date"] = pd.to_datetime(lead_in["date"], errors="coerce")
        li = lead_in.dropna(subset=["date"]).sort_values(["qualified_team_id", "date"], kind="stable").copy()
        if {"team_elo_start", "team_elo_delta"}.issubset(li.columns):
            li["elo_end"] = pd.to_numeric(li["team_elo_start"], errors="coerce") + pd.to_numeric(
                li["team_elo_delta"],
                errors="coerce",
            )
            li["next_elo_start"] = li.groupby("qualified_team_id")["team_elo_start"].shift(-1)
            gaps = (li["elo_end"] - pd.to_numeric(li["next_elo_start"], errors="coerce")).abs().dropna()
            large_gap_count = int(gaps.gt(1.0).sum())
            if large_gap_count:
                print(f"WARNING: {large_gap_count} Elo continuity gaps > 1 point in team_results_lead_in.csv")
