from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from .config import CHAMPION_TROPHY_PATH, DATA_DIR, WORLD_CUP_LOGO_PATH

def fix_mojibake(value: str) -> str:
    """Repair common UTF-8 decoding artifacts in source text fields."""
    if not isinstance(value, str):
        return value
    if all(marker not in value for marker in ("\u00c3", "\u00c2")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


@st.cache_data(show_spinner=False)
def load_svg_data_uri(svg_path: str) -> str:
    """Load a local SVG file as a data URI for inline display and export."""
    path = Path(svg_path)
    if not path.exists():
        return ""
    svg_bytes = path.read_bytes()
    encoded = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_world_cup_logo_data_uri() -> str:
    """Load the dashboard World Cup logo as a data URI."""
    return load_svg_data_uri(str(WORLD_CUP_LOGO_PATH))


@st.cache_data(show_spinner=False)
def load_champion_trophy_data_uri() -> str:
    """Load the champion trophy SVG as a data URI."""
    return load_svg_data_uri(str(CHAMPION_TROPHY_PATH))


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load the dashboard inputs: teams, ratings, fixtures, lead-in form, and metadata."""
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    groups = pd.read_csv(DATA_DIR / "groups.csv")
    fifa = pd.read_csv(DATA_DIR / "fifa_rank_snapshots.csv")
    elo = pd.read_csv(DATA_DIR / "elo_snapshots.csv")
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    lead_in = pd.read_csv(DATA_DIR / "team_results_lead_in.csv")
    manifest = pd.read_json(DATA_DIR / "manifest.json", typ="series").to_dict()

    text_columns = ["team", "canonical_name", "tournament_name"]
    for frame in (teams, groups, fifa, elo, fixtures, lead_in):
        for column in text_columns:
            if column in frame.columns:
                frame[column] = frame[column].map(fix_mojibake)
    groups["team_name"] = groups["team_name"].map(fix_mojibake)
    if "qualified_team_name" in lead_in.columns:
        lead_in["qualified_team_name"] = lead_in["qualified_team_name"].map(fix_mojibake)
    if "opponent_name" in lead_in.columns:
        lead_in["opponent_name"] = lead_in["opponent_name"].map(fix_mojibake)

    latest_fifa = (
        fifa.sort_values(["snapshot_date", "source_as_of"])
        .drop_duplicates(subset=["team_id"], keep="last")
        .loc[:, ["team_id", "rank", "points", "snapshot_date"]]
        .rename(columns={"rank": "world_rank", "points": "fifa_points", "snapshot_date": "fifa_snapshot_date"})
    )
    latest_elo = (
        elo.sort_values(["snapshot_date", "source_as_of"])
        .drop_duplicates(subset=["team_id"], keep="last")
        .loc[:, ["team_id", "elo_rank", "elo_rating", "snapshot_date"]]
        .rename(columns={"snapshot_date": "elo_snapshot_date"})
    )

    team_columns = [
        "team_id",
        "team",
        "tournament_name",
        "canonical_name",
        "flag_icon_code",
        "group_code",
        "confederation",
        "is_host",
        "world_cup_participations",
        "weighted_world_cup_participations",
        "weighted_world_cup_placement_score",
    ]
    available_team_columns = [column_name for column_name in team_columns if column_name in teams.columns]

    merged = (
        groups.merge(
            teams.loc[:, available_team_columns],
            on=["team_id", "group_code"],
            how="left",
        )
        .merge(latest_fifa, on="team_id", how="left")
        .merge(latest_elo, on="team_id", how="left")
    )

    display_name_source = (
        merged["team"]
        if "team" in merged.columns
        else pd.Series(pd.NA, index=merged.index, dtype="object")
    )
    merged["display_name"] = (
        display_name_source.fillna(merged.get("tournament_name")).fillna(merged["team_name"]).map(fix_mojibake)
    )
    merged["world_rank"] = pd.to_numeric(merged["world_rank"], errors="coerce")
    merged["fifa_points"] = pd.to_numeric(merged["fifa_points"], errors="coerce")
    merged["elo_rating"] = pd.to_numeric(merged["elo_rating"], errors="coerce")
    merged["elo_rank"] = pd.to_numeric(merged["elo_rank"], errors="coerce")
    if "world_cup_participations" in merged.columns:
        merged["world_cup_participations"] = pd.to_numeric(merged["world_cup_participations"], errors="coerce")
    if "weighted_world_cup_participations" in merged.columns:
        merged["weighted_world_cup_participations"] = pd.to_numeric(
            merged["weighted_world_cup_participations"],
            errors="coerce",
        )
    if "weighted_world_cup_placement_score" in merged.columns:
        merged["weighted_world_cup_placement_score"] = pd.to_numeric(
            merged["weighted_world_cup_placement_score"],
            errors="coerce",
        )

    metadata = {
        "build_date": manifest.get("build_date", ""),
        "fifa_snapshot_date": latest_fifa["fifa_snapshot_date"].dropna().max(),
        "elo_snapshot_date": latest_elo["elo_snapshot_date"].dropna().max(),
    }
    return merged, fixtures, lead_in, metadata
