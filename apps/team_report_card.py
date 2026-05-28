from __future__ import annotations

import html
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from apps import home
from apps.dashboard.model_registry import PRIMARY_MODEL
from apps.dashboard.simulation_store import (
    DEFAULT_SIMULATION_SEED,
    ArtifactSettings,
    load_official_artifact,
)
from world_cup_sim.constants import (
    WEIGHTED_FORM_ELO_BOUNDS,
    WEIGHTED_FORM_GD_BOUNDS,
    WEIGHTED_FORM_PERF_BOUNDS,
)
from world_cup_sim.shared import (
    clip_scale,
    compute_elo_expected_score,
    normalize_historical_team_name,
    normalize_weighted_form_result,
)
from world_cup_sim.analysis import add_era_column
import world_cup_simulation as simulation


ROOT = Path(__file__).resolve().parents[1]
WORLD_CUP_ROOT = simulation.WORLD_CUP_ROOT
SUBJECT_ORDER = (
    "Overall Strength",
    "Attack",
    "Defense",
    "Recent Form",
    "World Cup History",
    "Tournament Outlook",
)
PENDING_SUBJECTS = ("Squad Quality", "Qualification Strength")
QUALIFICATION_CYCLE_START = pd.Timestamp("2022-12-19")
QUALIFICATION_PLAYOFF_START = pd.Timestamp("2026-03-26")
QUALIFICATION_PLAYOFF_END = pd.Timestamp("2026-03-31")
GRADE_BANDS = (
    (9.5, "A+"),
    (8.8, "A"),
    (7.5, "B"),
    (6.0, "C"),
    (4.5, "D"),
    (-float("inf"), "F"),
)
SUBJECT_NOTE_TEMPLATES = {
    "Overall Strength": (
        "Outstanding all-round outlook compared with the field.",
        "Strong overall profile with only minor soft spots.",
        "Solid baseline, but not yet among the elite favorites.",
        "Mixed profile with clear volatility in the model.",
        "Undersized overall profile for a deep run.",
    ),
    "Attack": (
        "Creates one of the sharpest attacking profiles in the tournament.",
        "Carries reliable scoring upside against most opponents.",
        "Can score, but not consistently enough to dominate.",
        "Needs favorable game states to create clear attacking value.",
        "Attack is a major limiter in the current model.",
    ),
    "Defense": (
        "Defensive profile is among the most reliable in the field.",
        "Usually suppresses chances well enough to stay in control.",
        "Functional defense, though not especially dominant.",
        "Can be exposed when the match tempo rises.",
        "Defensive resilience is a major concern.",
    ),
    "Recent Form": (
        "Lead-in form is excellent and supports the projection.",
        "Recent results meaningfully strengthen the baseline.",
        "Recent form is acceptable but not carrying the team.",
        "Lead-in form is uneven and adds some downside.",
        "Recent form is working against the projection.",
    ),
    "World Cup History": (
        "Tournament pedigree is elite by any historical standard.",
        "History adds meaningful credibility to the forecast.",
        "History is respectable without being a major edge.",
        "Limited recent pedigree reduces the margin for error.",
        "World Cup history offers little support here.",
    ),
    "Tournament Outlook": (
        "Model gives this team a genuine title-level outlook.",
        "Knockout advancement chances are consistently strong.",
        "Projection points to a plausible but imperfect run.",
        "Needs several outcomes to break correctly for a deep run.",
        "Current tournament path is difficult in the model.",
    ),
}
DRIVER_LABELS = {
    "elo_rating": "Elite Elo base",
    "results_form": "Strong recent results",
    "gd_form": "Healthy recent goal difference",
    "placement_metric": "Strong World Cup history",
    "goals_for": "Reliable attacking output",
    "host_flag": "Host advantage",
}
CHART_BACKGROUND = "#EFE3CF"
CHART_TEXT_COLOR = "#3A2A1A"
CHART_AXIS_COLOR = "#5A4632"
CHART_GRID_COLOR = "#D8C8AF"
CHART_FONT_FAMILY = "Gill Sans, sans-serif"
CHART_POSITIVE_COLOR = "#2F6F3E"
CHART_NEGATIVE_COLOR = "#B23A30"
CHART_ACCENT_COLOR = "#7A4E2D"
CHART_SECONDARY_COLOR = "#2F6F73"
QUALIFICATION_STAGE_COLORS = {"Qualifiers": CHART_SECONDARY_COLOR, "Playoffs": "#C99700"}
QUALIFICATION_STAGE_DISPLAY_LABELS = {"Qualifiers": "Qualifiers", "Playoffs": "Qualifier playoffs"}
MATCH_TYPE_COLORS = {
    "Friendlies": "#8B7355",
    "World Cup qualifiers": CHART_SECONDARY_COLOR,
    "Qualifier playoffs": "#C99700",
    "World Cup finals": CHART_ACCENT_COLOR,
    "Continental finals": "#8A3FFC",
    "Continental qualifiers": "#D1495B",
    "Nations League": "#2F6F3E",
    "Other tournaments": CHART_AXIS_COLOR,
}
MATCH_TYPE_DISPLAY_LABELS = {
    "Friendlies": "Friendlies",
    "World Cup qualifiers": "WC qualifiers",
    "Qualifier playoffs": "Playoffs",
    "World Cup finals": "WC finals",
    "Continental finals": "Continental finals",
    "Continental qualifiers": "Continental qualifiers",
    "Nations League": "Nations League",
    "Other tournaments": "Other",
}
SOURCE_NOTE = "Data Source: Kaggle | @cartierkut1"
ERA_COLORS = {
    "Early Era": "#7A4E2D",
    "Golden Age": "#C99700",
    "Modern Era": "#2F6F73",
    "Contemporary": "#8A3FFC",
    "Recent": "#D1495B",
}
PLACEMENT_SHORT_LABELS = {
    "Winner": "W",
    "Runner-up": "F",
    "Third Place": "3P",
    "Fourth Place": "4P",
    "Semi-final": "SF",
    "Quarter-final": "QF",
    "Round of 16": "R16",
    "Group Stage": "GS",
    "DNQ": "DNQ",
}
PLOTLY_EXPORT_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 600,
        "width": 1000,
        "scale": 3,
    }
}


def format_artifact_updated_at(value: str | None) -> str:
    """Return a compact report-card artifact timestamp label."""
    if not value:
        return "last updated time unavailable"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "last updated time unavailable"
    period_label = parsed.strftime("%p").lower()
    return f"last updated on {parsed:%Y-%m-%d} @ {parsed:%I:%M}{period_label}"


def score_to_grade(score: float) -> str:
    """Map a 1-10 score to the fixed report-card grade bands."""
    numeric = float(score)
    for minimum, grade in GRADE_BANDS:
        if numeric >= minimum:
            return grade
    return "F"


def score_to_verdict(score: float) -> str:
    """Translate a report-card score into one short outlook verdict."""
    numeric = float(score)
    if numeric >= 9.0:
        return "Contender"
    if numeric >= 8.0:
        return "Strong knockout candidate"
    if numeric >= 7.0:
        return "Dangerous outsider"
    if numeric >= 6.0:
        return "Competitive but vulnerable"
    return "Likely group-stage struggler"


def choose_column(df: pd.DataFrame, candidates: Iterable[str], fallback: str = "") -> str:
    """Return the first available column name from a list of candidates."""
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    return fallback


def build_display_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Return a simple team-id to display-name lookup."""
    return (
        df.loc[:, ["team_id", "display_name"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .assign(team_id=lambda frame: frame["team_id"].astype(str), display_name=lambda frame: frame["display_name"].astype(str))
        .set_index("team_id")["display_name"]
        .to_dict()
    )


def build_flag_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Return a simple team-id to flag-icon-code lookup."""
    return (
        df.loc[:, ["team_id", "flag_icon_code"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .assign(team_id=lambda frame: frame["team_id"].astype(str), flag_icon_code=lambda frame: frame["flag_icon_code"].fillna("").astype(str))
        .set_index("team_id")["flag_icon_code"]
        .to_dict()
    )


def series_to_report_scores(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Scale a metric into deterministic 1-10 report-card scores via rank percentiles."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.full(len(series), 1.0), index=series.index, dtype=float)
    if len(valid) == 1:
        scores = pd.Series(np.full(len(series), 10.0), index=series.index, dtype=float)
        scores.loc[numeric.isna()] = 1.0
        return scores

    ranking_source = -valid if reverse else valid
    rank = ranking_source.rank(method="average", ascending=True)
    scaled = 1.0 + 9.0 * ((rank - 1.0) / (len(valid) - 1.0))
    scores = pd.Series(1.0, index=series.index, dtype=float)
    scores.loc[valid.index] = scaled.astype(float)
    return scores.clip(lower=1.0, upper=10.0).round(1)


def describe_subject_score(subject: str, score: float) -> str:
    """Return a short teacher-style note for one subject score."""
    elite, strong, solid, mixed, weak = SUBJECT_NOTE_TEMPLATES[subject]
    if score >= 9.0:
        return elite
    if score >= 8.0:
        return strong
    if score >= 6.5:
        return solid
    if score >= 5.0:
        return mixed
    return weak


def add_report_card_metrics(dashboard_df: pd.DataFrame) -> pd.DataFrame:
    """Add the raw and scored report-card metrics to the V3 simulation table."""
    df = dashboard_df.copy()
    appearance_max = max(float(pd.to_numeric(df["appearance"], errors="coerce").fillna(0.0).max()), 1.0)
    df["appearance_norm"] = pd.to_numeric(df["appearance"], errors="coerce").fillna(0.0) / appearance_max
    df["recent_form_metric"] = (
        0.5 * pd.to_numeric(df["results_form"], errors="coerce").fillna(0.0)
        + 0.3 * pd.to_numeric(df["gd_form"], errors="coerce").fillna(0.0)
        + 0.2 * pd.to_numeric(df["perf_vs_exp"], errors="coerce").fillna(0.0)
    )
    df["history_metric"] = (
        0.7 * pd.to_numeric(df["placement"], errors="coerce").fillna(0.0)
        + 0.3 * pd.to_numeric(df["appearance_norm"], errors="coerce").fillna(0.0)
    )
    df["outlook_metric"] = (
        0.10 * pd.to_numeric(df["ko_prob"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(df["r16_prob"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["qf_prob"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["sf_prob"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["final_prob"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(df["champion_prob"], errors="coerce").fillna(0.0)
    )

    raw_subject_metrics = {
        "Overall Strength": pd.to_numeric(df["team_strength"], errors="coerce").fillna(0.0),
        "Attack": pd.to_numeric(df["goals_for"], errors="coerce").fillna(0.0),
        "Defense": pd.to_numeric(df["goals_against"], errors="coerce").fillna(0.0),
        "Recent Form": pd.to_numeric(df["recent_form_metric"], errors="coerce").fillna(0.0),
        "World Cup History": pd.to_numeric(df["history_metric"], errors="coerce").fillna(0.0),
        "Tournament Outlook": pd.to_numeric(df["outlook_metric"], errors="coerce").fillna(0.0),
    }
    reverse_subjects = {"Defense"}
    for subject, metric_series in raw_subject_metrics.items():
        score_column = subject.lower().replace(" ", "_").replace("-", "_") + "_score"
        df[score_column] = series_to_report_scores(metric_series, reverse=subject in reverse_subjects)

    overall_columns = [subject.lower().replace(" ", "_").replace("-", "_") + "_score" for subject in SUBJECT_ORDER]
    df["overall_report_score"] = df.loc[:, overall_columns].mean(axis=1).round(1)
    df["overall_grade"] = df["overall_report_score"].map(score_to_grade)
    df["overall_verdict"] = df["overall_report_score"].map(score_to_verdict)
    return df


def build_subject_rows(team_row: pd.Series) -> list[dict[str, str | float]]:
    """Build the subject-score rows shown in the report card."""
    rows: list[dict[str, str | float]] = []
    for subject in SUBJECT_ORDER:
        score_column = subject.lower().replace(" ", "_").replace("-", "_") + "_score"
        score = float(team_row[score_column])
        rows.append(
            {
                "subject": subject,
                "score": round(score, 1),
                "grade": score_to_grade(score),
                "note": describe_subject_score(subject, score),
            }
        )
    return rows


def build_pending_subject_rows() -> list[dict[str, str]]:
    """Return the unsupported subject rows for the MVP."""
    return [{"subject": subject, "value": "Pending data"} for subject in PENDING_SUBJECTS]


def is_debut_tournament(team_row: pd.Series) -> bool:
    """Return whether the selected team is making its first World Cup appearance."""
    appearances_value = pd.to_numeric(team_row.get("world_cup_participations", np.nan), errors="coerce")
    return bool(pd.notna(appearances_value) and int(appearances_value) == 1)


@st.cache_data(show_spinner=False)
def load_squad_identity_lookup() -> dict[str, dict[str, str]]:
    """Load 2026 squad coach and captain facts for report-card identity KPIs."""
    status_path = WORLD_CUP_ROOT / "2026" / "squads_teams_status.csv"
    squads_path = WORLD_CUP_ROOT / "2026" / "squads.csv"
    if not status_path.exists() or not squads_path.exists():
        return {}

    status = pd.read_csv(status_path).fillna("")
    squads = pd.read_csv(squads_path).fillna("")
    captains = squads.loc[squads.get("is_captain", "").astype(str).str.upper().eq("TRUE")].copy()
    captain_lookup = (
        captains.drop_duplicates(subset=["team_id"], keep="first")
        .assign(team_id=lambda frame: frame["team_id"].astype(str), player_name=lambda frame: frame["player_name"].astype(str).map(home.fix_mojibake))
        .set_index("team_id")["player_name"]
        .to_dict()
        if not captains.empty and {"team_id", "player_name"}.issubset(captains.columns)
        else {}
    )

    lookup: dict[str, dict[str, str]] = {}
    for row in status.itertuples(index=False):
        team_id = str(getattr(row, "team_id", ""))
        if not team_id:
            continue
        row_count = str(getattr(row, "row_count", ""))
        is_final = str(getattr(row, "is_final_squad", "")).upper() == "TRUE"
        if row_count in {"", "0"}:
            squad_status = "pending_no_table"
        elif is_final:
            squad_status = "final"
        else:
            squad_status = "preliminary"
        lookup[team_id] = {
            "coach": home.fix_mojibake(str(getattr(row, "coach", ""))).strip(),
            "captain": captain_lookup.get(team_id, "").strip(),
            "squad_status": squad_status,
            "row_count": row_count,
            "source_as_of": str(getattr(row, "source_as_of", "")),
        }
    return lookup


def pending_if_blank(value: object) -> str:
    """Return a display value or the report-card pending placeholder."""
    if value is None or pd.isna(value):
        return "Pending data"
    text = str(value).strip()
    return text if text else "Pending data"


def build_identity_rows(team_row: pd.Series, best_finish: str, squad_identity: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Return the key identity facts for the selected team."""
    squad_identity = squad_identity or {}
    appearances_value = team_row.get("world_cup_participations", "")
    if pd.isna(appearances_value) or appearances_value == "":
        appearances_value = ""
    else:
        appearances_value = f"{int(float(appearances_value))}"
    best_finish_value = "Debut tournament" if is_debut_tournament(team_row) and best_finish == "No appearances" else best_finish

    rows = [
        {"label": "Confederation", "value": str(team_row.get("confederation", ""))},
        {"label": "Group", "value": f"Group {team_row.get('group_code', '')}"},
        {"label": "FIFA Rank", "value": f"{int(float(team_row['world_rank']))}" if pd.notna(team_row.get("world_rank")) else "N/A"},
        {"label": "Elo Rating", "value": f"{int(round(float(team_row['elo_rating'])))}" if pd.notna(team_row.get("elo_rating")) else "N/A"},
        {"label": "World Cup Appearances", "value": str(appearances_value or "N/A")},
        {"label": "Best Finish", "value": best_finish_value},
    ]
    rows.extend(
        [
            {"label": "Coach", "value": pending_if_blank(squad_identity.get("coach", ""))},
            {"label": "Captain", "value": pending_if_blank(squad_identity.get("captain", ""))},
        ]
    )
    return rows


def normalize_team_best_finish(placement_df: pd.DataFrame, team_names: Iterable[str]) -> str:
    """Resolve the best historical finish for a current team across name aliases."""
    normalized_names = {
        normalize_historical_team_name(name)
        for name in team_names
        if isinstance(name, str) and name.strip()
    }
    if not normalized_names:
        return "No appearances"
    team_rows = placement_df[placement_df["team_key"].isin(normalized_names)].copy()
    if team_rows.empty:
        return "No appearances"
    best_row = team_rows.sort_values(["position", "edition"], ascending=[True, True], kind="stable").iloc[0]
    return str(best_row["placement"])


def build_best_finish_lookup(base_df: pd.DataFrame) -> dict[str, str]:
    """Build a team-id to best-finish lookup from historical placement data."""
    placement_df = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "placement.csv")
    placement_df["position"] = pd.to_numeric(placement_df["position"], errors="coerce")
    placement_df["team_key"] = placement_df["country"].map(normalize_historical_team_name)
    canonical_column = choose_column(base_df, ("canonical_name", "canonical_name_y", "canonical_name_x"), fallback="")

    lookup: dict[str, str] = {}
    for row in base_df.drop_duplicates(subset=["team_id"], keep="first").itertuples(index=False):
        team_names = [
            getattr(row, "display_name", ""),
            getattr(row, "tournament_name", ""),
            getattr(row, "team_name", ""),
            getattr(row, canonical_column, "") if canonical_column else "",
        ]
        lookup[str(getattr(row, "team_id"))] = normalize_team_best_finish(placement_df, team_names)
    return lookup


def build_recent_matches_table(lead_in_df: pd.DataFrame, team_id: str, match_window: int = 10) -> pd.DataFrame:
    """Return the selected team's latest matches with report-card performance grades."""
    team_matches = lead_in_df[lead_in_df["qualified_team_id"].astype(str) == str(team_id)].copy()
    if team_matches.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Opponent",
                "Competition",
                "Result",
                "Score",
                "Elo Change",
                "Performance Score",
                "Grade",
            ]
        )

    team_matches["date"] = pd.to_datetime(team_matches["date"], errors="coerce")
    for column_name in ("team_score", "opponent_score", "team_elo_start", "opponent_elo_start", "team_elo_delta"):
        team_matches[column_name] = pd.to_numeric(team_matches[column_name], errors="coerce")
    team_matches = team_matches.sort_values(["date", "lead_in_id"], kind="stable").tail(match_window).copy()
    team_matches["normalized_result"] = normalize_weighted_form_result(
        team_matches["result"],
        team_matches["team_score"],
        team_matches["opponent_score"],
    )
    team_matches["actual_score"] = team_matches["normalized_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0}).astype(float)
    team_matches["goal_difference"] = (team_matches["team_score"] - team_matches["opponent_score"]).fillna(0.0)
    team_matches["expected_score"] = compute_elo_expected_score(
        team_matches["team_elo_start"],
        team_matches["opponent_elo_start"],
    ).astype(float)
    team_matches["perf_vs_exp"] = team_matches["actual_score"] - team_matches["expected_score"]
    team_matches["gd_score"] = clip_scale(team_matches["goal_difference"].clip(lower=-4.0, upper=4.0), *WEIGHTED_FORM_GD_BOUNDS)
    team_matches["perf_score"] = clip_scale(team_matches["perf_vs_exp"], *WEIGHTED_FORM_PERF_BOUNDS)
    team_matches["elo_score"] = clip_scale(team_matches["team_elo_delta"], *WEIGHTED_FORM_ELO_BOUNDS)
    team_matches["performance_index"] = (
        0.4 * team_matches["actual_score"]
        + 0.25 * team_matches["gd_score"]
        + 0.25 * team_matches["perf_score"]
        + 0.10 * team_matches["elo_score"]
    )
    team_matches["Performance Score"] = (1.0 + 9.0 * team_matches["performance_index"]).round(1)
    team_matches["Grade"] = team_matches["Performance Score"].map(score_to_grade)
    team_matches["Date"] = team_matches["date"].dt.strftime("%Y-%m-%d")
    team_matches["Opponent"] = team_matches["opponent_name"].fillna("").astype(str)
    team_matches["Competition"] = team_matches["tournament"].fillna("").astype(str)
    team_matches["Result"] = team_matches["normalized_result"].map({"win": "W", "draw": "D", "loss": "L"}).fillna("")
    team_matches["Score"] = (
        team_matches["team_score"].fillna(0).astype(int).astype(str)
        + "-"
        + team_matches["opponent_score"].fillna(0).astype(int).astype(str)
    )
    team_matches["Elo Change"] = team_matches["team_elo_delta"].map(lambda value: f"{float(value):+0.1f}")
    team_matches["post_match_elo"] = team_matches["team_elo_start"].fillna(0.0) + team_matches["team_elo_delta"].fillna(0.0)
    return team_matches.sort_values(["date", "lead_in_id"], ascending=[False, False], kind="stable").reset_index(drop=True)


def is_world_cup_qualification_row(tournament: object) -> bool:
    """Return whether a tournament label is part of World Cup qualification."""
    label = str(tournament or "").strip().lower()
    if not label:
        return False
    if "world cup qualification" in label:
        return True
    if "world cup" in label and ("playoff" in label or "play-off" in label or "inter-confederation" in label):
        return True
    return False


def qualification_stage_label(tournament: object, match_date: object | None = None) -> str:
    date_value = pd.to_datetime(match_date, errors="coerce")
    if pd.notna(date_value) and QUALIFICATION_PLAYOFF_START <= date_value <= QUALIFICATION_PLAYOFF_END:
        return "Playoffs"

    label = str(tournament or "").strip().lower()
    return "Playoffs" if "playoff" in label or "play-off" in label or "inter-confederation" in label else "Qualifiers"


def match_type_label(tournament: object, match_date: object | None = None) -> str:
    """Return the report-card match category used for Road Here background bands."""
    label = (
        unicodedata.normalize("NFKD", str(tournament or "").strip())
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    date_value = pd.to_datetime(match_date, errors="coerce")
    is_playoff_window = pd.notna(date_value) and QUALIFICATION_PLAYOFF_START <= date_value <= QUALIFICATION_PLAYOFF_END

    if (
        "inter-confederation" in label
        or ("world cup" in label and ("playoff" in label or "play-off" in label))
        or ("fifa world cup qualification" in label and is_playoff_window)
    ):
        return "Qualifier playoffs"
    if "fifa world cup qualification" in label:
        return "World Cup qualifiers"
    if "friendly" in label:
        return "Friendlies"
    if "nations league" in label:
        return "Nations League"
    if "qualification" in label or "qualifier" in label:
        return "Continental qualifiers"
    if label == "fifa world cup":
        return "World Cup finals"

    continental_final_markers = (
        "african cup of nations",
        "afc asian cup",
        "copa america",
        "gold cup",
        "uefa euro",
        "arab cup",
        "oceania nations cup",
        "concacaf championship",
        "copa centroamericana",
    )
    if any(marker in label for marker in continental_final_markers):
        return "Continental finals"
    return "Other tournaments"


def build_qualification_path_table(lead_in_df: pd.DataFrame, team_id: str) -> pd.DataFrame:
    """Return the selected team's 2026 qualification-cycle path."""
    output_columns = [
        "lead_in_id",
        "date",
        "Date",
        "Opponent",
        "Competition",
        "Venue",
        "qualification_stage",
        "Result",
        "Score",
        "points",
        "cumulative_points",
        "goal_difference",
        "cumulative_goal_difference",
        "team_score",
        "opponent_score",
        "opponent_elo_start",
        "team_elo_delta",
        "post_match_elo",
        "match_label",
    ]
    if lead_in_df.empty or "qualified_team_id" not in lead_in_df.columns:
        return pd.DataFrame(columns=output_columns)

    df = lead_in_df.loc[lead_in_df["qualified_team_id"].astype(str).eq(str(team_id))].copy()
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    tournament_mask = (
        df["tournament"].map(is_world_cup_qualification_row)
        if "tournament" in df.columns
        else pd.Series(False, index=df.index)
    )
    df = df.loc[tournament_mask & df["date"].ge(QUALIFICATION_CYCLE_START)].copy()
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    for column_name in ("team_score", "opponent_score", "opponent_elo_start", "team_elo_start", "team_elo_delta"):
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        else:
            df[column_name] = np.nan

    df = df.sort_values(["date", "lead_in_id"], kind="stable").copy()
    df["normalized_result"] = normalize_weighted_form_result(
        df["result"],
        df["team_score"],
        df["opponent_score"],
    )
    df["Result"] = df["normalized_result"].map({"win": "W", "draw": "D", "loss": "L"}).fillna("")
    df["points"] = df["normalized_result"].map({"win": 3, "draw": 1, "loss": 0}).fillna(0).astype(int)
    df["goal_difference"] = (df["team_score"] - df["opponent_score"]).fillna(0).astype(float)
    df["cumulative_points"] = df["points"].cumsum()
    df["cumulative_goal_difference"] = df["goal_difference"].cumsum()
    df["post_match_elo"] = df["team_elo_start"].fillna(0.0) + df["team_elo_delta"].fillna(0.0)
    df["Date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["Opponent"] = df["opponent_name"].fillna("").astype(str) if "opponent_name" in df.columns else ""
    df["Competition"] = df["tournament"].fillna("").astype(str) if "tournament" in df.columns else ""
    df["qualification_stage"] = [
        qualification_stage_label(tournament, match_date)
        for tournament, match_date in zip(df["Competition"], df["date"], strict=False)
    ]
    df["Venue"] = (
        df["city"].fillna("").astype(str).str.strip()
        + np.where(df.get("country", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().ne(""), ", ", "")
        + df.get("country", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    ).str.strip(", ")
    df["Score"] = (
        df["team_score"].fillna(0).astype(int).astype(str)
        + "-"
        + df["opponent_score"].fillna(0).astype(int).astype(str)
    )
    df["match_label"] = df["Date"] + " vs " + df["Opponent"]
    return df.loc[:, output_columns].reset_index(drop=True)


def build_group_fixtures_table(fixtures_df: pd.DataFrame, team_id: str, display_lookup: dict[str, str]) -> pd.DataFrame:
    """Return the selected team's upcoming group-stage fixtures."""
    df = fixtures_df.copy()
    df["match_number"] = pd.to_numeric(df["match_number"], errors="coerce")
    df["kickoff_datetime_utc"] = pd.to_datetime(df["kickoff_datetime_utc"], errors="coerce", utc=True)
    team_fixtures = df[
        (df["round_code"] == "GS")
        & (
            df["home_team_id"].astype(str).eq(str(team_id))
            | df["away_team_id"].astype(str).eq(str(team_id))
        )
    ].copy()
    if team_fixtures.empty:
        return pd.DataFrame(columns=["Date", "Opponent", "Stage", "Venue"])

    team_fixtures["Date"] = team_fixtures["kickoff_datetime_utc"].dt.strftime("%Y-%m-%d")
    team_fixtures["Opponent"] = np.where(
        team_fixtures["home_team_id"].astype(str).eq(str(team_id)),
        team_fixtures["away_team_id"].map(display_lookup).fillna(team_fixtures["away_tournament_name"]),
        team_fixtures["home_team_id"].map(display_lookup).fillna(team_fixtures["home_tournament_name"]),
    )
    team_fixtures["Stage"] = "Group Stage"
    team_fixtures["Venue"] = team_fixtures["venue_name"].fillna("").astype(str)
    return team_fixtures.sort_values(["kickoff_datetime_utc", "match_number"], kind="stable").loc[:, ["Date", "Opponent", "Stage", "Venue"]].reset_index(drop=True)


def build_knockout_path_table(bracket_data: dict[str, Any], team_id: str, display_lookup: dict[str, str]) -> pd.DataFrame:
    """Return the selected team's projected knockout path from the deterministic bracket."""
    rows: list[dict[str, str | float]] = []
    for round_data in bracket_data.get("rounds", []):
        for match in round_data.get("matches", []):
            home_team_id = str(match.get("home_team_id", ""))
            away_team_id = str(match.get("away_team_id", ""))
            if str(team_id) not in {home_team_id, away_team_id}:
                continue
            is_home = str(team_id) == home_team_id
            opponent_id = away_team_id if is_home else home_team_id
            matchup_win_prob = float(match.get("home_win_prob" if is_home else "away_win_prob", 0.0))
            rows.append(
                {
                    "Stage": str(match.get("round_label", match.get("round_code", ""))),
                    "Opponent": display_lookup.get(opponent_id, opponent_id),
                    "Matchup Win %": round(matchup_win_prob, 1),
                    "Projected Winner": display_lookup.get(str(match.get("winner_team_id", "")), str(match.get("winner_team_id", ""))),
                }
            )
            break
    return pd.DataFrame(rows)


def build_probability_tables(team_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact probability tables for group finish and knockout stages."""
    group_finish = pd.DataFrame(
        [
            {"Finish": "1st", "Probability": float(team_row.get("prob_1", 0.0))},
            {"Finish": "2nd", "Probability": float(team_row.get("prob_2", 0.0))},
            {"Finish": "3rd", "Probability": float(team_row.get("prob_3", 0.0))},
            {"Finish": "4th", "Probability": float(team_row.get("prob_4", 0.0))},
        ]
    )
    stage_progression = pd.DataFrame(
        [
            {"Stage": "Round of 32", "Probability": float(team_row.get("ko_prob", 0.0))},
            {"Stage": "Round of 16", "Probability": float(team_row.get("r16_prob", 0.0))},
            {"Stage": "Quarter-finals", "Probability": float(team_row.get("qf_prob", 0.0))},
            {"Stage": "Semi-finals", "Probability": float(team_row.get("sf_prob", 0.0))},
            {"Stage": "Final", "Probability": float(team_row.get("final_prob", 0.0))},
            {"Stage": "Winner", "Probability": float(team_row.get("champion_prob", 0.0))},
        ]
    )
    return group_finish, stage_progression


def build_model_reason_bullets(team_row: pd.Series, full_df: pd.DataFrame) -> list[str]:
    """Return a short set of model-friendly reasons for the selected team."""
    metric_frame = pd.DataFrame(
        {
            "team_id": full_df["team_id"].astype(str),
            "elo_rating": pd.to_numeric(full_df["elo_rating"], errors="coerce").fillna(0.0),
            "results_form": pd.to_numeric(full_df["results_form"], errors="coerce").fillna(0.0),
            "gd_form": pd.to_numeric(full_df["gd_form"], errors="coerce").fillna(0.0),
            "placement_metric": pd.to_numeric(full_df["history_metric"], errors="coerce").fillna(0.0),
            "goals_for": pd.to_numeric(full_df["goals_for"], errors="coerce").fillna(0.0),
            "host_flag": pd.to_numeric(full_df["host_flag"], errors="coerce").fillna(0.0),
        }
    ).drop_duplicates(subset=["team_id"], keep="first").set_index("team_id")
    ranked_rows: list[tuple[float, str]] = []
    team_index = str(team_row.get("team_id", team_row.name))
    if team_index not in metric_frame.index:
        return []
    for column_name, label in DRIVER_LABELS.items():
        if column_name == "host_flag" and float(team_row.get("host_flag", 0.0)) <= 0:
            continue
        score = float(series_to_report_scores(metric_frame[column_name]).loc[team_index])
        ranked_rows.append((score, label))
    ranked_rows.sort(reverse=True)
    return [label for _, label in ranked_rows[:3]]


def official_report_card_settings(metadata: dict[str, Any]) -> tuple[ArtifactSettings, int, int]:
    """Return the fixed official projection settings used by the report card."""
    default_settings = home.default_simulation_settings()
    simulation_label = str(default_settings["simulation_label"])
    simulations = int(home.SIMULATION_OPTIONS[simulation_label])
    match_window = int(default_settings["form_match_window"])
    artifact_settings = ArtifactSettings(
        model_id=PRIMARY_MODEL.model_id,
        model_version=PRIMARY_MODEL.model_version,
        data_build_date=str(metadata.get("build_date", "")),
        simulations=simulations,
        match_window=match_window,
        training_scope=PRIMARY_MODEL.default_training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=home.BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    return artifact_settings, simulations, match_window


@st.cache_data(show_spinner=False)
def build_report_card_dataset() -> dict[str, Any]:
    """Build and cache the shared primary-model report-card dataset from the official artifact."""
    base_df, fixtures_df, lead_in_df, metadata = home.load_data()
    base_team_lookup = (
        base_df.drop_duplicates(subset=["team_id"], keep="first")
        .set_index("team_id")
        .to_dict("index")
    )
    artifact_settings, simulations, match_window = official_report_card_settings(metadata)
    load_result = load_official_artifact(artifact_settings)
    if load_result.artifact is None:
        raise RuntimeError(
            "The official primary report-card simulation artifact is unavailable. "
            "Run the dashboard simulation prewarm for the primary V4 model."
        )
    artifact = load_result.artifact
    dashboard_df = artifact.dashboard_df
    dashboard_df = home.ensure_dashboard_probability_columns(dashboard_df)
    dashboard_df = add_report_card_metrics(dashboard_df)
    return {
        "base_df": base_df,
        "fixtures_df": fixtures_df,
        "lead_in_df": lead_in_df,
        "metadata": metadata,
        "dashboard_df": dashboard_df,
        "bracket_data": artifact.bracket_data,
        "artifact_metadata": artifact.metadata,
        "artifact_source": artifact.source,
        "artifact_created": load_result.created,
        "artifact_created_at_utc": artifact.created_at_utc,
        "artifact_warnings": load_result.warnings,
        "official_simulations": simulations,
        "official_match_window": match_window,
        "display_lookup": build_display_lookup(base_df),
        "flag_lookup": build_flag_lookup(base_df),
        "best_finish_lookup": build_best_finish_lookup(base_df),
        "base_team_lookup": base_team_lookup,
        "squad_identity_lookup": load_squad_identity_lookup(),
    }


def select_report_card_context(dataset: dict[str, Any], team_id: str, recent_match_count: int = 10) -> dict[str, Any]:
    """Filter the shared dataset into one selected-team context payload."""
    dashboard_df = dataset["dashboard_df"]
    team_row = dashboard_df.loc[dashboard_df["team_id"].astype(str) == str(team_id)].copy()
    if team_row.empty:
        raise ValueError(f"Unknown team_id: {team_id}")
    team_row = team_row.iloc[0].copy()
    team_row.name = str(team_row["team_id"])
    base_snapshot = dataset["base_team_lookup"].get(str(team_id), {})
    for column_name, value in base_snapshot.items():
        if column_name not in team_row.index or pd.isna(team_row[column_name]):
            team_row[column_name] = value

    recent_matches = build_recent_matches_table(dataset["lead_in_df"], str(team_id), match_window=recent_match_count)
    qualification_path = build_qualification_path_table(dataset["lead_in_df"], str(team_id))
    group_fixtures = build_group_fixtures_table(dataset["fixtures_df"], str(team_id), dataset["display_lookup"])
    knockout_path = build_knockout_path_table(dataset["bracket_data"], str(team_id), dataset["display_lookup"])
    group_finish_table, stage_probability_table = build_probability_tables(team_row)
    subject_rows = build_subject_rows(team_row)
    overall_summary = {
        "score": float(team_row["overall_report_score"]),
        "grade": str(team_row["overall_grade"]),
        "verdict": str(team_row["overall_verdict"]),
    }
    strongest = max(subject_rows, key=lambda row: float(row["score"]))
    weakest = min(subject_rows, key=lambda row: float(row["score"]))
    overall_summary["summary"] = f"{strongest['subject']} leads this profile, while {weakest['subject']} is the main pressure point."
    return {
        "team_row": team_row,
        "identity_rows": build_identity_rows(
            team_row,
            dataset["best_finish_lookup"].get(str(team_id), "No appearances"),
            dataset["squad_identity_lookup"].get(str(team_id), {}),
        ),
        "subject_rows": subject_rows,
        "pending_subject_rows": build_pending_subject_rows(),
        "recent_matches": recent_matches,
        "qualification_path": qualification_path,
        "group_fixtures": group_fixtures,
        "knockout_path": knockout_path,
        "first_knockout_match": knockout_path.iloc[0].to_dict() if not knockout_path.empty else None,
        "group_finish_table": group_finish_table,
        "stage_probability_table": stage_probability_table,
        "overall_summary": overall_summary,
        "model_reason_bullets": build_model_reason_bullets(team_row, dashboard_df),
        "display_lookup": dataset["display_lookup"],
        "flag_lookup": dataset["flag_lookup"],
        "metadata": dataset["metadata"],
        "simulation_count": len(dashboard_df),
    }


def report_card_css() -> str:
    """Return custom CSS used by the team report-card page."""
    return """
    :root {
        --trc-bg: #EFE3CF;
        --trc-surface: #F6EBD8;
        --trc-surface-strong: #E8D5B8;
        --trc-text: #3A2A1A;
        --trc-muted: #5A4632;
        --trc-line: #D8C8AF;
    }
    .stApp {
        background: var(--trc-bg);
        color: var(--trc-text);
        font-family: Gill Sans, Inter, sans-serif;
    }
    .block-container {
        background: var(--trc-bg);
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--trc-line);
        border-radius: 8px;
        background: var(--trc-surface);
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.05);
    }
    [data-testid="stExpander"] details {
        border: none;
    }
    [data-testid="stExpander"] summary {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: var(--trc-surface-strong);
        color: var(--trc-text);
        border-radius: 8px 8px 0 0;
        font-family: Gill Sans, Inter, sans-serif;
        font-weight: 800;
        min-height: 2.65rem;
        padding: 0.7rem 1rem;
    }
    [data-testid="stExpander"] summary svg {
        color: var(--trc-muted);
        flex: 0 0 auto;
    }
    [data-testid="stExpander"] summary p {
        margin: 0;
        color: var(--trc-text);
        font-family: Gill Sans, Inter, sans-serif;
        font-weight: 800;
        line-height: 1.2;
    }
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] p {
        color: var(--trc-text);
        font-family: Gill Sans, Inter, sans-serif;
    }
    [data-testid="stExpander"] [role="radiogroup"] label {
        color: var(--trc-muted);
        font-weight: 700;
    }
    [data-testid="stExpander"] [role="radiogroup"] label:has(input:checked) {
        color: var(--trc-text);
    }
    [data-testid="stExpander"] [data-baseweb="slider"] div {
        color: var(--trc-text);
    }
    [data-testid="stExpander"] [data-baseweb="select"] > div {
        background: var(--trc-surface-strong);
        border-color: var(--trc-muted);
        color: var(--trc-text);
    }
    .trc-shell {
        display: grid;
        gap: 18px;
        margin-top: 0.5rem;
        color: var(--trc-text);
        font-family: Gill Sans, Inter, sans-serif;
    }
    .trc-hero {
        border: 1px solid var(--trc-line);
        border-radius: 10px;
        padding: 22px 24px;
        background: var(--trc-surface);
        box-shadow: 0 10px 22px rgba(58, 42, 26, 0.08);
    }
    .trc-hero-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        flex-wrap: wrap;
    }
    .trc-title {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .trc-title .fi {
        font-size: 1.8rem;
        border-radius: 999px;
        box-shadow: inset 0 0 0 1px rgba(90, 70, 50, 0.28);
    }
    .trc-title h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
        color: var(--trc-text);
    }
    .trc-subhead {
        margin-top: 0.35rem;
        color: var(--trc-muted);
        font-weight: 600;
    }
    .trc-grade-panel {
        min-width: 220px;
        border: 1px solid var(--trc-muted);
        border-radius: 10px;
        padding: 18px 20px;
        background: var(--trc-surface-strong);
        color: var(--trc-text);
        text-align: center;
    }
    .trc-grade-country {
        margin-bottom: 0.35rem;
        color: var(--trc-text);
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .trc-grade-kicker {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: var(--trc-muted);
        margin-bottom: 0.45rem;
    }
    .trc-grade {
        font-size: 2.9rem;
        font-weight: 900;
        line-height: 1;
        color: var(--trc-text);
    }
    .trc-score {
        margin-top: 0.35rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--trc-text);
    }
    .trc-verdict {
        margin-top: 0.55rem;
        font-size: 0.95rem;
        color: var(--trc-muted);
    }
    .trc-facts {
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }
    .trc-fact {
        border: 1px solid var(--trc-line);
        border-radius: 8px;
        padding: 14px 15px;
        background: rgba(239, 227, 207, 0.72);
    }
    .trc-fact-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.7rem;
        color: var(--trc-muted);
        margin-bottom: 0.35rem;
    }
    .trc-fact-value {
        font-weight: 700;
        color: var(--trc-text);
        line-height: 1.3;
    }
    .trc-subject-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
    }
    .trc-subject-card, .trc-pending-card {
        border: 1px solid var(--trc-line);
        border-radius: 8px;
        background: var(--trc-surface);
        padding: 16px 17px;
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.06);
    }
    .trc-subject-card h3, .trc-pending-card h3 {
        margin: 0;
        font-size: 1rem;
        color: var(--trc-text);
    }
    .trc-grade-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 2.3rem;
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        background: var(--trc-muted);
        color: var(--trc-bg);
        font-size: 0.8rem;
        font-weight: 800;
    }
    .trc-subject-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
    .trc-subject-score {
        margin-top: 0.7rem;
        font-size: 2rem;
        line-height: 1;
        font-weight: 900;
        color: var(--trc-text);
    }
    .trc-subject-note {
        margin-top: 0.7rem;
        color: var(--trc-muted);
        line-height: 1.45;
        font-size: 0.92rem;
    }
    .trc-pending-value {
        margin-top: 0.75rem;
        color: var(--trc-muted);
        font-weight: 700;
    }
    .trc-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 10px 0 18px;
    }
    .trc-kpi {
        border: 1px solid var(--trc-line);
        border-radius: 8px;
        background: var(--trc-surface);
        padding: 12px 14px;
        box-shadow: 0 6px 14px rgba(58, 42, 26, 0.05);
    }
    .trc-kpi-label {
        color: var(--trc-muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        line-height: 1.2;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }
    .trc-kpi-value {
        color: var(--trc-text);
        font-size: 1.85rem;
        font-weight: 900;
        line-height: 1;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--trc-line);
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--trc-muted);
        font-family: Gill Sans, Inter, sans-serif;
        font-weight: 800;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--trc-text);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--trc-muted);
    }
    @media (max-width: 1200px) {
        .trc-facts,
        .trc-subject-grid,
        .trc-kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 700px) {
        .trc-facts,
        .trc-subject-grid,
        .trc-kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    """


def get_query_team_param() -> str | None:
    """Read the active team query parameter from the Streamlit page."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        value = query_params.get("team")
        if isinstance(value, list):
            return str(value[0]) if value else None
        return str(value) if value else None
    getter = getattr(st, "experimental_get_query_params", None)
    if getter is None:
        return None
    values = getter().get("team", [])
    return str(values[0]) if values else None


def set_query_team_param(team_id: str) -> None:
    """Set the active team query parameter on the Streamlit page."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params["team"] = str(team_id)
        return
    setter = getattr(st, "experimental_set_query_params", None)
    if setter is not None:
        setter(team=str(team_id))


def format_percent(value: float) -> str:
    """Format one probability value for display."""
    return f"{float(value):.1f}%"


def chart_title(title: str, country_name: str | None = None) -> str:
    title_text = str(title)
    country_text = str(country_name or "").strip()
    if country_text and title_text.startswith(f"{country_text}'s "):
        title_text = title_text[len(f"{country_text}'s ") :]
    if "FIFA Men's World Cup" not in title_text:
        title_text = f"FIFA Men's World Cup {title_text}"
    if country_text and not title_text.startswith(f"{country_text}'s "):
        title_text = f"{country_text}'s {title_text}"
    return title_text


def apply_report_card_chart_style(
    fig: Any,
    title: str,
    height: int = 360,
    source_note: str | None = None,
    country_name: str | None = None,
) -> Any:
    """Apply the notebook-style historical EDA chart treatment to report-card figures."""
    fig.update_layout(
        height=height,
        title={
            "text": chart_title(title, country_name),
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 18, "color": CHART_TEXT_COLOR},
        },
        font={"family": CHART_FONT_FAMILY, "size": 11, "color": CHART_TEXT_COLOR},
        margin={"l": 40, "r": 40, "t": 70, "b": 72},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        legend={"font": {"color": CHART_TEXT_COLOR}, "title": {"font": {"color": CHART_TEXT_COLOR}}},
        hoverlabel={
            "bgcolor": CHART_BACKGROUND,
            "bordercolor": CHART_AXIS_COLOR,
            "font": {"family": CHART_FONT_FAMILY, "color": CHART_TEXT_COLOR},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=CHART_AXIS_COLOR,
        tickcolor=CHART_AXIS_COLOR,
        tickfont={"size": 13, "color": CHART_AXIS_COLOR},
        title={"font": {"color": CHART_TEXT_COLOR}},
        title_standoff=10,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=CHART_GRID_COLOR,
        zeroline=False,
        linecolor=CHART_AXIS_COLOR,
        tickcolor=CHART_AXIS_COLOR,
        tickfont={"size": 13, "color": CHART_AXIS_COLOR},
        title={"font": {"color": CHART_TEXT_COLOR}},
        title_standoff=10,
    )
    if source_note:
        fig.add_annotation(
            text=source_note,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.16,
            showarrow=False,
            font={"size": 10, "color": CHART_AXIS_COLOR},
        )
    return fig


def render_report_plotly_chart(fig: Any) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG)


def render_report_column_chart(column: Any, fig: Any) -> None:
    fig.update_layout(
        height=max(int(fig.layout.height or 360), 620),
        margin={"l": 28, "r": 18, "t": 82, "b": 72},
        title={"font": {"size": 16, "color": CHART_TEXT_COLOR}},
        font={"size": 10, "color": CHART_TEXT_COLOR},
        legend={"font": {"size": 9, "color": CHART_TEXT_COLOR}, "title": {"font": {"color": CHART_TEXT_COLOR}}},
    )
    fig.update_xaxes(
        tickfont={"size": 10, "color": CHART_AXIS_COLOR},
        title={"font": {"color": CHART_TEXT_COLOR}},
        title_standoff=8,
    )
    fig.update_yaxes(
        tickfont={"size": 10, "color": CHART_AXIS_COLOR},
        title={"font": {"color": CHART_TEXT_COLOR}},
        title_standoff=8,
    )
    column.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG)


def edition_tick_values(frame: pd.DataFrame) -> list[int]:
    return sorted(pd.to_numeric(frame.get("edition", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())


def set_edition_ticks(fig: Any, frame: pd.DataFrame, tickangle: int = 35) -> Any:
    ticks = edition_tick_values(frame)
    if ticks:
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[str(tick) for tick in ticks], tickangle=tickangle)
    return fig


def report_axis_title(text: str) -> dict[str, Any]:
    return {"text": text, "font": {"color": CHART_TEXT_COLOR}}


def render_kpi_cards(values: list[tuple[str, object]]) -> None:
    cards = "".join(
        (
            f'<div class="trc-kpi">'
            f'<div class="trc-kpi-label">{label}</div>'
            f'<div class="trc-kpi-value">{value}</div>'
            f"</div>"
        )
        for label, value in values
    )
    st.markdown(f'<div class="trc-kpi-grid">{cards}</div>', unsafe_allow_html=True)


def add_era_backgrounds(fig: Any, frame: pd.DataFrame) -> Any:
    if "era" not in frame.columns:
        return fig
    era_frame = frame.dropna(subset=["edition", "era"]).copy()
    if era_frame.empty:
        return fig
    era_ranges = (
        era_frame.groupby("era", observed=True)["edition"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values("min")
    )
    for row in era_ranges.itertuples(index=False):
        era_name = str(row.era)
        fig.add_vrect(
            x0=float(row.min) - 1.8,
            x1=float(row.max) + 1.8,
            fillcolor=ERA_COLORS.get(era_name, "#8B7355"),
            opacity=0.12,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            text=era_name,
            x=(float(row.min) + float(row.max)) / 2,
            y=1.04,
            xref="x",
            yref="paper",
            showarrow=False,
            font={"size": 9, "color": CHART_AXIS_COLOR},
        )
    return fig


def add_qualification_stage_backgrounds(fig: Any, frame: pd.DataFrame) -> Any:
    if frame.empty or "qualification_stage" not in frame.columns or "match_index" not in frame.columns:
        return fig

    stage_runs: list[dict[str, object]] = []
    for row in frame.loc[:, ["match_index", "qualification_stage"]].itertuples(index=False):
        stage = str(row.qualification_stage)
        match_index = int(row.match_index)
        if not stage_runs or stage_runs[-1]["stage"] != stage:
            stage_runs.append({"stage": stage, "start": match_index, "end": match_index})
        else:
            stage_runs[-1]["end"] = match_index

    for run in stage_runs:
        stage = str(run["stage"])
        start = int(run["start"])
        end = int(run["end"])
        fig.add_vrect(
            x0=start - 0.5,
            x1=end + 0.5,
            fillcolor=QUALIFICATION_STAGE_COLORS.get(stage, CHART_AXIS_COLOR),
            opacity=0.12,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            text=QUALIFICATION_STAGE_DISPLAY_LABELS.get(stage, stage),
            x=(start + end) / 2,
            y=1.04,
            xref="x",
            yref="paper",
            showarrow=False,
            font={"size": 10, "color": CHART_AXIS_COLOR},
        )
    return fig


def add_match_type_backgrounds(fig: Any, frame: pd.DataFrame) -> Any:
    if frame.empty or "match_type" not in frame.columns or "match_index" not in frame.columns:
        return fig

    match_type_runs: list[dict[str, object]] = []
    for row in frame.loc[:, ["match_index", "match_type"]].itertuples(index=False):
        match_type = str(row.match_type)
        match_index = int(row.match_index)
        if not match_type_runs or match_type_runs[-1]["match_type"] != match_type:
            match_type_runs.append({"match_type": match_type, "start": match_index, "end": match_index})
        else:
            match_type_runs[-1]["end"] = match_index

    for run in match_type_runs:
        match_type = str(run["match_type"])
        start = int(run["start"])
        end = int(run["end"])
        fig.add_vrect(
            x0=start - 0.5,
            x1=end + 0.5,
            fillcolor=MATCH_TYPE_COLORS.get(match_type, CHART_AXIS_COLOR),
            opacity=0.11,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            text=MATCH_TYPE_DISPLAY_LABELS.get(match_type, match_type),
            x=(start + end) / 2,
            y=1.04,
            xref="x",
            yref="paper",
            showarrow=False,
            font={"size": 9, "color": CHART_AXIS_COLOR},
        )
    return fig


def render_identity_header(context: dict[str, Any]) -> None:
    """Render the top hero block for the selected team."""
    team_row = context["team_row"]
    overall = context["overall_summary"]
    flag_code = str(team_row.get("flag_icon_code", "") or "")
    flag_html = f'<span class="fi fi-{flag_code}"></span>' if flag_code else ""
    display_name = html.escape(str(team_row["display_name"]))
    group_code = html.escape(str(team_row["group_code"]))
    confederation = html.escape(str(team_row["confederation"]))
    overall_grade = html.escape(str(overall["grade"]))
    overall_verdict = html.escape(str(overall["verdict"]))
    fact_cards = "".join(
        (
            f'<div class="trc-fact">'
            f'<div class="trc-fact-label">{html.escape(str(row["label"]))}</div>'
            f'<div class="trc-fact-value">{html.escape(str(row["value"]))}</div>'
            f"</div>"
        )
        for row in context["identity_rows"]
    )
    st.markdown(
        f"""
        <div class="trc-hero">
          <div class="trc-hero-top">
            <div>
              <div class="trc-title">
                {flag_html}
                <div>
                  <h1>{display_name}</h1>
                  <div class="trc-subhead">Group {group_code} · {confederation}</div>
                </div>
              </div>
            </div>
            <div class="trc-grade-panel">
              <div class="trc-grade-country">{display_name}</div>
              <div class="trc-grade-kicker">Overall Report Card</div>
              <div class="trc-grade">{overall_grade}</div>
              <div class="trc-score">{overall['score']:.1f} / 10</div>
              <div class="trc-verdict">{overall_verdict}</div>
            </div>
          </div>
          <div class="trc-facts">
            {fact_cards}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(overall["summary"])


def render_subject_cards(context: dict[str, Any]) -> None:
    """Render the scored and pending subject cards."""
    scored_html = "".join(
        (
            f'<div class="trc-subject-card">'
            f'<div class="trc-subject-head"><h3>{row["subject"]}</h3><span class="trc-grade-chip">{row["grade"]}</span></div>'
            f'<div class="trc-subject-score">{row["score"]:.1f} / 10</div>'
            f'<div class="trc-subject-note">{row["note"]}</div>'
            f"</div>"
        )
        for row in context["subject_rows"]
    )
    pending_html = "".join(
        (
            f'<div class="trc-pending-card">'
            f"<h3>{row['subject']}</h3>"
            f'<div class="trc-pending-value">{row["value"]}</div>'
            f"</div>"
        )
        for row in context["pending_subject_rows"]
    )
    st.markdown(f'<div class="trc-subject-grid">{scored_html}{pending_html}</div>', unsafe_allow_html=True)


def build_plotly_figure_library() -> tuple[Any, Any]:
    """Import Plotly lazily so tests do not require it at module import time."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@st.cache_data(show_spinner=False)
def load_report_card_historical_placement() -> pd.DataFrame:
    placement = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "placement.csv")
    placement = add_era_column(placement)
    placement["team_key"] = placement["country"].map(normalize_historical_team_name)
    for column_name in ["edition", "position", "gs", "ga", "matches_played"]:
        if column_name in placement.columns:
            placement[column_name] = pd.to_numeric(placement[column_name], errors="coerce")
    return placement


def selected_team_history_keys(team_row: pd.Series) -> set[str]:
    candidates = {
        str(team_row.get("team_id", "")),
        str(team_row.get("display_name", "")),
        str(team_row.get("canonical_name", "")),
        str(team_row.get("team_name", "")),
        str(team_row.get("tournament_name", "")),
    }
    return {normalize_historical_team_name(candidate) for candidate in candidates if candidate and candidate != "nan"}


def historical_placement_label(row: Any) -> str:
    placement = str(getattr(row, "placement", "") or "")
    return placement if placement and placement != "nan" else f"Position {int(getattr(row, 'position'))}"


def prepare_team_historical_profile(context: dict[str, Any]) -> pd.DataFrame:
    placement = load_report_card_historical_placement()
    keys = selected_team_history_keys(context["team_row"])
    team_history = placement.loc[placement["team_key"].isin(keys)].copy()
    if team_history.empty:
        return team_history
    team_history = team_history.sort_values(["edition", "position"], na_position="last")
    team_history = (
        team_history.groupby(["edition", "era"], as_index=False, observed=True)
        .agg(
            country=("country", "first"),
            placement=("placement", "first"),
            position=("position", "min"),
            goals_for=("gs", "sum"),
            goals_against=("ga", "sum"),
            matches_played=("matches_played", "sum"),
        )
        .sort_values("edition")
        .reset_index(drop=True)
    )
    team_history["goals_for_per_game"] = (
        team_history["goals_for"] / team_history["matches_played"].replace(0, pd.NA)
    ).astype("Float64").round(3)
    team_history["goals_against_per_game"] = (
        team_history["goals_against"] / team_history["matches_played"].replace(0, pd.NA)
    ).astype("Float64").round(3)
    team_history["placement_label"] = [
        historical_placement_label(row) for row in team_history.itertuples(index=False)
    ]
    team_history["placement_short_label"] = team_history["placement"].map(PLACEMENT_SHORT_LABELS).fillna(
        team_history["placement_label"]
    )
    return team_history


def render_historical_team_charts(context: dict[str, Any]) -> None:
    history = prepare_team_historical_profile(context)
    team_name = str(context["team_row"].get("display_name", context["team_row"].get("team_id", "Team")))
    if history.empty:
        if is_debut_tournament(context["team_row"]):
            st.info("No historical World Cup placement or scoring history is available for this team. This is their debut tournament.")
            return
        st.info("No historical World Cup placement or scoring history is available for this team.")
        return

    go, _ = build_plotly_figure_library()
    placement_fig = go.Figure()
    placement_fig.add_trace(
        go.Scatter(
            x=history["edition"],
            y=history["position"],
            mode="lines+markers+text",
            text=history["placement_short_label"],
            textposition="top center",
            name="Placement",
            hovertemplate=(
                "Edition: %{x}<br>"
                "Placement: %{customdata[0]}<br>"
                "Position: %{y}<extra></extra>"
            ),
            customdata=history[["placement"]],
            line={"color": CHART_ACCENT_COLOR, "width": 1.8},
            marker={"color": CHART_ACCENT_COLOR, "size": 6},
        )
    )
    placement_ticks = history.dropna(subset=["position"]).sort_values("position").drop_duplicates("position")
    apply_report_card_chart_style(
        placement_fig,
        "Placement by Edition",
        height=460,
        source_note=SOURCE_NOTE,
        country_name=team_name,
    )
    add_era_backgrounds(placement_fig, history)
    placement_fig.update_yaxes(
        autorange="reversed",
        title=report_axis_title("Placement"),
        tickmode="array",
        tickvals=placement_ticks["position"].tolist(),
        ticktext=placement_ticks["placement_label"].tolist(),
    )
    set_edition_ticks(placement_fig, history)
    placement_fig.update_xaxes(title=report_axis_title("Edition"))
    render_report_plotly_chart(placement_fig)

    goals_for_fig = go.Figure()
    goals_for_fig.add_trace(
        go.Scatter(
            x=history["edition"],
            y=history["goals_for"],
            mode="lines+markers+text",
            text=history["goals_for"].map(lambda value: f"{float(value):.0f}" if pd.notna(value) else ""),
            textposition="top center",
            name="Goals scored",
            hovertemplate="Edition: %{x}<br>Goals scored: %{y:.0f}<extra></extra>",
            line={"color": CHART_POSITIVE_COLOR, "width": 1.8},
            marker={"color": CHART_POSITIVE_COLOR, "size": 6},
        )
    )
    apply_report_card_chart_style(
        goals_for_fig,
        "Goals Scored",
        height=430,
        source_note=SOURCE_NOTE,
        country_name=team_name,
    )
    add_era_backgrounds(goals_for_fig, history)
    goals_for_fig.update_yaxes(title=report_axis_title("Goals scored"))
    set_edition_ticks(goals_for_fig, history, tickangle=45)
    goals_for_fig.update_xaxes(title=report_axis_title("Edition"))

    goals_against_fig = go.Figure()
    goals_against_fig.add_trace(
        go.Scatter(
            x=history["edition"],
            y=history["goals_against"],
            mode="lines+markers+text",
            text=history["goals_against"].map(lambda value: f"{float(value):.0f}" if pd.notna(value) else ""),
            textposition="top center",
            name="Goals conceded",
            hovertemplate="Edition: %{x}<br>Goals conceded: %{y:.0f}<extra></extra>",
            line={"color": CHART_NEGATIVE_COLOR, "width": 1.8},
            marker={"color": CHART_NEGATIVE_COLOR, "size": 6},
        )
    )
    apply_report_card_chart_style(
        goals_against_fig,
        "Goals Conceded",
        height=430,
        source_note=SOURCE_NOTE,
        country_name=team_name,
    )
    add_era_backgrounds(goals_against_fig, history)
    goals_against_fig.update_yaxes(title=report_axis_title("Goals conceded"))
    set_edition_ticks(goals_against_fig, history, tickangle=45)
    goals_against_fig.update_xaxes(title=report_axis_title("Edition"))

    scored_col, conceded_col = st.columns(2)
    render_report_column_chart(scored_col, goals_for_fig)
    render_report_column_chart(conceded_col, goals_against_fig)


def render_qualification_path_section(context: dict[str, Any]) -> None:
    """Render the selected team's World Cup qualification path."""
    qualification_path = context["qualification_path"].copy()
    team_name = str(context["team_row"].get("display_name", context["team_row"].get("team_id", "Team")))
    st.subheader("Qualification Path")
    if qualification_path.empty:
        st.info("No World Cup qualification path is available for this team in the current lead-in data.")
        return

    wins = int(qualification_path["Result"].eq("W").sum())
    draws = int(qualification_path["Result"].eq("D").sum())
    losses = int(qualification_path["Result"].eq("L").sum())
    goals_for = int(qualification_path["team_score"].fillna(0).sum())
    goals_against = int(qualification_path["opponent_score"].fillna(0).sum())
    goal_difference = int(qualification_path["goal_difference"].sum())
    points = int(qualification_path["points"].sum())
    avg_opp_elo = qualification_path["opponent_elo_start"].dropna().mean()
    total_elo_change = float(qualification_path["team_elo_delta"].fillna(0.0).sum())
    render_kpi_cards(
        [
            ("Matches", len(qualification_path)),
            ("W-D-L", f"{wins}-{draws}-{losses}"),
            ("Goals", f"{goals_for}-{goals_against}"),
            ("Goal Diff", f"{goal_difference:+d}"),
            ("Points", points),
            ("Avg Opp Elo", f"{avg_opp_elo:.0f}" if pd.notna(avg_opp_elo) else "N/A"),
            ("Elo Change", f"{total_elo_change:+.1f}"),
        ]
    )

    go, _ = build_plotly_figure_library()
    chart_df = qualification_path.copy()
    chart_df["match_index"] = range(len(chart_df))
    result_colors = qualification_path["Result"].map(
        {"W": CHART_POSITIVE_COLOR, "D": "#C99700", "L": CHART_NEGATIVE_COLOR}
    ).fillna(CHART_AXIS_COLOR)
    customdata = chart_df[["Opponent", "Score", "Venue", "Competition", "team_elo_delta", "Result", "Date"]]

    timeline_fig = go.Figure()
    timeline_fig.add_trace(
        go.Scatter(
            x=chart_df["match_index"],
            y=chart_df["cumulative_points"],
            mode="lines+markers+text",
            text=chart_df["Result"],
            textposition="top center",
            name="Cumulative points",
            line={"color": CHART_ACCENT_COLOR, "width": 1.8},
            marker={"color": result_colors, "size": 8, "line": {"color": CHART_AXIS_COLOR, "width": 0.5}},
            customdata=customdata,
            hovertemplate=(
                "Date: %{customdata[6]}<br>"
                "Opponent: %{customdata[0]}<br>"
                "Score: %{customdata[1]} (%{customdata[5]})<br>"
                "Cumulative points: %{y}<br>"
                "Venue: %{customdata[2]}<br>"
                "Competition: %{customdata[3]}<br>"
                "Elo change: %{customdata[4]:+.1f}<extra></extra>"
            ),
        )
    )
    apply_report_card_chart_style(timeline_fig, "Qualification Results Timeline", height=420, country_name=team_name)
    add_qualification_stage_backgrounds(timeline_fig, chart_df)
    timeline_fig.update_xaxes(tickmode="array", tickvals=chart_df["match_index"], ticktext=chart_df["Date"], tickangle=35)
    timeline_fig.update_xaxes(title=report_axis_title("Match Date"))
    timeline_fig.update_yaxes(title=report_axis_title("Cumulative Points"))

    goals_fig = go.Figure()
    goals_fig.add_trace(
        go.Bar(
            x=chart_df["match_index"],
            y=chart_df["team_score"],
            name="Goals Scored",
            marker={"color": CHART_POSITIVE_COLOR, "line": {"color": CHART_AXIS_COLOR, "width": 0.5}},
            text=chart_df["team_score"].map(lambda value: f"{float(value):.0f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            customdata=chart_df[["Opponent", "Result", "match_label"]],
            hovertemplate="Match: %{customdata[2]}<br>Opponent: %{customdata[0]}<br>Goals scored: %{y:.0f}<br>Result: %{customdata[1]}<extra></extra>",
        )
    )
    goals_fig.add_trace(
        go.Bar(
            x=chart_df["match_index"],
            y=chart_df["opponent_score"],
            name="Goals Against",
            marker={"color": CHART_NEGATIVE_COLOR, "line": {"color": CHART_AXIS_COLOR, "width": 0.5}},
            text=chart_df["opponent_score"].map(lambda value: f"{float(value):.0f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            customdata=chart_df[["Opponent", "Result", "match_label"]],
            hovertemplate="Match: %{customdata[2]}<br>Opponent: %{customdata[0]}<br>Goals against: %{y:.0f}<br>Result: %{customdata[1]}<extra></extra>",
        )
    )
    apply_report_card_chart_style(goals_fig, "Qualification Goals", height=420, country_name=team_name)
    add_qualification_stage_backgrounds(goals_fig, chart_df)
    goals_fig.update_xaxes(tickmode="array", tickvals=chart_df["match_index"], ticktext=chart_df["match_label"], tickangle=35)
    goals_fig.update_xaxes(title=report_axis_title("Match"))
    goals_fig.update_yaxes(title=report_axis_title("Goals"))
    goals_fig.update_layout(barmode="group")

    elo_fig = go.Figure()
    elo_delta = chart_df["team_elo_delta"].fillna(0.0).astype(float)
    elo_fig.add_trace(
        go.Bar(
            x=chart_df["match_index"],
            y=elo_delta,
            name="Elo change",
            marker={
                "color": elo_delta.map(lambda value: CHART_POSITIVE_COLOR if value >= 0 else CHART_NEGATIVE_COLOR),
                "line": {"color": CHART_AXIS_COLOR, "width": 0.5},
            },
            text=elo_delta.map(lambda value: f"{value:+.1f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            customdata=chart_df[["Opponent", "Score", "Result", "match_label"]],
            hovertemplate="Match: %{customdata[3]}<br>Opponent: %{customdata[0]}<br>Score: %{customdata[1]} (%{customdata[2]})<br>Elo change: %{y:+.1f}<extra></extra>",
        )
    )
    apply_report_card_chart_style(elo_fig, "Qualification Elo Path", height=420, country_name=team_name)
    add_qualification_stage_backgrounds(elo_fig, chart_df)
    elo_fig.add_hline(y=0, line_color=CHART_AXIS_COLOR, line_width=1)
    elo_fig.update_xaxes(tickmode="array", tickvals=chart_df["match_index"], ticktext=chart_df["match_label"], tickangle=35)
    elo_fig.update_xaxes(title=report_axis_title("Match"))
    elo_fig.update_yaxes(title=report_axis_title("Elo Change"))

    render_report_plotly_chart(timeline_fig)
    goals_column, elo_column = st.columns(2)
    render_report_column_chart(goals_column, goals_fig)
    render_report_column_chart(elo_column, elo_fig)


def render_road_here_charts(context: dict[str, Any]) -> None:
    """Render recent lead-in charts for the selected team."""
    recent_matches = context["recent_matches"].copy()
    team_name = str(context["team_row"].get("display_name", context["team_row"].get("team_id", "Team")))
    if recent_matches.empty:
        st.info("No recent match history is available for this team.")
        return

    go, _ = build_plotly_figure_library()
    chart_df = recent_matches.sort_values(["date", "lead_in_id"], kind="stable").copy()
    chart_df["match_index"] = range(len(chart_df))
    chart_df["match_label"] = chart_df["Date"] + " vs " + chart_df["Opponent"]
    chart_df["match_type"] = [
        match_type_label(competition, match_date)
        for competition, match_date in zip(chart_df["Competition"], chart_df["date"], strict=False)
    ]

    elo_fig = go.Figure()
    elo_fig.add_trace(
        go.Scatter(
            x=chart_df["match_index"],
            y=chart_df["post_match_elo"],
            mode="lines+markers",
            name="Post-match Elo",
            line={"color": CHART_SECONDARY_COLOR, "width": 1.8},
            marker={"color": CHART_SECONDARY_COLOR, "size": 6},
            customdata=chart_df[["match_label", "Elo Change", "Result", "Score", "match_type"]],
            hovertemplate=(
                "Match: %{customdata[0]}<br>"
                "Type: %{customdata[4]}<br>"
                "Post-match Elo: %{y:.0f}<br>"
                "Elo change: %{customdata[1]}<br>"
                "Result: %{customdata[2]}<br>"
                "Score: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    apply_report_card_chart_style(elo_fig, "Recent Elo Trend", height=340, country_name=team_name)
    add_match_type_backgrounds(elo_fig, chart_df)
    elo_fig.update_xaxes(
        tickmode="array",
        tickvals=chart_df["match_index"],
        ticktext=chart_df["match_label"],
        tickangle=35,
        title=report_axis_title("Match"),
    )
    elo_fig.update_yaxes(title=report_axis_title("ELO Rating"))

    perf_fig = go.Figure()
    performance_delta = chart_df["perf_vs_exp"].astype(float)
    perf_fig.add_trace(
        go.Bar(
            x=chart_df["match_index"],
            y=performance_delta,
            text=performance_delta.map(lambda value: f"{value:+.2f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            name="Actual minus expected",
            marker={
                "color": performance_delta.map(lambda value: CHART_POSITIVE_COLOR if value >= 0 else CHART_NEGATIVE_COLOR),
                "line": {"color": CHART_AXIS_COLOR, "width": 0.5},
            },
            customdata=chart_df[["match_label", "actual_score", "expected_score", "match_type"]],
            hovertemplate=(
                "Match: %{customdata[0]}<br>"
                "Type: %{customdata[3]}<br>"
                "Actual score: %{customdata[1]:.2f}<br>"
                "Expected score: %{customdata[2]:.2f}<br>"
                "Difference: %{y:+.2f}<extra></extra>"
            ),
        )
    )
    apply_report_card_chart_style(
        perf_fig,
        "Actual vs Expected Performance Difference",
        height=340,
        country_name=team_name,
    )
    add_match_type_backgrounds(perf_fig, chart_df)
    perf_fig.add_hline(y=0, line_color=CHART_AXIS_COLOR, line_width=1)
    perf_fig.update_xaxes(
        tickmode="array",
        tickvals=chart_df["match_index"],
        ticktext=chart_df["match_label"],
        tickangle=35,
        title=report_axis_title("Match"),
    )
    perf_fig.update_yaxes(title=report_axis_title("Performance Differential Score"))

    goal_fig = go.Figure()
    goal_fig.add_trace(
        go.Bar(
            x=chart_df["match_index"],
            y=chart_df["team_score"],
            name="Goals Scored",
            marker={"color": CHART_POSITIVE_COLOR, "line": {"color": CHART_AXIS_COLOR, "width": 0.5}},
            text=chart_df["team_score"].map(lambda value: f"{float(value):.0f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            customdata=chart_df[["match_label", "opponent_score", "Result", "match_type"]],
            hovertemplate=(
                "Match: %{customdata[0]}<br>"
                "Type: %{customdata[3]}<br>"
                "Goals scored: %{y:.0f}<br>"
                "Goals against: %{customdata[1]:.0f}<br>"
                "Result: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    goal_fig.add_trace(
        go.Bar(
            x=chart_df["match_index"],
            y=chart_df["opponent_score"],
            name="Goals Against",
            marker={"color": CHART_NEGATIVE_COLOR, "line": {"color": CHART_AXIS_COLOR, "width": 0.5}},
            text=chart_df["opponent_score"].map(lambda value: f"{float(value):.0f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            customdata=chart_df[["match_label", "team_score", "Result", "match_type"]],
            hovertemplate=(
                "Match: %{customdata[0]}<br>"
                "Type: %{customdata[3]}<br>"
                "Goals against: %{y:.0f}<br>"
                "Goals scored: %{customdata[1]:.0f}<br>"
                "Result: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    apply_report_card_chart_style(goal_fig, "Goals Scored vs Goals Against", height=340, country_name=team_name)
    add_match_type_backgrounds(goal_fig, chart_df)
    goal_fig.update_xaxes(
        tickmode="array",
        tickvals=chart_df["match_index"],
        ticktext=chart_df["match_label"],
        tickangle=35,
        title=report_axis_title("Match"),
    )
    goal_fig.update_yaxes(title=report_axis_title("Goals"))
    goal_fig.update_layout(barmode="group")

    breakdown_fig = go.Figure(
        go.Pie(
            labels=["Wins", "Draws", "Losses"],
            values=[
                int(chart_df["normalized_result"].eq("win").sum()),
                int(chart_df["normalized_result"].eq("draw").sum()),
                int(chart_df["normalized_result"].eq("loss").sum()),
            ],
            hole=0.55,
            marker={
                "colors": [CHART_POSITIVE_COLOR, "#C99700", CHART_NEGATIVE_COLOR],
                "line": {"color": CHART_BACKGROUND, "width": 2},
            },
            textfont={"color": CHART_TEXT_COLOR, "size": 12},
            hovertemplate="%{label}: %{value} matches<br>%{percent}<extra></extra>",
        )
    )
    apply_report_card_chart_style(breakdown_fig, "Win / Draw / Loss Breakdown", height=340, country_name=team_name)
    breakdown_fig.update_traces(textinfo="label+value", insidetextfont={"color": CHART_BACKGROUND})

    top_cols = st.columns(2)
    render_report_column_chart(top_cols[0], elo_fig)
    render_report_column_chart(top_cols[1], perf_fig)

    middle_cols = st.columns(2)
    render_report_column_chart(middle_cols[0], goal_fig)
    render_report_column_chart(middle_cols[1], breakdown_fig)


def render_outlook_charts(context: dict[str, Any]) -> None:
    """Render prediction-focused charts for the selected team."""
    team_name = str(context["team_row"].get("display_name", context["team_row"].get("team_id", "Team")))
    go, _ = build_plotly_figure_library()
    stage_prob = context["stage_probability_table"].copy()
    prob_fig = go.Figure(
        go.Bar(
            x=stage_prob["Probability"],
            y=stage_prob["Stage"],
            orientation="h",
            marker={"color": CHART_SECONDARY_COLOR, "line": {"color": CHART_AXIS_COLOR, "width": 0.5}},
            text=stage_prob["Probability"].map(lambda value: f"{float(value):.1f}%"),
            textposition="outside",
            textfont={"color": CHART_TEXT_COLOR, "size": 10},
            cliponaxis=False,
            hovertemplate="Stage: %{y}<br>Probability: %{x:.1f}%<extra></extra>",
        )
    )
    apply_report_card_chart_style(prob_fig, "Tournament Probability Breakdown", height=340, country_name=team_name)
    prob_fig.update_xaxes(title=report_axis_title("Probability"))
    prob_fig.update_yaxes(title=report_axis_title("Stage"))

    radar_rows = context["subject_rows"]
    radar_labels = [row["subject"] for row in radar_rows]
    radar_scores = [float(row["score"]) for row in radar_rows]
    radar_labels.append(radar_labels[0])
    radar_scores.append(radar_scores[0])
    radar_fig = go.Figure(
        go.Scatterpolar(
            r=radar_scores,
            theta=radar_labels,
            fill="toself",
            name=context["team_row"]["display_name"],
            line={"color": CHART_ACCENT_COLOR, "width": 2},
            marker={"color": CHART_ACCENT_COLOR, "size": 5},
            fillcolor="rgba(122, 78, 45, 0.22)",
            hovertemplate="%{theta}: %{r:.1f} / 10<extra></extra>",
        )
    )
    apply_report_card_chart_style(radar_fig, "Team Profile Radar", height=340, country_name=team_name)
    radar_fig.update_layout(
        polar={
            "bgcolor": CHART_BACKGROUND,
            "radialaxis": {
                "visible": True,
                "range": [1, 10],
                "gridcolor": CHART_GRID_COLOR,
                "linecolor": CHART_AXIS_COLOR,
                "tickfont": {"color": CHART_AXIS_COLOR, "size": 10},
            },
            "angularaxis": {
                "gridcolor": CHART_GRID_COLOR,
                "linecolor": CHART_AXIS_COLOR,
                "tickfont": {"color": CHART_AXIS_COLOR, "size": 10},
            },
        },
        showlegend=False,
    )

    cols = st.columns(2)
    render_report_column_chart(cols[0], prob_fig)
    render_report_column_chart(cols[1], radar_fig)


def render_history_tab(context: dict[str, Any]) -> None:
    st.subheader("Historical World Cup Performance")
    render_historical_team_charts(context)


def render_road_here_tab(context: dict[str, Any]) -> None:
    st.subheader("Road Here")
    render_qualification_path_section(context)
    render_road_here_charts(context)
    render_recent_performance(context)


def render_outlook_tab(context: dict[str, Any]) -> None:
    st.subheader("Tournament Outlook")
    render_outlook_charts(context)
    render_prediction_outlook(context)
    render_fixtures_and_path(context)


def render_recent_performance(context: dict[str, Any]) -> None:
    """Render the recent performance section."""
    recent_table = context["recent_matches"].loc[
        :,
        ["Date", "Opponent", "Competition", "Result", "Score", "Elo Change", "Performance Score", "Grade"],
    ]
    st.subheader("Recent Performance")
    st.dataframe(recent_table, width="stretch", hide_index=True)


def render_prediction_outlook(context: dict[str, Any]) -> None:
    """Render the probability tables and model explanation."""
    st.subheader("Prediction Outlook")
    cols = st.columns(2)
    with cols[0]:
        group_table = context["group_finish_table"].copy()
        group_table["Probability"] = group_table["Probability"].map(format_percent)
        st.caption("Group Finish Probabilities")
        st.dataframe(group_table, width="stretch", hide_index=True)
    with cols[1]:
        stage_table = context["stage_probability_table"].copy()
        stage_table["Probability"] = stage_table["Probability"].map(format_percent)
        st.caption("Tournament Stage Probabilities")
        st.dataframe(stage_table, width="stretch", hide_index=True)

    st.caption("Why the model likes this team")
    for bullet in context["model_reason_bullets"]:
        st.write(f"- {bullet}")


def render_fixtures_and_path(context: dict[str, Any]) -> None:
    """Render upcoming fixtures and projected knockout path."""
    st.subheader("Fixtures And Path")
    cols = st.columns(2)
    with cols[0]:
        st.caption("Group Stage Fixtures")
        st.dataframe(context["group_fixtures"], width="stretch", hide_index=True)
    with cols[1]:
        first_knockout = context["first_knockout_match"]
        if first_knockout is None:
            st.caption("Projected Knockout Entry")
            st.info("The modal bracket currently projects a group-stage exit.")
        else:
            st.caption("Projected First Knockout Match")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Stage": first_knockout["Stage"],
                            "Opponent": first_knockout["Opponent"],
                            "Matchup Win %": format_percent(first_knockout["Matchup Win %"]),
                            "Projected Winner": first_knockout["Projected Winner"],
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    st.caption("Projected Knockout Path")
    if context["knockout_path"].empty:
        st.info("No knockout path is projected for this team in the modal bracket.")
    else:
        path_table = context["knockout_path"].copy()
        path_table["Matchup Win %"] = path_table["Matchup Win %"].map(format_percent)
        st.dataframe(path_table, width="stretch", hide_index=True)


def render_team_report_card_page() -> None:
    """Render the dedicated primary-model team report-card page."""
    home.inject_styles()
    st.markdown(f"<style>{report_card_css()}</style>", unsafe_allow_html=True)

    _, _, _, metadata = home.load_data()
    _, official_simulations, official_match_window = official_report_card_settings(metadata)
    home.render_dashboard_header(
        home.load_world_cup_logo_data_uri(),
        metadata,
        official_simulations,
        title="World Cup 2026 Team Report Card",
        model_version=PRIMARY_MODEL.model_version,
        model_label=PRIMARY_MODEL.model_label,
    )

    try:
        with st.spinner(f"Loading official report-card projection with {official_simulations:,} simulations..."):
            dataset = build_report_card_dataset()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    for warning in dataset.get("artifact_warnings", ()):
        st.warning(warning)
    team_choices = (
        dataset["dashboard_df"]
        .loc[:, ["team_id", "display_name", "group_code"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .sort_values(["group_code", "display_name"], kind="stable")
        .reset_index(drop=True)
    )
    team_ids = team_choices["team_id"].astype(str).tolist()
    labels = [f'{row.display_name} (Group {row.group_code})' for row in team_choices.itertuples(index=False)]
    query_team_id = get_query_team_param()
    selected_index = team_ids.index(query_team_id) if query_team_id in team_ids else 0
    artifact_updated_at = format_artifact_updated_at(dataset.get("artifact_created_at_utc"))
    st.caption(
        f"This report card uses the official Enhanced Poisson Model projections | "
        f"{artifact_updated_at} |  "
        f"the last {official_match_window} Elo-rated matches, "
        "historical World Cup pedigree, and the modal deterministic bracket."
    )

    selector_columns = st.columns([2, 1])
    with selector_columns[0]:
        selected_team_id = st.selectbox(
            "Choose a team to view",
            team_ids,
            index=selected_index,
            format_func=lambda value: labels[team_ids.index(value)],
            key="team_report_card_team_id",
        )
    set_query_team_param(selected_team_id)

    context = select_report_card_context(dataset, selected_team_id)
    render_identity_header(context)
    render_subject_cards(context)
    history_tab, road_here_tab, outlook_tab = st.tabs(["History", "Road Here", "Outlook"])
    with history_tab:
        render_history_tab(context)
    with road_here_tab:
        render_road_here_tab(context)
    with outlook_tab:
        render_outlook_tab(context)

