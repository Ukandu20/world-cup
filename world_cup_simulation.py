from __future__ import annotations

import base64
import json
import zlib
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GROUP_ORDER = tuple("ABCDEFGHIJKL")
MODEL_VERSION = "v1"
MODEL_LABEL = "Team-strength Monte Carlo"
MODEL_SUMMARY = "Elo + recent form -> expected goals -> Poisson simulation"
RECENT_MATCH_WINDOW = 10
RESULT_POINTS = {"win": 3, "draw": 1, "loss": 0}
BASELINE_RATING_WEIGHTS = (1.0, 0.0)
FORM_COMPONENT_WEIGHTS = (0.7, 0.3)
STRENGTH_BLEND_WEIGHTS = (0.5, 0.5)
WEIGHTED_FORM_COMPOSITE_WEIGHTS = (0.4, 0.25, 0.25, 0.10)
WEIGHTED_FORM_GOAL_DIFFERENCE_CAP = 4
WEIGHTED_FORM_GD_BOUNDS = (-4.0, 4.0)
WEIGHTED_FORM_ELO_BOUNDS = (-15.0, 15.0)
WEIGHTED_FORM_PERF_BOUNDS = (-0.5, 0.5)
V2_HISTORY_COMPONENT_WEIGHTS = (0.7, 0.3)
V2_STRENGTH_BLEND_WEIGHTS = (0.4, 0.4, 0.2)
WORLD_CUP_HISTORY_EDITION_COUNT = 22
WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT = float(
    sum((edition_index + 1) ** 2 for edition_index in range(WORLD_CUP_HISTORY_EDITION_COUNT))
)
EXPECTED_GOALS_BASE = 1.20
EXPECTED_GOALS_SCALE = 0.40
EXPECTED_GOALS_MIN = 0.20
EXPECTED_GOALS_MAX = 3.00
FORM_SCHEDULE_DIFFICULTY_MIN = 1.0
FORM_SCHEDULE_DIFFICULTY_MAX = 5.0
FORM_SCHEDULE_DIFFICULTY_NEUTRAL = 3.0
BEST_THIRD_QUALIFICATION_SLOTS = 8
EXTRA_TIME_FACTOR = 1.0 / 3.0
THIRD_PLACE_ROUTE_MATCHES = (79, 85, 81, 74, 82, 77, 87, 80)
MAIN_BRACKET_ROUND_CODES = ("R32", "R16", "QF", "SF", "F")
ROUND_CODE_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
    "F": "Final",
}
WORLD_CUP_ROOT = Path(__file__).resolve().parent / "INT-World Cup" / "world_cup"
HISTORICAL_RESULTS_START_YEAR = 1950
HISTORICAL_RESULTS_END_YEAR = 2022
V2_MODEL_VERSION = "v2"
V2_MODEL_LABEL = "Multinomial Match Model"
V2_MODEL_SUMMARY = "Historical World Cup multinomial regression -> Monte Carlo simulation"
V2_OUTCOME_LABELS = ("home_win", "draw", "away_win")
V2_FEATURE_COLUMNS = (
    "elo_diff",
    "results_form_diff",
    "gd_form_diff",
    "perf_vs_exp_diff",
    "goals_for_diff",
    "goals_against_diff",
    "placement_diff",
    "appearance_diff",
)
V2_STAGE_GROUP = "group"
V2_STAGE_KNOCKOUT = "knockout"
DEFAULT_SCORELINE_BY_OUTCOME = {
    "home_win": (1, 0),
    "draw": (0, 0),
    "away_win": (0, 1),
}
HISTORICAL_TEAM_NAME_ALIASES = {
    "congo dr": "dr congo",
    "dem rep of congo": "dr congo",
    "dr congo": "dr congo",
    "east germany": "german dr",
    "fr yugoslavia": "serbia",
    "ir iran": "iran",
    "korea republic": "south korea",
    "serbia and montenegro": "serbia",
    "turkiye": "turkey",
    "usa": "united states",
    "zaire": "dr congo",
}
THIRD_PLACE_ROUTING_COMPRESSED = (
    "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"
)


def normalize_weight_pair(primary_weight: float, secondary_weight: float) -> tuple[float, float]:
    """Normalize a two-value weight pair so it sums to 1.0."""
    total = float(primary_weight) + float(secondary_weight)
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return float(primary_weight) / total, float(secondary_weight) / total


def normalize_key(value: str | None) -> str:
    """Normalize text keys for stable historical-name matching."""
    if value is None:
        return ""
    normalized = str(value).strip().lower().replace("-", " ")
    return " ".join(normalized.split())


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a numeric series while handling empty or constant inputs safely."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - series.mean()) / std


def scale_to_range(
    series: pd.Series,
    lower: float = FORM_SCHEDULE_DIFFICULTY_MIN,
    upper: float = FORM_SCHEDULE_DIFFICULTY_MAX,
    neutral: float = FORM_SCHEDULE_DIFFICULTY_NEUTRAL,
) -> pd.Series:
    """Scale a numeric series into a fixed range, falling back to a neutral midpoint when constant."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=float)
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(np.full(len(numeric), neutral), index=numeric.index, dtype=float)
    scaled = lower + (numeric - minimum) * (upper - lower) / (maximum - minimum)
    return scaled.astype(float)


def clip_scale(
    value: float | pd.Series,
    lower: float,
    upper: float,
) -> float | pd.Series:
    """Clip a value into a fixed range and normalize it onto the 0-1 interval."""
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    scaled = (pd.to_numeric(value, errors="coerce") - lower) / (upper - lower)
    clipped = np.clip(scaled, 0.0, 1.0)
    if isinstance(clipped, pd.Series):
        return clipped.astype(float)
    return float(clipped)


def compute_elo_expected_score(team_elo: float | pd.Series, opp_elo: float | pd.Series) -> float | pd.Series:
    """Compute the Elo expected-score value for one team against one opponent."""
    dr = pd.to_numeric(team_elo, errors="coerce") - pd.to_numeric(opp_elo, errors="coerce")
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def normalize_weighted_form_result(
    result_series: pd.Series,
    team_score_series: pd.Series,
    opponent_score_series: pd.Series,
) -> pd.Series:
    """Normalize match outcomes to win/draw/loss, falling back to the scoreline when needed."""
    normalized = (
        result_series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "w": "win",
                "d": "draw",
                "l": "loss",
                "win": "win",
                "draw": "draw",
                "loss": "loss",
            }
        )
    )
    score_based = pd.Series(
        np.select(
            [
                team_score_series > opponent_score_series,
                team_score_series == opponent_score_series,
                team_score_series < opponent_score_series,
            ],
            ["win", "draw", "loss"],
            default=None,
        ),
        index=result_series.index,
        dtype="object",
    )
    return normalized.fillna(score_based)


def load_third_place_routing_map() -> dict[str, dict[int, str]]:
    """Decode the static Round of 32 routing map for best third-place teams."""
    payload = zlib.decompress(base64.b64decode(THIRD_PLACE_ROUTING_COMPRESSED)).decode("utf-8")
    raw_mapping = json.loads(payload)
    return {
        combo_key: {int(match_number): group_code for match_number, group_code in match_mapping.items()}
        for combo_key, match_mapping in raw_mapping.items()
    }


THIRD_PLACE_ROUTING_MAP = load_third_place_routing_map()


def extract_group_stage_fixtures(fixtures_df: pd.DataFrame, group_order: Iterable[str] = DEFAULT_GROUP_ORDER) -> pd.DataFrame:
    """Return validated group-stage fixtures sorted in kickoff order."""
    group_order = list(group_order)
    df = fixtures_df.copy()
    df["match_number"] = pd.to_numeric(df["match_number"], errors="coerce")
    df["kickoff_datetime_utc"] = pd.to_datetime(df["kickoff_datetime_utc"], errors="coerce", utc=True)
    df["group_code"] = df["group_code"].fillna("")

    group_fixtures = (
        df[
            (df["round_code"] == "GS")
            & df["group_code"].isin(group_order)
            & df["home_team_id"].fillna("").ne("")
            & df["away_team_id"].fillna("").ne("")
        ]
        .sort_values(["group_code", "kickoff_datetime_utc", "match_number"], kind="stable")
        .loc[:, ["group_code", "match_number", "kickoff_datetime_utc", "home_team_id", "away_team_id"]]
        .reset_index(drop=True)
    )

    for group_code in group_order:
        group = group_fixtures[group_fixtures["group_code"] == group_code]
        if group.empty:
            continue
        if len(group) != 6:
            raise ValueError(f"Expected 6 group-stage fixtures for Group {group_code}, found {len(group)}")
        appearance_counts = pd.concat([group["home_team_id"], group["away_team_id"]]).value_counts()
        if len(appearance_counts) != 4 or not appearance_counts.eq(3).all():
            raise ValueError(f"Expected 4 teams with 3 fixtures each in Group {group_code}")

    return group_fixtures


def extract_knockout_fixtures(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Return knockout fixtures with their bracket slot labels."""
    df = fixtures_df.copy()
    df["match_number"] = pd.to_numeric(df["match_number"], errors="coerce")
    knockout_fixtures = (
        df[df["round_code"].isin(["R32", "R16", "QF", "SF", "3P", "F"])]
        .sort_values(["match_number"], kind="stable")
        .loc[:, ["match_number", "round_code", "home_slot_label", "away_slot_label"]]
        .reset_index(drop=True)
    )
    if knockout_fixtures.empty:
        raise ValueError("Expected knockout fixtures in fixtures_df")
    return knockout_fixtures


def extract_main_bracket_fixtures(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Return knockout fixtures excluding the third-place playoff."""
    knockout_fixtures = extract_knockout_fixtures(fixtures_df)
    return knockout_fixtures[knockout_fixtures["round_code"].isin(MAIN_BRACKET_ROUND_CODES)].reset_index(drop=True)


def build_recent_form_metrics(
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    form_component_weights: tuple[float, float] = FORM_COMPONENT_WEIGHTS,
) -> pd.DataFrame:
    """Build recent-form metrics from each team's most recent lead-in matches."""
    if match_window <= 0:
        raise ValueError("match_window must be positive")
    points_weight, goal_diff_weight = normalize_weight_pair(*form_component_weights)

    df = lead_in_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["goal_difference"] = pd.to_numeric(df["goal_difference"], errors="coerce").fillna(0.0)
    df["form_points"] = df["result"].map(RESULT_POINTS).fillna(0).astype(float)

    recent = (
        df.sort_values(["qualified_team_id", "date", "lead_in_id"], kind="stable")
        .groupby("qualified_team_id", group_keys=False)
        .tail(match_window)
    )

    form = (
        recent.groupby("qualified_team_id", as_index=False)
        .agg(
            recent_matches=("lead_in_id", "count"),
            points_per_match=("form_points", "mean"),
            goal_diff_per_match=("goal_difference", "mean"),
        )
    )
    form["points_form_z"] = zscore(form["points_per_match"])
    form["goal_diff_form_z"] = zscore(form["goal_diff_per_match"])
    form["form_score"] = points_weight * form["points_form_z"] + goal_diff_weight * form["goal_diff_form_z"]
    return form


def build_weighted_form_table(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    composite_weights: tuple[float, float, float, float] = WEIGHTED_FORM_COMPOSITE_WEIGHTS,
) -> pd.DataFrame:
    """Build an all-teams weighted recent-form table from the last k Elo-rated lead-in matches."""
    if match_window <= 0:
        raise ValueError("match_window must be positive")
    composite_weight_total = float(sum(composite_weights))
    if composite_weight_total <= 0:
        raise ValueError("composite_weights must contain at least one positive value")
    results_weight, gd_weight, perf_weight, elo_delta_weight = tuple(
        float(weight) / composite_weight_total for weight in composite_weights
    )

    required_columns = {
        "lead_in_id",
        "date",
        "qualified_team_id",
        "team_score",
        "opponent_score",
        "result",
        "team_elo_start",
        "opponent_elo_start",
        "team_elo_delta",
    }
    missing_columns = required_columns.difference(lead_in_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"lead_in_df is missing required columns for weighted form metrics: {missing}")

    df = lead_in_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_columns = [
        "team_score",
        "opponent_score",
        "team_elo_start",
        "opponent_elo_start",
        "team_elo_delta",
    ]
    for column_name in numeric_columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df = df.dropna(subset=numeric_columns).copy()
    if df.empty:
        raise ValueError("lead_in_df must include at least one lead-in row with Elo fields")

    df["normalized_result"] = normalize_weighted_form_result(df["result"], df["team_score"], df["opponent_score"])
    df = df.dropna(subset=["normalized_result"]).copy()
    if df.empty:
        raise ValueError("lead_in_df must include at least one rated lead-in row with a valid result")

    df["goal_difference"] = df["team_score"] - df["opponent_score"]
    df["gd_capped"] = df["goal_difference"].clip(
        lower=-WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
        upper=WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
    )
    df["elo_gap"] = df["opponent_elo_start"] - df["team_elo_start"]
    df["actual_score"] = df["normalized_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0}).astype(float)
    df["expected_score"] = compute_elo_expected_score(df["team_elo_start"], df["opponent_elo_start"]).astype(float)
    df["perf_vs_exp"] = df["actual_score"] - df["expected_score"]

    rows: list[dict[str, float | int | str]] = []
    grouped = df.sort_values(["qualified_team_id", "date", "lead_in_id"], kind="stable").groupby("qualified_team_id")
    for team_id, matches in grouped:
        recent = matches.tail(match_window).reset_index(drop=True).copy()
        if recent.empty:
            continue
        recent["recency_weight"] = np.arange(1, len(recent) + 1, dtype=float)
        total_weight = float(recent["recency_weight"].sum())
        rows.append(
            {
                "team_id": str(team_id),
                "recent_matches": int(len(recent)),
                "wins": int(recent["normalized_result"].eq("win").sum()),
                "draws": int(recent["normalized_result"].eq("draw").sum()),
                "losses": int(recent["normalized_result"].eq("loss").sum()),
                "goals_for": int(recent["team_score"].sum()),
                "goals_against": int(recent["opponent_score"].sum()),
                "avg_opp_elo": float((recent["opponent_elo_start"] * recent["recency_weight"]).sum() / total_weight),
                "avg_elo_gap": float((recent["elo_gap"] * recent["recency_weight"]).sum() / total_weight),
                "results_form": float((recent["actual_score"] * recent["recency_weight"]).sum() / total_weight),
                "gd_form": float((recent["gd_capped"] * recent["recency_weight"]).sum() / total_weight),
                "difficulty": float((recent["elo_gap"] * recent["recency_weight"]).sum() / total_weight),
                "expected_score": float((recent["expected_score"] * recent["recency_weight"]).sum() / total_weight),
                "perf_vs_exp": float((recent["perf_vs_exp"] * recent["recency_weight"]).sum() / total_weight),
                "elo_delta_form": float((recent["team_elo_delta"] * recent["recency_weight"]).sum() / total_weight),
            }
        )

    form_df = pd.DataFrame(rows)
    if form_df.empty:
        return form_df

    team_metadata_columns = [
        "team_id",
        "display_name",
        "flag_icon_code",
        "group_code",
        "confederation",
        "elo_rating",
        "world_rank",
        "world_cup_participations",
        "weighted_world_cup_participations",
        "weighted_world_cup_placement_score",
    ]
    available_team_metadata_columns = [column_name for column_name in team_metadata_columns if column_name in base_df.columns]
    team_metadata = base_df.loc[:, available_team_metadata_columns].drop_duplicates(subset=["team_id"], keep="first").copy()
    team_metadata["elo_rating"] = pd.to_numeric(team_metadata["elo_rating"], errors="coerce")
    team_metadata["world_rank"] = pd.to_numeric(team_metadata["world_rank"], errors="coerce")

    form_df = team_metadata.merge(form_df, on="team_id", how="left")
    fill_zero_columns = [
        "recent_matches",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "avg_opp_elo",
        "avg_elo_gap",
        "results_form",
        "gd_form",
        "difficulty",
        "expected_score",
        "perf_vs_exp",
        "elo_delta_form",
    ]
    for column_name in fill_zero_columns:
        form_df[column_name] = pd.to_numeric(form_df[column_name], errors="coerce").fillna(0.0)
    integer_columns = ["recent_matches", "wins", "draws", "losses", "goals_for", "goals_against"]
    for column_name in integer_columns:
        form_df[column_name] = form_df[column_name].astype(int)

    form_df["results_form_z"] = zscore(form_df["results_form"])
    form_df["gd_form_z"] = zscore(form_df["gd_form"])
    form_df["perf_vs_exp_z"] = zscore(form_df["perf_vs_exp"])
    form_df["elo_delta_form_z"] = zscore(form_df["elo_delta_form"])

    form_df["results_score"] = np.clip(form_df["results_form"], 0.0, 1.0)
    form_df["gd_score"] = clip_scale(form_df["gd_form"], *WEIGHTED_FORM_GD_BOUNDS)
    form_df["perf_score"] = clip_scale(form_df["perf_vs_exp"], *WEIGHTED_FORM_PERF_BOUNDS)
    form_df["elo_score"] = clip_scale(form_df["elo_delta_form"], *WEIGHTED_FORM_ELO_BOUNDS)
    form_df["form_index_0to1"] = (
        results_weight * form_df["results_score"]
        + gd_weight * form_df["gd_score"]
        + perf_weight * form_df["perf_score"]
        + elo_delta_weight * form_df["elo_score"]
    )
    form_df["form"] = 1.0 + 9.0 * form_df["form_index_0to1"]

    form_df["schedule_difficulty"] = scale_to_range(form_df["difficulty"]).round(1)
    form_df["avg_opp_elo"] = form_df["avg_opp_elo"].round(1)
    form_df["avg_elo_gap"] = form_df["avg_elo_gap"].round(1)
    form_df["results_form"] = form_df["results_form"].round(3)
    form_df["gd_form"] = form_df["gd_form"].round(3)
    form_df["difficulty"] = form_df["difficulty"].round(3)
    form_df["expected_score"] = form_df["expected_score"].round(3)
    form_df["perf_vs_exp"] = form_df["perf_vs_exp"].round(3)
    form_df["elo_delta_form"] = form_df["elo_delta_form"].round(3)
    form_df["results_form_z"] = form_df["results_form_z"].round(4)
    form_df["gd_form_z"] = form_df["gd_form_z"].round(4)
    form_df["perf_vs_exp_z"] = form_df["perf_vs_exp_z"].round(4)
    form_df["elo_delta_form_z"] = form_df["elo_delta_form_z"].round(4)
    form_df["results_score"] = form_df["results_score"].round(4)
    form_df["gd_score"] = form_df["gd_score"].round(4)
    form_df["perf_score"] = form_df["perf_score"].round(4)
    form_df["elo_score"] = form_df["elo_score"].round(4)
    form_df["form_index_0to1"] = form_df["form_index_0to1"].round(4)
    form_df["form"] = form_df["form"].round(4)

    return form_df.sort_values(
        ["form", "elo_rating", "world_rank"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_team_strengths(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    baseline_rating_weights: tuple[float, float] = BASELINE_RATING_WEIGHTS,
    form_component_weights: tuple[float, float] = FORM_COMPONENT_WEIGHTS,
    strength_blend_weights: tuple[float, float] = STRENGTH_BLEND_WEIGHTS,
) -> pd.DataFrame:
    """Blend ratings and recent form into one pre-tournament strength score."""
    elo_weight, fifa_weight = normalize_weight_pair(*baseline_rating_weights)
    rating_weight, form_weight = normalize_weight_pair(*strength_blend_weights)
    df = base_df.copy()
    df["elo_rating"] = pd.to_numeric(df["elo_rating"], errors="coerce")
    df["fifa_points"] = pd.to_numeric(df["fifa_points"], errors="coerce")
    df["rating_score"] = elo_weight * zscore(df["elo_rating"]) + fifa_weight * zscore(df["fifa_points"])

    form = build_recent_form_metrics(
        lead_in_df,
        match_window=match_window,
        form_component_weights=form_component_weights,
    )
    df = df.merge(form, left_on="team_id", right_on="qualified_team_id", how="left")
    df["recent_matches"] = df["recent_matches"].fillna(0).astype(int)
    df["points_per_match"] = df["points_per_match"].fillna(0.0)
    df["goal_diff_per_match"] = df["goal_diff_per_match"].fillna(0.0)
    df["points_form_z"] = df["points_form_z"].fillna(0.0)
    df["goal_diff_form_z"] = df["goal_diff_form_z"].fillna(0.0)
    df["form_score"] = df["form_score"].fillna(0.0)
    df["team_strength"] = rating_weight * df["rating_score"] + form_weight * df["form_score"]
    return df.drop(columns=["qualified_team_id"])


def build_v2_team_strengths(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    baseline_rating_weights: tuple[float, float] = BASELINE_RATING_WEIGHTS,
    form_composite_weights: tuple[float, float, float, float] = WEIGHTED_FORM_COMPOSITE_WEIGHTS,
    history_component_weights: tuple[float, float] = V2_HISTORY_COMPONENT_WEIGHTS,
    strength_blend_weights: tuple[float, float, float] = V2_STRENGTH_BLEND_WEIGHTS,
) -> pd.DataFrame:
    """Blend ratings, weighted recent form, and World Cup history into one V2 score."""
    elo_weight, fifa_weight = normalize_weight_pair(*baseline_rating_weights)
    history_placement_weight, history_participation_weight = normalize_weight_pair(*history_component_weights)
    blend_total = float(sum(strength_blend_weights))
    if blend_total <= 0:
        raise ValueError("strength_blend_weights must contain at least one positive value")
    rating_weight, form_weight, history_weight = tuple(
        float(weight) / blend_total for weight in strength_blend_weights
    )

    ratings_df = (
        base_df.loc[:, [column_name for column_name in ["team_id", "elo_rating", "fifa_points"] if column_name in base_df.columns]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .copy()
    )
    for column_name in ["elo_rating", "fifa_points"]:
        if column_name not in ratings_df.columns:
            ratings_df[column_name] = 0.0
    ratings_df["elo_rating"] = pd.to_numeric(ratings_df["elo_rating"], errors="coerce")
    ratings_df["fifa_points"] = pd.to_numeric(ratings_df["fifa_points"], errors="coerce")
    ratings_df["rating_score"] = elo_weight * zscore(ratings_df["elo_rating"]) + fifa_weight * zscore(ratings_df["fifa_points"])
    ratings_df["rating_index_0to1"] = scale_to_range(
        ratings_df["rating_score"],
        lower=0.0,
        upper=1.0,
        neutral=0.5,
    )

    df = build_weighted_form_table(
        base_df,
        lead_in_df,
        match_window=match_window,
        composite_weights=form_composite_weights,
    ).merge(
        ratings_df.loc[:, ["team_id", "fifa_points", "rating_score", "rating_index_0to1"]],
        on="team_id",
        how="left",
    )

    for column_name in ["weighted_world_cup_participations", "weighted_world_cup_placement_score", "world_cup_participations"]:
        if column_name not in df.columns:
            df[column_name] = 0.0
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)
    df["world_cup_participations"] = df["world_cup_participations"].astype(int)
    df["weighted_world_cup_participation_ratio"] = (
        df["weighted_world_cup_participations"] / WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT
    ).clip(lower=0.0, upper=1.0)
    df["history_score"] = (
        history_placement_weight * df["weighted_world_cup_placement_score"]
        + history_participation_weight * df["weighted_world_cup_participation_ratio"]
    )
    df["v2_strength_index_0to1"] = (
        rating_weight * df["rating_index_0to1"]
        + form_weight * df["form_index_0to1"]
        + history_weight * df["history_score"]
    )
    df["v2_strength"] = 1.0 + 9.0 * df["v2_strength_index_0to1"]

    df["rating_score"] = df["rating_score"].round(4)
    df["rating_index_0to1"] = df["rating_index_0to1"].round(4)
    df["weighted_world_cup_participations"] = df["weighted_world_cup_participations"].round(1)
    df["weighted_world_cup_placement_score"] = df["weighted_world_cup_placement_score"].round(4)
    df["weighted_world_cup_participation_ratio"] = df["weighted_world_cup_participation_ratio"].round(4)
    df["history_score"] = df["history_score"].round(4)
    df["v2_strength_index_0to1"] = df["v2_strength_index_0to1"].round(4)
    df["v2_strength"] = df["v2_strength"].round(4)

    return df.sort_values(
        ["v2_strength", "form", "elo_rating", "world_rank"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def normalize_historical_team_name(name: str | None) -> str:
    """Normalize a historical team label onto the repo's local naming conventions."""
    key = normalize_key(name)
    return HISTORICAL_TEAM_NAME_ALIASES.get(key, key)


def outcome_label_from_scoreline(home_score: int, away_score: int) -> str:
    """Convert a scoreline into the three-class v2 outcome label."""
    if int(home_score) > int(away_score):
        return "home_win"
    if int(home_score) < int(away_score):
        return "away_win"
    return "draw"


def match_stage_bucket(stage: str) -> str:
    """Bucket a World Cup stage into group-stage vs knockout for scoreline sampling."""
    return V2_STAGE_GROUP if str(stage).strip() == "Group Stage" else V2_STAGE_KNOCKOUT


def compute_history_placement_score(
    rank: int | None,
    n_teams: int,
    qualified: bool,
    epsilon: float = 0.05,
    gamma: float = 0.8,
) -> float:
    """Compute the weighted placement score used by the v2 history feature."""
    if not qualified or rank is None or n_teams <= 1:
        return 0.0
    if int(rank) <= 1:
        return 1.0
    scale = (int(rank) - 1) / (int(n_teams) - 1)
    return float(epsilon + (1.0 - epsilon) * (1.0 - scale**gamma))


def parse_country_results_file(path: Path) -> pd.DataFrame:
    """Load one country results file with normalized date and numeric columns."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    parsed = df.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    for column_name in (
        "team_score",
        "opponent_score",
        "team_elo_start",
        "opponent_elo_start",
        "team_elo_end",
        "opponent_elo_end",
        "team_elo_delta",
    ):
        if column_name in parsed.columns:
            parsed[column_name] = pd.to_numeric(parsed[column_name], errors="coerce")
    return parsed.sort_values(["date"], kind="stable").reset_index(drop=True)


def load_historical_country_results_lookup() -> dict[str, pd.DataFrame]:
    """Index the per-country result files used for historical pre-tournament features."""
    root = WORLD_CUP_ROOT / "by_confederation"
    lookup: dict[str, pd.DataFrame] = {}
    if not root.exists():
        return lookup

    for results_path in root.glob("*/*/results.csv"):
        parsed = parse_country_results_file(results_path)
        key_candidates = {normalize_historical_team_name(results_path.parent.name.replace("_", " "))}
        if not parsed.empty and "team" in parsed.columns:
            team_values = parsed["team"].dropna().astype(str)
            if not team_values.empty:
                key_candidates.add(normalize_historical_team_name(team_values.iloc[0]))
        for key in key_candidates:
            if key and key not in lookup:
                lookup[key] = parsed
    return lookup


def latest_pre_tournament_elo(results_df: pd.DataFrame) -> float:
    """Return the latest available Elo rating before a tournament starts."""
    if results_df.empty:
        return 0.0
    for column_name in ("team_elo_end", "team_elo_start"):
        if column_name not in results_df.columns:
            continue
        valid = pd.to_numeric(results_df[column_name], errors="coerce").dropna()
        if not valid.empty:
            return float(valid.iloc[-1])
    return 0.0


def compute_weighted_form_snapshot(results_df: pd.DataFrame, match_window: int = RECENT_MATCH_WINDOW) -> dict[str, float]:
    """Summarize the last k Elo-rated matches into raw pre-tournament team features."""
    if match_window <= 0:
        raise ValueError("match_window must be positive")
    if results_df.empty:
        return {
            "results_form": 0.0,
            "gd_form": 0.0,
            "perf_vs_exp": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "pre_tournament_elo": 0.0,
        }

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pre_tournament_elo"] = latest_pre_tournament_elo(df)
    for column_name in ("team_score", "opponent_score", "team_elo_start", "opponent_elo_start"):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    if "team_elo_delta" in df.columns:
        df["team_elo_delta"] = pd.to_numeric(df["team_elo_delta"], errors="coerce")

    df = df.dropna(subset=["team_score", "opponent_score", "team_elo_start", "opponent_elo_start"]).copy()
    if df.empty:
        return {
            "results_form": 0.0,
            "gd_form": 0.0,
            "perf_vs_exp": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "pre_tournament_elo": float(results_df.attrs.get("pre_tournament_elo", 0.0)),
        }

    df["normalized_result"] = normalize_weighted_form_result(df["result"], df["team_score"], df["opponent_score"])
    df = df.dropna(subset=["normalized_result"]).sort_values(["date"], kind="stable").tail(match_window).reset_index(drop=True)
    if df.empty:
        return {
            "results_form": 0.0,
            "gd_form": 0.0,
            "perf_vs_exp": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "pre_tournament_elo": 0.0,
        }

    df["recency_weight"] = np.arange(1, len(df) + 1, dtype=float)
    total_weight = float(df["recency_weight"].sum())
    df["goal_difference"] = df["team_score"] - df["opponent_score"]
    df["gd_capped"] = df["goal_difference"].clip(
        lower=-WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
        upper=WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
    )
    df["actual_score"] = df["normalized_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0}).astype(float)
    df["expected_score"] = compute_elo_expected_score(df["team_elo_start"], df["opponent_elo_start"]).astype(float)
    df["perf_vs_exp"] = df["actual_score"] - df["expected_score"]

    return {
        "results_form": float((df["actual_score"] * df["recency_weight"]).sum() / total_weight),
        "gd_form": float((df["gd_capped"] * df["recency_weight"]).sum() / total_weight),
        "perf_vs_exp": float((df["perf_vs_exp"] * df["recency_weight"]).sum() / total_weight),
        "goals_for": float((df["team_score"] * df["recency_weight"]).sum() / total_weight),
        "goals_against": float((df["opponent_score"] * df["recency_weight"]).sum() / total_weight),
        "pre_tournament_elo": latest_pre_tournament_elo(df),
    }


def build_weighted_form_feature_lookup(
    results_df: pd.DataFrame,
    team_key_column: str,
    match_window: int = RECENT_MATCH_WINDOW,
) -> dict[str, dict[str, float]]:
    """Build weighted raw feature snapshots keyed by team id/name."""
    if team_key_column not in results_df.columns:
        raise ValueError(f"results_df is missing {team_key_column}")
    lookup: dict[str, dict[str, float]] = {}
    grouped = results_df.sort_values(["date"], kind="stable").groupby(team_key_column, dropna=False)
    for team_key, matches in grouped:
        lookup[str(team_key)] = compute_weighted_form_snapshot(matches, match_window=match_window)
    return lookup


def load_historical_world_cup_results() -> pd.DataFrame:
    """Load the historical match results used to train the v2 multinomial model."""
    rows: list[pd.DataFrame] = []
    for year in range(HISTORICAL_RESULTS_START_YEAR, HISTORICAL_RESULTS_END_YEAR + 1):
        path = WORLD_CUP_ROOT / str(year) / "results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        parsed = df.copy()
        parsed["edition"] = year
        parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
        for column_name in ("home_score", "away_score", "home_elo_start", "away_elo_start"):
            parsed[column_name] = pd.to_numeric(parsed[column_name], errors="coerce")
        parsed["home_team_key"] = parsed["home_team"].map(normalize_historical_team_name)
        parsed["away_team_key"] = parsed["away_team"].map(normalize_historical_team_name)
        parsed["stage_bucket"] = parsed["stage"].map(match_stage_bucket)
        rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_historical_placement_history() -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Load placement history plus global edition weights and team counts."""
    placement_df = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "placement.csv")
    history_df = pd.read_csv(WORLD_CUP_ROOT / "fifa_world_cup_history.csv")
    placement_df["edition"] = pd.to_numeric(placement_df["edition"], errors="coerce").astype(int)
    placement_df["position"] = pd.to_numeric(placement_df["position"], errors="coerce")
    if "start_elo" not in placement_df.columns:
        placement_df["start_elo"] = np.nan
    placement_df["start_elo"] = pd.to_numeric(placement_df["start_elo"], errors="coerce")
    placement_df["team_key"] = placement_df["country"].map(normalize_historical_team_name)
    edition_years = sorted(pd.to_numeric(history_df["Year"], errors="coerce").dropna().astype(int).tolist())
    edition_weight_map = {edition: (index + 1) ** 2 for index, edition in enumerate(edition_years)}
    edition_team_counts = {
        int(year): int(teams)
        for year, teams in zip(
            pd.to_numeric(history_df["Year"], errors="coerce"),
            pd.to_numeric(history_df["Teams"], errors="coerce"),
            strict=False,
        )
        if pd.notna(year) and pd.notna(teams)
    }
    return placement_df, edition_team_counts, edition_weight_map


def compute_pre_tournament_history_features(
    team_key: str,
    edition_year: int,
    placement_df: pd.DataFrame,
    edition_team_counts: dict[int, int],
    edition_weight_map: dict[int, int],
) -> dict[str, float]:
    """Compute appearance and weighted placement features using only earlier editions."""
    earlier_editions = [year for year in sorted(edition_weight_map) if year < int(edition_year)]
    if not earlier_editions:
        return {"placement": 0.0, "appearance": 0.0}

    team_rows = placement_df[(placement_df["team_key"] == team_key) & (placement_df["edition"] < int(edition_year))]
    position_by_edition = {
        int(row.edition): int(row.position)
        for row in team_rows.dropna(subset=["position"]).drop_duplicates(subset=["edition"], keep="first").itertuples(index=False)
    }

    weighted_total = 0.0
    total_weight = float(sum(edition_weight_map[year] for year in earlier_editions))
    for prior_edition in earlier_editions:
        rank = position_by_edition.get(prior_edition)
        qualified = rank is not None
        n_teams = max(int(edition_team_counts.get(prior_edition, 32)), int(rank or 0))
        weighted_total += float(edition_weight_map[prior_edition]) * compute_history_placement_score(
            rank=rank,
            n_teams=n_teams,
            qualified=qualified,
        )

    return {
        "placement": weighted_total / total_weight if total_weight else 0.0,
        "appearance": float(len(position_by_edition)),
    }


def build_pre_tournament_team_features_by_edition(
    edition_matches: pd.DataFrame,
    country_results_lookup: dict[str, pd.DataFrame],
    placement_df: pd.DataFrame,
    edition_team_counts: dict[int, int],
    edition_weight_map: dict[int, int],
    match_window: int = RECENT_MATCH_WINDOW,
) -> dict[str, dict[str, float]]:
    """Build raw team features for every participant in one historical edition."""
    edition_year = int(edition_matches["edition"].iloc[0])
    edition_start = pd.to_datetime(edition_matches["date"], errors="coerce").min()
    edition_placement_path = WORLD_CUP_ROOT / str(edition_year) / "placement.csv"
    if edition_placement_path.exists():
        placement_rows = pd.read_csv(edition_placement_path)
        if "start_elo" not in placement_rows.columns:
            placement_rows["start_elo"] = np.nan
    else:
        placement_rows = placement_df[placement_df["edition"] == edition_year].copy()
    placement_rows["start_elo"] = pd.to_numeric(placement_rows["start_elo"], errors="coerce")
    placement_elo_lookup = {
        str(row.country): float(row.start_elo)
        for row in placement_rows.dropna(subset=["start_elo"]).itertuples(index=False)
    }

    team_names = sorted(
        {
            str(team_name)
            for team_name in pd.concat([edition_matches["home_team"], edition_matches["away_team"]]).dropna().unique()
        }
    )
    features: dict[str, dict[str, float]] = {}
    for team_name in team_names:
        team_key = normalize_historical_team_name(team_name)
        team_results = country_results_lookup.get(team_key, pd.DataFrame()).copy()
        if not team_results.empty:
            team_results = team_results[team_results["date"] < edition_start].copy()
        snapshot = compute_weighted_form_snapshot(team_results, match_window=match_window)
        history_features = compute_pre_tournament_history_features(
            team_key,
            edition_year,
            placement_df,
            edition_team_counts,
            edition_weight_map,
        )
        pre_tournament_elo = snapshot["pre_tournament_elo"]
        if pre_tournament_elo == 0.0:
            pre_tournament_elo = float(placement_elo_lookup.get(team_name, 0.0))
        features[team_name] = {
            **snapshot,
            **history_features,
            "pre_tournament_elo": pre_tournament_elo,
        }
    return features


def build_v2_training_frame(match_window: int = RECENT_MATCH_WINDOW) -> pd.DataFrame:
    """Build the historical match-level training frame for the v2 multinomial model."""
    historical_results = load_historical_world_cup_results()
    if historical_results.empty:
        raise ValueError("Historical World Cup results are unavailable for v2 training")

    country_results_lookup = load_historical_country_results_lookup()
    placement_df, edition_team_counts, edition_weight_map = load_historical_placement_history()
    rows: list[dict[str, object]] = []
    for edition_year, edition_matches in historical_results.groupby("edition", sort=True):
        team_features = build_pre_tournament_team_features_by_edition(
            edition_matches,
            country_results_lookup,
            placement_df,
            edition_team_counts,
            edition_weight_map,
            match_window=match_window,
        )
        for match in edition_matches.itertuples(index=False):
            if pd.isna(match.home_score) or pd.isna(match.away_score):
                continue
            home_features = team_features.get(str(match.home_team), {})
            away_features = team_features.get(str(match.away_team), {})
            home_elo = float(match.home_elo_start) if pd.notna(match.home_elo_start) else float(home_features.get("pre_tournament_elo", 0.0))
            away_elo = float(match.away_elo_start) if pd.notna(match.away_elo_start) else float(away_features.get("pre_tournament_elo", 0.0))
            rows.append(
                {
                    "edition": int(edition_year),
                    "stage": str(match.stage),
                    "stage_bucket": str(match.stage_bucket),
                    "home_team": str(match.home_team),
                    "away_team": str(match.away_team),
                    "home_score": int(match.home_score),
                    "away_score": int(match.away_score),
                    "outcome_label": outcome_label_from_scoreline(int(match.home_score), int(match.away_score)),
                    "elo_diff": home_elo - away_elo,
                    "results_form_diff": float(home_features.get("results_form", 0.0)) - float(away_features.get("results_form", 0.0)),
                    "gd_form_diff": float(home_features.get("gd_form", 0.0)) - float(away_features.get("gd_form", 0.0)),
                    "perf_vs_exp_diff": float(home_features.get("perf_vs_exp", 0.0)) - float(away_features.get("perf_vs_exp", 0.0)),
                    "goals_for_diff": float(home_features.get("goals_for", 0.0)) - float(away_features.get("goals_for", 0.0)),
                    "goals_against_diff": float(home_features.get("goals_against", 0.0)) - float(away_features.get("goals_against", 0.0)),
                    "placement_diff": float(home_features.get("placement", 0.0)) - float(away_features.get("placement", 0.0)),
                    "appearance_diff": float(home_features.get("appearance", 0.0)) - float(away_features.get("appearance", 0.0)),
                }
            )

    training_df = pd.DataFrame(rows)
    if training_df.empty:
        raise ValueError("Historical World Cup training frame is empty")
    return training_df


def build_v2_scoreline_distributions(training_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, np.ndarray | list[tuple[int, int]]]]:
    """Build empirical scoreline samplers by stage bucket and outcome class."""
    distributions: dict[tuple[str, str], dict[str, np.ndarray | list[tuple[int, int]]]] = {}
    for stage_bucket in (V2_STAGE_GROUP, V2_STAGE_KNOCKOUT):
        for outcome_label in V2_OUTCOME_LABELS:
            subset = training_df[
                (training_df["stage_bucket"] == stage_bucket) & (training_df["outcome_label"] == outcome_label)
            ]
            if subset.empty:
                subset = training_df[training_df["outcome_label"] == outcome_label]
            if subset.empty:
                scorelines = [DEFAULT_SCORELINE_BY_OUTCOME[outcome_label]]
                probabilities = np.array([1.0], dtype=float)
            else:
                counts = subset.groupby(["home_score", "away_score"]).size().sort_values(ascending=False)
                scorelines = [(int(home_score), int(away_score)) for home_score, away_score in counts.index]
                probabilities = (counts / counts.sum()).to_numpy(dtype=float)
            distributions[(stage_bucket, outcome_label)] = {
                "scorelines": scorelines,
                "probabilities": probabilities,
            }
    return distributions


@lru_cache(maxsize=8)
def fit_v2_match_multinomial_model(match_window: int = RECENT_MATCH_WINDOW) -> dict[str, object]:
    """Fit the sklearn multinomial model plus empirical scoreline samplers for v2."""
    training_df = build_v2_training_frame(match_window=match_window)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
        raise ImportError("scikit-learn is required for the v2 multinomial simulator") from exc

    X = training_df.loc[:, list(V2_FEATURE_COLUMNS)].astype(float)
    y = training_df["outcome_label"].astype(str)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=5000,
        random_state=20260403,
    )
    model.fit(X_scaled, y)
    return {
        "training_frame": training_df,
        "feature_columns": V2_FEATURE_COLUMNS,
        "scaler": scaler,
        "model": model,
        "scoreline_distributions": build_v2_scoreline_distributions(training_df),
    }


def build_v2_match_feature_table(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
) -> pd.DataFrame:
    """Build the current-team feature table consumed by the v2 matchup model."""
    team_features = build_v2_team_strengths(base_df, lead_in_df, match_window=match_window).copy()
    form_lookup = build_weighted_form_feature_lookup(lead_in_df, "qualified_team_id", match_window=match_window)

    feature_rows = []
    for row in team_features.itertuples(index=False):
        form_snapshot = form_lookup.get(str(row.team_id), {})
        appearance_count = max(int(getattr(row, "world_cup_participations", 1)) - 1, 0)
        feature_rows.append(
            {
                "team_id": str(row.team_id),
                "display_name": str(row.display_name),
                "flag_icon_code": str(row.flag_icon_code) if pd.notna(row.flag_icon_code) else "",
                "group_code": str(row.group_code),
                "confederation": str(row.confederation),
                "elo_rating": float(pd.to_numeric(getattr(row, "elo_rating", 0.0), errors="coerce") or 0.0),
                "team_strength": float(getattr(row, "v2_strength", 0.0)),
                "v2_strength": float(getattr(row, "v2_strength", 0.0)),
                "results_form": float(form_snapshot.get("results_form", 0.0)),
                "gd_form": float(form_snapshot.get("gd_form", 0.0)),
                "perf_vs_exp": float(form_snapshot.get("perf_vs_exp", 0.0)),
                "goals_for": float(form_snapshot.get("goals_for", 0.0)),
                "goals_against": float(form_snapshot.get("goals_against", 0.0)),
                "placement": float(pd.to_numeric(getattr(row, "weighted_world_cup_placement_score", 0.0), errors="coerce") or 0.0),
                "appearance": float(appearance_count),
            }
        )

    return pd.DataFrame(feature_rows).sort_values(
        ["team_strength", "elo_rating"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def predict_match_probabilities_v2(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
) -> dict[str, float | str]:
    """Predict a single 2026 matchup using the v2 multinomial model."""
    home_features = team_feature_lookup[str(home_team_id)]
    away_features = team_feature_lookup[str(away_team_id)]
    feature_row = pd.DataFrame(
        [
            {
                "elo_diff": float(home_features["elo_rating"]) - float(away_features["elo_rating"]),
                "results_form_diff": float(home_features["results_form"]) - float(away_features["results_form"]),
                "gd_form_diff": float(home_features["gd_form"]) - float(away_features["gd_form"]),
                "perf_vs_exp_diff": float(home_features["perf_vs_exp"]) - float(away_features["perf_vs_exp"]),
                "goals_for_diff": float(home_features["goals_for"]) - float(away_features["goals_for"]),
                "goals_against_diff": float(home_features["goals_against"]) - float(away_features["goals_against"]),
                "placement_diff": float(home_features["placement"]) - float(away_features["placement"]),
                "appearance_diff": float(home_features["appearance"]) - float(away_features["appearance"]),
            }
        ],
        columns=list(model_bundle["feature_columns"]),
    )
    scaled = model_bundle["scaler"].transform(feature_row)
    probability_values = model_bundle["model"].predict_proba(scaled)[0]
    probability_map = {
        str(label): float(probability)
        for label, probability in zip(model_bundle["model"].classes_, probability_values, strict=False)
    }
    return {
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home_win_prob": probability_map.get("home_win", 0.0),
        "draw_prob": probability_map.get("draw", 0.0),
        "away_win_prob": probability_map.get("away_win", 0.0),
    }


def sample_scoreline_v2(
    stage_bucket: str,
    outcome_label: str,
    model_bundle: dict[str, object],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample a scoreline from the empirical historical distribution for one outcome class."""
    distribution = model_bundle["scoreline_distributions"][(stage_bucket, outcome_label)]
    scorelines = distribution["scorelines"]
    probabilities = distribution["probabilities"]
    choice = int(rng.choice(len(scorelines), p=probabilities))
    scoreline = scorelines[choice]
    return int(scoreline[0]), int(scoreline[1])


def simulate_knockout_match_v2(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    rng: np.random.Generator,
    matchup_probability_cache: dict[tuple[str, str], dict[str, float | str]] | None = None,
) -> tuple[str, str]:
    """Simulate one knockout match using the v2 outcome model and empirical scorelines."""
    cache_key = (str(home_team_id), str(away_team_id))
    if matchup_probability_cache is not None and cache_key in matchup_probability_cache:
        probabilities = matchup_probability_cache[cache_key]
    else:
        probabilities = predict_match_probabilities_v2(
            home_team_id,
            away_team_id,
            team_feature_lookup,
            model_bundle,
        )
        if matchup_probability_cache is not None:
            matchup_probability_cache[cache_key] = probabilities

    outcome_label = str(
        rng.choice(
            np.array(V2_OUTCOME_LABELS, dtype=object),
            p=[
                float(probabilities["home_win_prob"]),
                float(probabilities["draw_prob"]),
                float(probabilities["away_win_prob"]),
            ],
        )
    )
    if outcome_label == "home_win":
        return str(home_team_id), str(away_team_id)
    if outcome_label == "away_win":
        return str(away_team_id), str(home_team_id)

    sample_scoreline_v2(V2_STAGE_KNOCKOUT, "draw", model_bundle, rng)
    home_non_draw = float(probabilities["home_win_prob"])
    away_non_draw = float(probabilities["away_win_prob"])
    non_draw_total = home_non_draw + away_non_draw
    if non_draw_total <= 0:
        home_wins_penalties = bool(rng.integers(0, 2))
    else:
        home_wins_penalties = bool(rng.random() < (home_non_draw / non_draw_total))
    winner_team_id = str(home_team_id) if home_wins_penalties else str(away_team_id)
    loser_team_id = str(away_team_id) if home_wins_penalties else str(home_team_id)
    return winner_team_id, loser_team_id


def predict_knockout_matchup_v2(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, float | str]:
    """Estimate one knockout matchup with the v2 multinomial model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)
    matchup_probability_cache: dict[tuple[str, str], dict[str, float | str]] = {}
    home_wins = 0
    for _ in range(simulations):
        winner_team_id, _ = simulate_knockout_match_v2(
            home_team_id,
            away_team_id,
            team_feature_lookup,
            model_bundle,
            rng,
            matchup_probability_cache=matchup_probability_cache,
        )
        if winner_team_id == home_team_id:
            home_wins += 1

    home_win_prob = home_wins / simulations * 100.0
    away_win_prob = 100.0 - home_win_prob
    if home_win_prob >= away_win_prob:
        winner_team_id = home_team_id
        winner_win_prob = home_win_prob
    else:
        winner_team_id = away_team_id
        winner_win_prob = away_win_prob
    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "winner_team_id": winner_team_id,
        "winner_win_prob": winner_win_prob,
    }


def _head_to_head_stats(
    tied_indices: list[int],
    fixture_pairs: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    team_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute head-to-head points, goal difference, and goals scored for tied teams."""
    tied_set = set(tied_indices)
    h2h_points = np.zeros(team_count, dtype=int)
    h2h_goal_diff = np.zeros(team_count, dtype=int)
    h2h_goals_for = np.zeros(team_count, dtype=int)

    for match_index, (home_idx, away_idx) in enumerate(fixture_pairs):
        if home_idx not in tied_set or away_idx not in tied_set:
            continue

        home_score = int(home_goals[match_index])
        away_score = int(away_goals[match_index])
        h2h_goals_for[home_idx] += home_score
        h2h_goals_for[away_idx] += away_score
        h2h_goal_diff[home_idx] += home_score - away_score
        h2h_goal_diff[away_idx] += away_score - home_score

        if home_score > away_score:
            h2h_points[home_idx] += 3
        elif home_score < away_score:
            h2h_points[away_idx] += 3
        else:
            h2h_points[home_idx] += 1
            h2h_points[away_idx] += 1

    return h2h_points, h2h_goal_diff, h2h_goals_for


def _rank_group_indices(
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    fixture_pairs: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    team_strength: np.ndarray,
) -> list[int]:
    """Rank a four-team group using overall stats, head-to-head, then strength fallback."""
    goal_difference = goals_for - goals_against
    sorted_indices = sorted(
        range(len(points)),
        key=lambda idx: (int(points[idx]), int(goal_difference[idx]), int(goals_for[idx])),
        reverse=True,
    )

    ranked_indices: list[int] = []
    cursor = 0
    while cursor < len(sorted_indices):
        tied_indices = [sorted_indices[cursor]]
        cursor += 1

        while cursor < len(sorted_indices):
            current_idx = sorted_indices[cursor]
            previous_idx = tied_indices[0]
            current_tuple = (int(points[current_idx]), int(goal_difference[current_idx]), int(goals_for[current_idx]))
            previous_tuple = (int(points[previous_idx]), int(goal_difference[previous_idx]), int(goals_for[previous_idx]))
            if current_tuple != previous_tuple:
                break
            tied_indices.append(current_idx)
            cursor += 1

        if len(tied_indices) == 1:
            ranked_indices.extend(tied_indices)
            continue

        h2h_points, h2h_goal_diff, h2h_goals_for = _head_to_head_stats(
            tied_indices, fixture_pairs, home_goals, away_goals, len(points)
        )
        ranked_indices.extend(
            sorted(
                tied_indices,
                key=lambda idx: (
                    int(h2h_points[idx]),
                    int(h2h_goal_diff[idx]),
                    int(h2h_goals_for[idx]),
                    float(team_strength[idx]),
                ),
                reverse=True,
            )
        )

    return ranked_indices


def rank_best_third_place_teams(
    table_df: pd.DataFrame,
    qualification_slots: int = BEST_THIRD_QUALIFICATION_SLOTS,
    strength_column: str = "team_strength",
) -> pd.DataFrame:
    """Rank third-place teams across groups and flag the best qualifiers."""
    table = table_df.copy().reset_index(drop=True)
    table["points"] = pd.to_numeric(table["points"], errors="coerce").fillna(0).astype(int)
    table["goal_difference"] = pd.to_numeric(table["goal_difference"], errors="coerce").fillna(0).astype(int)
    table["goals_for"] = pd.to_numeric(table["goals_for"], errors="coerce").fillna(0).astype(int)
    table[strength_column] = pd.to_numeric(table[strength_column], errors="coerce").fillna(0.0)

    ranked = table.sort_values(
        ["points", "goal_difference", "goals_for", strength_column],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked["best_third_rank"] = np.arange(1, len(ranked) + 1)
    ranked["qualifies_as_best_third"] = ranked["best_third_rank"] <= min(qualification_slots, len(ranked))
    return ranked


def rank_group_standings(
    table_df: pd.DataFrame,
    fixture_results_df: pd.DataFrame,
    strength_column: str = "team_strength",
) -> pd.DataFrame:
    """Rank one group table using the simulator's tie-break logic."""
    table = table_df.copy().reset_index(drop=True)
    table["points"] = pd.to_numeric(table["points"], errors="coerce").fillna(0).astype(int)
    table["goals_for"] = pd.to_numeric(table["goals_for"], errors="coerce").fillna(0).astype(int)
    table["goals_against"] = pd.to_numeric(table["goals_against"], errors="coerce").fillna(0).astype(int)
    table[strength_column] = pd.to_numeric(table[strength_column], errors="coerce").fillna(0.0)
    table["goal_difference"] = table["goals_for"] - table["goals_against"]

    team_index = {team_id: idx for idx, team_id in enumerate(table["team_id"])}
    fixture_pairs = np.array(
        [
            (team_index[row.home_team_id], team_index[row.away_team_id])
            for row in fixture_results_df.itertuples(index=False)
        ],
        dtype=int,
    )
    home_goals = pd.to_numeric(fixture_results_df["home_goals"], errors="coerce").fillna(0).to_numpy(dtype=int)
    away_goals = pd.to_numeric(fixture_results_df["away_goals"], errors="coerce").fillna(0).to_numpy(dtype=int)

    ranked_indices = _rank_group_indices(
        points=table["points"].to_numpy(dtype=int),
        goals_for=table["goals_for"].to_numpy(dtype=int),
        goals_against=table["goals_against"].to_numpy(dtype=int),
        fixture_pairs=fixture_pairs,
        home_goals=home_goals,
        away_goals=away_goals,
        team_strength=table[strength_column].to_numpy(dtype=float),
    )
    return table.iloc[ranked_indices].reset_index(drop=True)


def expected_goals_from_strengths(home_strength: float, away_strength: float) -> tuple[float, float]:
    """Return expected goals for one match using the shared strength model."""
    delta = float(home_strength) - float(away_strength)
    home_xg = float(np.clip(EXPECTED_GOALS_BASE + EXPECTED_GOALS_SCALE * delta, EXPECTED_GOALS_MIN, EXPECTED_GOALS_MAX))
    away_xg = float(np.clip(EXPECTED_GOALS_BASE - EXPECTED_GOALS_SCALE * delta, EXPECTED_GOALS_MIN, EXPECTED_GOALS_MAX))
    return home_xg, away_xg


def resolve_knockout_slot(
    slot_label: str,
    match_number: int,
    group_rankings: dict[str, list[str]],
    match_results: dict[int, dict[str, str]],
    third_place_routing: dict[int, str],
) -> str:
    """Resolve a knockout slot label to a concrete team id for one simulation run."""
    slot = str(slot_label).strip()
    if not slot:
        raise ValueError(f"Blank slot label for knockout match {match_number}")

    if slot.startswith("W"):
        prior_match = int(slot[1:])
        return match_results[prior_match]["winner_team_id"]
    if slot.startswith("RU"):
        prior_match = int(slot[2:])
        return match_results[prior_match]["loser_team_id"]
    if slot[0] in {"1", "2"} and len(slot) == 2 and slot[1].isalpha():
        return group_rankings[slot[1]][int(slot[0]) - 1]
    if slot.startswith("3"):
        if len(slot) == 2 and slot[1].isalpha():
            return group_rankings[slot[1]][2]
        resolved_group = third_place_routing[match_number]
        return group_rankings[resolved_group][2]

    raise ValueError(f"Unsupported knockout slot label '{slot}' for match {match_number}")


def simulate_knockout_match(
    home_team_id: str,
    away_team_id: str,
    team_strength_lookup: dict[str, float],
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Simulate one knockout match to a winner, including extra time and penalties."""
    home_xg, away_xg = expected_goals_from_strengths(
        team_strength_lookup[home_team_id],
        team_strength_lookup[away_team_id],
    )
    home_goals = int(rng.poisson(home_xg))
    away_goals = int(rng.poisson(away_xg))

    if home_goals == away_goals:
        home_goals += int(rng.poisson(home_xg * EXTRA_TIME_FACTOR))
        away_goals += int(rng.poisson(away_xg * EXTRA_TIME_FACTOR))
    if home_goals == away_goals:
        home_wins = bool(rng.integers(0, 2))
        winner_team_id = home_team_id if home_wins else away_team_id
    else:
        winner_team_id = home_team_id if home_goals > away_goals else away_team_id
    loser_team_id = away_team_id if winner_team_id == home_team_id else home_team_id
    return winner_team_id, loser_team_id


def stable_seed_from_tokens(*tokens: object, base_seed: int = 20260403) -> int:
    """Build a deterministic integer seed from one or more tokens."""
    modulus = np.iinfo(np.uint32).max
    seed_value = int(base_seed) % modulus
    for token in tokens:
        for character in str(token):
            seed_value = (seed_value * 131 + ord(character)) % modulus
    return seed_value


def get_modal_group_rankings(simulation_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return the most common full finishing order for each group from simulator metadata."""
    modal_group_rankings = simulation_df.attrs.get("modal_group_rankings")
    if not modal_group_rankings:
        raise ValueError("simulation_df is missing modal_group_rankings metadata")
    return {group_code: list(team_ids) for group_code, team_ids in modal_group_rankings.items()}


def get_average_third_place_stats(simulation_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return average third-place stats accumulated during simulation runs."""
    average_stats = simulation_df.attrs.get("average_third_place_stats")
    if not average_stats:
        raise ValueError("simulation_df is missing average_third_place_stats metadata")
    return {
        team_id: {
            "points": float(stats["points"]),
            "goal_difference": float(stats["goal_difference"]),
            "goals_for": float(stats["goals_for"]),
            "team_strength": float(stats["team_strength"]),
        }
        for team_id, stats in average_stats.items()
    }


def predict_knockout_matchup(
    home_team_id: str,
    away_team_id: str,
    team_strength_lookup: dict[str, float],
    simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, float | str]:
    """Estimate one knockout matchup and return the likely winner plus win probabilities."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)
    home_wins = 0
    for _ in range(simulations):
        winner_team_id, _ = simulate_knockout_match(
            home_team_id,
            away_team_id,
            team_strength_lookup,
            rng,
        )
        if winner_team_id == home_team_id:
            home_wins += 1

    home_win_prob = home_wins / simulations * 100
    away_win_prob = 100.0 - home_win_prob
    if home_win_prob > away_win_prob:
        winner_team_id = home_team_id
        winner_win_prob = home_win_prob
    elif away_win_prob > home_win_prob:
        winner_team_id = away_team_id
        winner_win_prob = away_win_prob
    else:
        home_strength = float(team_strength_lookup[home_team_id])
        away_strength = float(team_strength_lookup[away_team_id])
        winner_team_id = home_team_id if home_strength >= away_strength else away_team_id
        winner_win_prob = 50.0

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "winner_team_id": winner_team_id,
        "winner_win_prob": winner_win_prob,
    }


def build_deterministic_bracket(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable knockout bracket from modal group rankings and matchup win probabilities."""
    modal_group_rankings = get_modal_group_rankings(simulation_df)
    average_third_place_stats = get_average_third_place_stats(simulation_df)
    main_bracket_fixtures = extract_main_bracket_fixtures(fixtures_df)
    team_strength_lookup = simulation_df.set_index("team_id")["team_strength"].astype(float).to_dict()

    third_place_rows = []
    for group_code, ranked_team_ids in modal_group_rankings.items():
        third_team_id = ranked_team_ids[2]
        average_stats = average_third_place_stats.get(
            third_team_id,
            {
                "points": 0.0,
                "goal_difference": 0.0,
                "goals_for": 0.0,
                "team_strength": team_strength_lookup[third_team_id],
            },
        )
        third_place_rows.append(
            {
                "team_id": third_team_id,
                "group_code": group_code,
                "points": average_stats["points"],
                "goal_difference": average_stats["goal_difference"],
                "goals_for": average_stats["goals_for"],
                "team_strength": average_stats["team_strength"],
            }
        )

    ranked_third_place = rank_best_third_place_teams(pd.DataFrame(third_place_rows))
    qualifying_third_place = ranked_third_place[ranked_third_place["qualifies_as_best_third"]].copy()
    qualifying_groups = "".join(sorted(qualifying_third_place["group_code"].astype(str).tolist()))
    if qualifying_groups not in THIRD_PLACE_ROUTING_MAP:
        raise ValueError(f"Missing Round of 32 routing for third-place combination {qualifying_groups}")
    third_place_routing = THIRD_PLACE_ROUTING_MAP[qualifying_groups]

    match_results: dict[int, dict[str, str]] = {}
    round_matches: dict[str, list[dict[str, object]]] = {round_code: [] for round_code in MAIN_BRACKET_ROUND_CODES}
    for match in main_bracket_fixtures.itertuples(index=False):
        match_number = int(match.match_number)
        home_team_id = resolve_knockout_slot(
            match.home_slot_label,
            match_number,
            modal_group_rankings,
            match_results,
            third_place_routing,
        )
        away_team_id = resolve_knockout_slot(
            match.away_slot_label,
            match_number,
            modal_group_rankings,
            match_results,
            third_place_routing,
        )
        prediction = predict_knockout_matchup(
            home_team_id,
            away_team_id,
            team_strength_lookup,
            simulations=head_to_head_simulations,
            seed=stable_seed_from_tokens(seed, match_number, home_team_id, away_team_id),
        )
        winner_team_id = str(prediction["winner_team_id"])
        loser_team_id = away_team_id if winner_team_id == home_team_id else home_team_id
        match_results[match_number] = {
            "winner_team_id": winner_team_id,
            "loser_team_id": loser_team_id,
        }
        round_matches[match.round_code].append(
            {
                "match_number": match_number,
                "round_code": match.round_code,
                "round_label": ROUND_CODE_LABELS[match.round_code],
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "winner_team_id": winner_team_id,
                "winner_win_prob": float(prediction["winner_win_prob"]),
                "home_win_prob": float(prediction["home_win_prob"]),
                "away_win_prob": float(prediction["away_win_prob"]),
            }
        )

    return {
        "modal_group_rankings": modal_group_rankings,
        "qualifying_third_place_team_ids": qualifying_third_place["team_id"].astype(str).tolist(),
        "qualifying_third_place_groups": qualifying_groups,
        "third_place_routing": third_place_routing,
        "rounds": [
            {
                "round_code": round_code,
                "round_label": ROUND_CODE_LABELS[round_code],
                "matches": round_matches[round_code],
            }
            for round_code in MAIN_BRACKET_ROUND_CODES
        ],
    }


def build_deterministic_bracket_v2(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable v2 knockout bracket from modal group rankings and v2 matchup odds."""
    modal_group_rankings = get_modal_group_rankings(simulation_df)
    average_third_place_stats = get_average_third_place_stats(simulation_df)
    main_bracket_fixtures = extract_main_bracket_fixtures(fixtures_df)
    team_feature_lookup = team_feature_df.set_index("team_id").to_dict("index")

    third_place_rows = []
    for group_code, ranked_team_ids in modal_group_rankings.items():
        third_team_id = ranked_team_ids[2]
        average_stats = average_third_place_stats.get(
            third_team_id,
            {
                "points": 0.0,
                "goal_difference": 0.0,
                "goals_for": 0.0,
                "team_strength": float(team_feature_lookup[third_team_id]["team_strength"]),
            },
        )
        third_place_rows.append(
            {
                "team_id": third_team_id,
                "group_code": group_code,
                "points": average_stats["points"],
                "goal_difference": average_stats["goal_difference"],
                "goals_for": average_stats["goals_for"],
                "team_strength": average_stats["team_strength"],
            }
        )

    ranked_third_place = rank_best_third_place_teams(pd.DataFrame(third_place_rows))
    qualifying_third_place = ranked_third_place[ranked_third_place["qualifies_as_best_third"]].copy()
    qualifying_groups = "".join(sorted(qualifying_third_place["group_code"].astype(str).tolist()))
    if qualifying_groups not in THIRD_PLACE_ROUTING_MAP:
        raise ValueError(f"Missing Round of 32 routing for third-place combination {qualifying_groups}")
    third_place_routing = THIRD_PLACE_ROUTING_MAP[qualifying_groups]

    match_results: dict[int, dict[str, str]] = {}
    round_matches: dict[str, list[dict[str, object]]] = {round_code: [] for round_code in MAIN_BRACKET_ROUND_CODES}
    for match in main_bracket_fixtures.itertuples(index=False):
        match_number = int(match.match_number)
        home_team_id = resolve_knockout_slot(
            match.home_slot_label,
            match_number,
            modal_group_rankings,
            match_results,
            third_place_routing,
        )
        away_team_id = resolve_knockout_slot(
            match.away_slot_label,
            match_number,
            modal_group_rankings,
            match_results,
            third_place_routing,
        )
        prediction = predict_knockout_matchup_v2(
            home_team_id,
            away_team_id,
            team_feature_lookup,
            model_bundle,
            simulations=head_to_head_simulations,
            seed=stable_seed_from_tokens(seed, match_number, home_team_id, away_team_id),
        )
        winner_team_id = str(prediction["winner_team_id"])
        loser_team_id = away_team_id if winner_team_id == home_team_id else home_team_id
        match_results[match_number] = {
            "winner_team_id": winner_team_id,
            "loser_team_id": loser_team_id,
        }
        round_matches[match.round_code].append(
            {
                "match_number": match_number,
                "round_code": match.round_code,
                "round_label": ROUND_CODE_LABELS[match.round_code],
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "winner_team_id": winner_team_id,
                "winner_win_prob": float(prediction["winner_win_prob"]),
                "home_win_prob": float(prediction["home_win_prob"]),
                "away_win_prob": float(prediction["away_win_prob"]),
            }
        )

    return {
        "modal_group_rankings": modal_group_rankings,
        "qualifying_third_place_team_ids": qualifying_third_place["team_id"].astype(str).tolist(),
        "qualifying_third_place_groups": qualifying_groups,
        "third_place_routing": third_place_routing,
        "rounds": [
            {
                "round_code": round_code,
                "round_label": ROUND_CODE_LABELS[round_code],
                "matches": round_matches[round_code],
            }
            for round_code in MAIN_BRACKET_ROUND_CODES
        ],
    }


def simulate_group_probabilities(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = DEFAULT_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
    baseline_rating_weights: tuple[float, float] = BASELINE_RATING_WEIGHTS,
    form_component_weights: tuple[float, float] = FORM_COMPONENT_WEIGHTS,
    strength_blend_weights: tuple[float, float] = STRENGTH_BLEND_WEIGHTS,
) -> pd.DataFrame:
    """Simulate full-tournament advancement probabilities from group and knockout fixtures."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    strengths_df = build_team_strengths(
        base_df,
        lead_in_df,
        match_window=match_window,
        baseline_rating_weights=baseline_rating_weights,
        form_component_weights=form_component_weights,
        strength_blend_weights=strength_blend_weights,
    )
    group_fixtures = extract_group_stage_fixtures(fixtures_df, group_order=group_order)
    knockout_fixtures = extract_knockout_fixtures(fixtures_df)

    team_global_index = {team_id: idx for idx, team_id in enumerate(strengths_df["team_id"])}
    team_strength_lookup = strengths_df.set_index("team_id")["team_strength"].astype(float).to_dict()
    ko_counts = np.zeros(len(strengths_df), dtype=np.int32)
    top8_third_counts = np.zeros(len(strengths_df), dtype=np.int32)
    r16_counts = np.zeros(len(strengths_df), dtype=np.int32)
    qf_counts = np.zeros(len(strengths_df), dtype=np.int32)
    sf_counts = np.zeros(len(strengths_df), dtype=np.int32)
    final_counts = np.zeros(len(strengths_df), dtype=np.int32)
    champion_counts = np.zeros(len(strengths_df), dtype=np.int32)
    third_place_finish_counts = np.zeros(len(strengths_df), dtype=np.int32)
    third_place_points_sum = np.zeros(len(strengths_df), dtype=np.float64)
    third_place_gd_sum = np.zeros(len(strengths_df), dtype=np.float64)
    third_place_gf_sum = np.zeros(len(strengths_df), dtype=np.float64)
    group_simulations: dict[str, dict[str, np.ndarray | list[str]]] = {}
    finish_counts_by_group: dict[str, np.ndarray] = {}
    group_order_counts_by_group: dict[str, Counter[tuple[str, ...]]] = {group_code: Counter() for group_code in group_order}

    for group_code in group_order:
        group_table = strengths_df[strengths_df["group_code"] == group_code].copy().reset_index(drop=True)
        fixtures = group_fixtures[group_fixtures["group_code"] == group_code].copy().reset_index(drop=True)
        if group_table.empty:
            continue
        if len(fixtures) != 6:
            raise ValueError(f"Group {group_code} requires 6 fixtures, found {len(fixtures)}")

        team_ids = group_table["team_id"].to_numpy()
        team_strength = group_table["team_strength"].to_numpy(dtype=float)
        team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}
        fixture_pairs = np.array(
            [(team_index[row.home_team_id], team_index[row.away_team_id]) for row in fixtures.itertuples(index=False)],
            dtype=int,
        )

        deltas = np.array(
            [team_strength[home_idx] - team_strength[away_idx] for home_idx, away_idx in fixture_pairs],
            dtype=float,
        )
        home_xg = np.clip(
            EXPECTED_GOALS_BASE + EXPECTED_GOALS_SCALE * deltas,
            EXPECTED_GOALS_MIN,
            EXPECTED_GOALS_MAX,
        )
        away_xg = np.clip(
            EXPECTED_GOALS_BASE - EXPECTED_GOALS_SCALE * deltas,
            EXPECTED_GOALS_MIN,
            EXPECTED_GOALS_MAX,
        )

        rng = np.random.default_rng(seed + ord(group_code))
        simulated_home_goals = rng.poisson(home_xg, size=(simulations, len(fixtures)))
        simulated_away_goals = rng.poisson(away_xg, size=(simulations, len(fixtures)))

        points = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_for = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_against = np.zeros((simulations, len(team_ids)), dtype=np.int16)

        simulation_indices = np.arange(simulations)
        for match_index, (home_idx, away_idx) in enumerate(fixture_pairs):
            home_scores = simulated_home_goals[:, match_index].astype(np.int16)
            away_scores = simulated_away_goals[:, match_index].astype(np.int16)

            goals_for[simulation_indices, home_idx] += home_scores
            goals_against[simulation_indices, home_idx] += away_scores
            goals_for[simulation_indices, away_idx] += away_scores
            goals_against[simulation_indices, away_idx] += home_scores

            points[simulation_indices, home_idx] += np.where(home_scores > away_scores, 3, np.where(home_scores == away_scores, 1, 0))
            points[simulation_indices, away_idx] += np.where(home_scores < away_scores, 3, np.where(home_scores == away_scores, 1, 0))

        group_simulations[group_code] = {
            "team_ids": list(team_ids),
            "team_global_indices": np.array([team_global_index[team_id] for team_id in team_ids], dtype=int),
            "team_strength": team_strength,
            "fixture_pairs": fixture_pairs,
            "points": points,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "simulated_home_goals": simulated_home_goals,
            "simulated_away_goals": simulated_away_goals,
        }
        finish_counts_by_group[group_code] = np.zeros((len(team_ids), len(team_ids)), dtype=np.int32)

    knockout_rng = np.random.default_rng(seed + 4096)
    for simulation_index in range(simulations):
        third_place_rows: list[dict[str, object]] = []
        group_rankings: dict[str, list[str]] = {}
        for group_code in group_order:
            if group_code not in group_simulations:
                continue

            group_simulation = group_simulations[group_code]
            points = group_simulation["points"][simulation_index]
            goals_for = group_simulation["goals_for"][simulation_index]
            goals_against = group_simulation["goals_against"][simulation_index]
            ranked_indices = _rank_group_indices(
                points=points,
                goals_for=goals_for,
                goals_against=goals_against,
                fixture_pairs=group_simulation["fixture_pairs"],
                home_goals=group_simulation["simulated_home_goals"][simulation_index],
                away_goals=group_simulation["simulated_away_goals"][simulation_index],
                team_strength=group_simulation["team_strength"],
            )

            finish_counts = finish_counts_by_group[group_code]
            team_global_indices = group_simulation["team_global_indices"]
            for place, team_idx in enumerate(ranked_indices):
                finish_counts[team_idx, place] += 1
                if place < 2:
                    ko_counts[team_global_indices[team_idx]] += 1
            group_rankings[group_code] = [group_simulation["team_ids"][team_idx] for team_idx in ranked_indices]
            group_order_counts_by_group[group_code][tuple(group_rankings[group_code])] += 1

            third_idx = ranked_indices[2]
            third_global_index = int(team_global_indices[third_idx])
            third_place_finish_counts[third_global_index] += 1
            third_place_points_sum[third_global_index] += int(points[third_idx])
            third_place_gd_sum[third_global_index] += int(goals_for[third_idx] - goals_against[third_idx])
            third_place_gf_sum[third_global_index] += int(goals_for[third_idx])
            third_place_rows.append(
                {
                    "team_id": group_simulation["team_ids"][third_idx],
                    "team_global_index": third_global_index,
                    "group_code": group_code,
                    "points": int(points[third_idx]),
                    "goal_difference": int(goals_for[third_idx] - goals_against[third_idx]),
                    "goals_for": int(goals_for[third_idx]),
                    "team_strength": float(group_simulation["team_strength"][third_idx]),
                }
            )

        if third_place_rows:
            ranked_third_place = rank_best_third_place_teams(pd.DataFrame(third_place_rows))
            qualifying_third_place = ranked_third_place[ranked_third_place["qualifies_as_best_third"]].copy()
            qualifying_groups = "".join(sorted(qualifying_third_place["group_code"].astype(str).tolist()))
            if qualifying_groups not in THIRD_PLACE_ROUTING_MAP:
                raise ValueError(f"Missing Round of 32 routing for third-place combination {qualifying_groups}")
            third_place_routing = THIRD_PLACE_ROUTING_MAP[qualifying_groups]

            for row in qualifying_third_place.itertuples(index=False):
                ko_counts[int(row.team_global_index)] += 1
                top8_third_counts[int(row.team_global_index)] += 1

            match_results: dict[int, dict[str, str]] = {}
            for match in knockout_fixtures.itertuples(index=False):
                match_number = int(match.match_number)
                home_team_id = resolve_knockout_slot(
                    match.home_slot_label,
                    match_number,
                    group_rankings,
                    match_results,
                    third_place_routing,
                )
                away_team_id = resolve_knockout_slot(
                    match.away_slot_label,
                    match_number,
                    group_rankings,
                    match_results,
                    third_place_routing,
                )
                winner_team_id, loser_team_id = simulate_knockout_match(
                    home_team_id,
                    away_team_id,
                    team_strength_lookup,
                    knockout_rng,
                )
                winner_global_idx = team_global_index[winner_team_id]
                if match.round_code == "R32":
                    r16_counts[winner_global_idx] += 1
                elif match.round_code == "R16":
                    qf_counts[winner_global_idx] += 1
                elif match.round_code == "QF":
                    sf_counts[winner_global_idx] += 1
                elif match.round_code == "SF":
                    final_counts[winner_global_idx] += 1
                elif match.round_code == "F":
                    champion_counts[winner_global_idx] += 1
                match_results[match_number] = {
                    "winner_team_id": winner_team_id,
                    "loser_team_id": loser_team_id,
                }

    results: list[pd.DataFrame] = []
    for group_code in group_order:
        if group_code not in group_simulations:
            continue
        team_ids = np.array(group_simulations[group_code]["team_ids"], dtype=object)
        finish_counts = finish_counts_by_group[group_code]
        probability_frame = pd.DataFrame(
            {f"prob_{place + 1}": finish_counts[:, place] / simulations * 100 for place in range(len(team_ids))}
        )
        probability_frame["team_id"] = team_ids
        probability_frame["group_code"] = group_code
        results.append(probability_frame)

    probabilities_df = pd.concat(results, ignore_index=True)
    team_probability_maps = {
        "top8_third_prob": top8_third_counts,
        "ko_prob": ko_counts,
        "r16_prob": r16_counts,
        "qf_prob": qf_counts,
        "sf_prob": sf_counts,
        "final_prob": final_counts,
        "champion_prob": champion_counts,
    }
    for column_name, counts in team_probability_maps.items():
        probabilities_df[column_name] = probabilities_df["team_id"].map(
            {team_id: counts[team_global_index[team_id]] / simulations * 100 for team_id in strengths_df["team_id"]}
        )
    result_df = strengths_df.merge(probabilities_df, on=["team_id", "group_code"], how="left")
    modal_group_rankings = {}
    for group_code, order_counter in group_order_counts_by_group.items():
        if not order_counter:
            continue
        modal_group_rankings[group_code] = list(
            sorted(order_counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        )
    average_third_place_stats = {}
    for team_id, global_index in team_global_index.items():
        finish_count = int(third_place_finish_counts[global_index])
        if finish_count == 0:
            continue
        average_third_place_stats[team_id] = {
            "points": third_place_points_sum[global_index] / finish_count,
            "goal_difference": third_place_gd_sum[global_index] / finish_count,
            "goals_for": third_place_gf_sum[global_index] / finish_count,
            "team_strength": float(team_strength_lookup[team_id]),
        }
    result_df.attrs["modal_group_rankings"] = modal_group_rankings
    result_df.attrs["average_third_place_stats"] = average_third_place_stats
    return result_df


def simulate_group_probabilities_v2(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = DEFAULT_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
) -> pd.DataFrame:
    """Simulate the 2026 tournament using the v2 multinomial outcome model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    model_bundle = fit_v2_match_multinomial_model(match_window=match_window)
    feature_df = build_v2_match_feature_table(base_df, lead_in_df, match_window=match_window)
    group_fixtures = extract_group_stage_fixtures(fixtures_df, group_order=group_order)
    knockout_fixtures = extract_knockout_fixtures(fixtures_df)

    team_global_index = {team_id: idx for idx, team_id in enumerate(feature_df["team_id"])}
    team_feature_lookup = feature_df.set_index("team_id").to_dict("index")
    team_strength_lookup = feature_df.set_index("team_id")["team_strength"].astype(float).to_dict()
    ko_counts = np.zeros(len(feature_df), dtype=np.int32)
    top8_third_counts = np.zeros(len(feature_df), dtype=np.int32)
    r16_counts = np.zeros(len(feature_df), dtype=np.int32)
    qf_counts = np.zeros(len(feature_df), dtype=np.int32)
    sf_counts = np.zeros(len(feature_df), dtype=np.int32)
    final_counts = np.zeros(len(feature_df), dtype=np.int32)
    champion_counts = np.zeros(len(feature_df), dtype=np.int32)
    third_place_finish_counts = np.zeros(len(feature_df), dtype=np.int32)
    third_place_points_sum = np.zeros(len(feature_df), dtype=np.float64)
    third_place_gd_sum = np.zeros(len(feature_df), dtype=np.float64)
    third_place_gf_sum = np.zeros(len(feature_df), dtype=np.float64)
    group_simulations: dict[str, dict[str, np.ndarray | list[str]]] = {}
    finish_counts_by_group: dict[str, np.ndarray] = {}
    group_order_counts_by_group: dict[str, Counter[tuple[str, ...]]] = {group_code: Counter() for group_code in group_order}

    for group_code in group_order:
        group_table = feature_df[feature_df["group_code"] == group_code].copy().reset_index(drop=True)
        fixtures = group_fixtures[group_fixtures["group_code"] == group_code].copy().reset_index(drop=True)
        if group_table.empty:
            continue
        if len(fixtures) != 6:
            raise ValueError(f"Group {group_code} requires 6 fixtures, found {len(fixtures)}")

        team_ids = group_table["team_id"].astype(str).to_numpy()
        team_strength = group_table["team_strength"].to_numpy(dtype=float)
        team_index = {team_id: idx for idx, team_id in enumerate(team_ids)}
        fixture_pairs = np.array(
            [(team_index[row.home_team_id], team_index[row.away_team_id]) for row in fixtures.itertuples(index=False)],
            dtype=int,
        )

        rng = np.random.default_rng(seed + ord(group_code))
        simulated_home_goals = np.zeros((simulations, len(fixtures)), dtype=np.int16)
        simulated_away_goals = np.zeros((simulations, len(fixtures)), dtype=np.int16)
        points = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_for = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_against = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        simulation_indices = np.arange(simulations)

        for match_index, match in enumerate(fixtures.itertuples(index=False)):
            probability_map = predict_match_probabilities_v2(
                str(match.home_team_id),
                str(match.away_team_id),
                team_feature_lookup,
                model_bundle,
            )
            outcomes = rng.choice(
                len(V2_OUTCOME_LABELS),
                size=simulations,
                p=[
                    float(probability_map["home_win_prob"]),
                    float(probability_map["draw_prob"]),
                    float(probability_map["away_win_prob"]),
                ],
            )
            home_scores = np.zeros(simulations, dtype=np.int16)
            away_scores = np.zeros(simulations, dtype=np.int16)

            for outcome_index, outcome_label in enumerate(V2_OUTCOME_LABELS):
                outcome_mask = outcomes == outcome_index
                sample_count = int(outcome_mask.sum())
                if sample_count == 0:
                    continue
                distribution = model_bundle["scoreline_distributions"][(V2_STAGE_GROUP, outcome_label)]
                scoreline_indices = rng.choice(len(distribution["scorelines"]), size=sample_count, p=distribution["probabilities"])
                sampled_scorelines = np.array(
                    [distribution["scorelines"][int(index)] for index in scoreline_indices],
                    dtype=np.int16,
                )
                home_scores[outcome_mask] = sampled_scorelines[:, 0]
                away_scores[outcome_mask] = sampled_scorelines[:, 1]

            simulated_home_goals[:, match_index] = home_scores
            simulated_away_goals[:, match_index] = away_scores

            home_idx, away_idx = fixture_pairs[match_index]
            goals_for[simulation_indices, home_idx] += home_scores
            goals_against[simulation_indices, home_idx] += away_scores
            goals_for[simulation_indices, away_idx] += away_scores
            goals_against[simulation_indices, away_idx] += home_scores
            points[simulation_indices, home_idx] += np.where(home_scores > away_scores, 3, np.where(home_scores == away_scores, 1, 0))
            points[simulation_indices, away_idx] += np.where(home_scores < away_scores, 3, np.where(home_scores == away_scores, 1, 0))

        group_simulations[group_code] = {
            "team_ids": list(team_ids),
            "team_global_indices": np.array([team_global_index[team_id] for team_id in team_ids], dtype=int),
            "team_strength": team_strength,
            "fixture_pairs": fixture_pairs,
            "points": points,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "simulated_home_goals": simulated_home_goals,
            "simulated_away_goals": simulated_away_goals,
        }
        finish_counts_by_group[group_code] = np.zeros((len(team_ids), len(team_ids)), dtype=np.int32)

    knockout_rng = np.random.default_rng(seed + 4096)
    knockout_probability_cache: dict[tuple[str, str], dict[str, float | str]] = {}
    for simulation_index in range(simulations):
        third_place_rows: list[dict[str, object]] = []
        group_rankings: dict[str, list[str]] = {}
        for group_code in group_order:
            if group_code not in group_simulations:
                continue

            group_simulation = group_simulations[group_code]
            points = group_simulation["points"][simulation_index]
            goals_for = group_simulation["goals_for"][simulation_index]
            goals_against = group_simulation["goals_against"][simulation_index]
            ranked_indices = _rank_group_indices(
                points=points,
                goals_for=goals_for,
                goals_against=goals_against,
                fixture_pairs=group_simulation["fixture_pairs"],
                home_goals=group_simulation["simulated_home_goals"][simulation_index],
                away_goals=group_simulation["simulated_away_goals"][simulation_index],
                team_strength=group_simulation["team_strength"],
            )

            finish_counts = finish_counts_by_group[group_code]
            team_global_indices = group_simulation["team_global_indices"]
            for place, team_idx in enumerate(ranked_indices):
                finish_counts[team_idx, place] += 1
                if place < 2:
                    ko_counts[team_global_indices[team_idx]] += 1
            group_rankings[group_code] = [group_simulation["team_ids"][team_idx] for team_idx in ranked_indices]
            group_order_counts_by_group[group_code][tuple(group_rankings[group_code])] += 1

            third_idx = ranked_indices[2]
            third_global_index = int(team_global_indices[third_idx])
            third_place_finish_counts[third_global_index] += 1
            third_place_points_sum[third_global_index] += int(points[third_idx])
            third_place_gd_sum[third_global_index] += int(goals_for[third_idx] - goals_against[third_idx])
            third_place_gf_sum[third_global_index] += int(goals_for[third_idx])
            third_place_rows.append(
                {
                    "team_id": group_simulation["team_ids"][third_idx],
                    "team_global_index": third_global_index,
                    "group_code": group_code,
                    "points": int(points[third_idx]),
                    "goal_difference": int(goals_for[third_idx] - goals_against[third_idx]),
                    "goals_for": int(goals_for[third_idx]),
                    "team_strength": float(group_simulation["team_strength"][third_idx]),
                }
            )

        if third_place_rows:
            ranked_third_place = rank_best_third_place_teams(pd.DataFrame(third_place_rows))
            qualifying_third_place = ranked_third_place[ranked_third_place["qualifies_as_best_third"]].copy()
            qualifying_groups = "".join(sorted(qualifying_third_place["group_code"].astype(str).tolist()))
            if qualifying_groups not in THIRD_PLACE_ROUTING_MAP:
                raise ValueError(f"Missing Round of 32 routing for third-place combination {qualifying_groups}")
            third_place_routing = THIRD_PLACE_ROUTING_MAP[qualifying_groups]

            for row in qualifying_third_place.itertuples(index=False):
                ko_counts[int(row.team_global_index)] += 1
                top8_third_counts[int(row.team_global_index)] += 1

            match_results: dict[int, dict[str, str]] = {}
            for match in knockout_fixtures.itertuples(index=False):
                match_number = int(match.match_number)
                home_team_id = resolve_knockout_slot(
                    match.home_slot_label,
                    match_number,
                    group_rankings,
                    match_results,
                    third_place_routing,
                )
                away_team_id = resolve_knockout_slot(
                    match.away_slot_label,
                    match_number,
                    group_rankings,
                    match_results,
                    third_place_routing,
                )
                winner_team_id, loser_team_id = simulate_knockout_match_v2(
                    home_team_id,
                    away_team_id,
                    team_feature_lookup,
                    model_bundle,
                    knockout_rng,
                    matchup_probability_cache=knockout_probability_cache,
                )
                winner_global_idx = team_global_index[winner_team_id]
                if match.round_code == "R32":
                    r16_counts[winner_global_idx] += 1
                elif match.round_code == "R16":
                    qf_counts[winner_global_idx] += 1
                elif match.round_code == "QF":
                    sf_counts[winner_global_idx] += 1
                elif match.round_code == "SF":
                    final_counts[winner_global_idx] += 1
                elif match.round_code == "F":
                    champion_counts[winner_global_idx] += 1
                match_results[match_number] = {
                    "winner_team_id": winner_team_id,
                    "loser_team_id": loser_team_id,
                }

    results: list[pd.DataFrame] = []
    for group_code in group_order:
        if group_code not in group_simulations:
            continue
        team_ids = np.array(group_simulations[group_code]["team_ids"], dtype=object)
        finish_counts = finish_counts_by_group[group_code]
        probability_frame = pd.DataFrame(
            {f"prob_{place + 1}": finish_counts[:, place] / simulations * 100.0 for place in range(len(team_ids))}
        )
        probability_frame["team_id"] = team_ids
        probability_frame["group_code"] = group_code
        results.append(probability_frame)

    probabilities_df = pd.concat(results, ignore_index=True)
    team_probability_maps = {
        "top8_third_prob": top8_third_counts,
        "ko_prob": ko_counts,
        "r16_prob": r16_counts,
        "qf_prob": qf_counts,
        "sf_prob": sf_counts,
        "final_prob": final_counts,
        "champion_prob": champion_counts,
    }
    for column_name, counts in team_probability_maps.items():
        probabilities_df[column_name] = probabilities_df["team_id"].map(
            {team_id: counts[team_global_index[team_id]] / simulations * 100.0 for team_id in feature_df["team_id"]}
        )

    result_df = feature_df.merge(probabilities_df, on=["team_id", "group_code"], how="left")
    modal_group_rankings = {}
    for group_code, order_counter in group_order_counts_by_group.items():
        if not order_counter:
            continue
        modal_group_rankings[group_code] = list(
            sorted(order_counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        )

    average_third_place_stats = {}
    for team_id, global_index in team_global_index.items():
        finish_count = int(third_place_finish_counts[global_index])
        if finish_count == 0:
            continue
        average_third_place_stats[team_id] = {
            "points": third_place_points_sum[global_index] / finish_count,
            "goal_difference": third_place_gd_sum[global_index] / finish_count,
            "goals_for": third_place_gf_sum[global_index] / finish_count,
            "team_strength": float(team_strength_lookup[team_id]),
        }

    result_df.attrs["modal_group_rankings"] = modal_group_rankings
    result_df.attrs["average_third_place_stats"] = average_third_place_stats
    return result_df
