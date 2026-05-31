from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import *  # noqa: F403


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


def normalize_excluded_editions(exclude_editions: Iterable[int] = ()) -> tuple[int, ...]:
    """Normalize an edition iterable into a stable tuple for filtering and caching."""
    return tuple(sorted({int(edition) for edition in exclude_editions}))


def normalize_training_scope(training_scope: str = DEFAULT_V3_TRAINING_SCOPE) -> str:
    """Normalize and validate a model training-data scope."""
    normalized = normalize_key(training_scope).replace(" ", "_")
    aliases = {
        "world_cup": TRAINING_SCOPE_WORLD_CUP_ONLY,
        "world_cup_only": TRAINING_SCOPE_WORLD_CUP_ONLY,
        "wc_only": TRAINING_SCOPE_WORLD_CUP_ONLY,
        "all_international": TRAINING_SCOPE_ALL_INTERNATIONAL,
        "all_international_since_anchor": TRAINING_SCOPE_ALL_INTERNATIONAL,
        "international": TRAINING_SCOPE_ALL_INTERNATIONAL,
    }
    value = aliases.get(normalized, normalized)
    if value not in TRAINING_SCOPE_OPTIONS:
        raise ValueError(f"Unsupported training_scope '{training_scope}'. Expected one of {TRAINING_SCOPE_OPTIONS}")
    return value


def classify_competition_importance(tournament: str | None) -> float:
    """Map a tournament name onto the shared competition-importance scale."""
    normalized = normalize_key(tournament)
    if not normalized:
        return float(V3_COMPETITION_IMPORTANCE["friendly"])
    if "friendly" in normalized or "exhibition" in normalized:
        return float(V3_COMPETITION_IMPORTANCE["friendly"])
    if any(token in normalized for token in ("qualification", "qualifying", "qualifier", "preliminary competition")):
        return float(V3_COMPETITION_IMPORTANCE["qualifier"])
    if "world cup" in normalized:
        return float(V3_COMPETITION_IMPORTANCE["world_cup_finals"])

    continental_keywords = (
        "african cup of nations",
        "asian cup",
        "copa america",
        "concacaf gold cup",
        "european championship",
        "uefa euro",
        "nations cup",
        "cup of nations",
    )
    if any(token in normalized for token in continental_keywords):
        return float(V3_COMPETITION_IMPORTANCE["continental_finals"])
    return float(V3_COMPETITION_IMPORTANCE["other_competitive"])


def load_world_cup_edition_start_dates() -> dict[int, pd.Timestamp]:
    """Return the first match date for each available World Cup edition."""
    results_path = WORLD_CUP_ROOT / "all_editions" / "results.csv"
    if not results_path.exists():
        raise ValueError(f"World Cup all-editions results file not found: {results_path}")
    results_df = pd.read_csv(results_path)
    return world_cup_edition_start_dates_from_results(results_df)


def world_cup_edition_start_dates_from_results(results_df: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """Return edition start dates from a preloaded all-editions results frame."""
    results_df = results_df.copy()
    results_df["edition"] = pd.to_numeric(results_df["edition"], errors="coerce")
    results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
    starts = (
        results_df.dropna(subset=["edition", "date"])
        .groupby("edition")["date"]
        .min()
        .sort_index()
    )
    return {int(edition): pd.Timestamp(date) for edition, date in starts.items()}


def resolve_training_anchor_year(reference_edition_year: int, lookback_editions: int = 5) -> int:
    """Resolve the oldest edition included by the rolling five-prior-World-Cup policy."""
    starts = load_world_cup_edition_start_dates()
    prior_editions = [edition for edition in sorted(starts) if edition < int(reference_edition_year)]
    required = int(lookback_editions) + 1
    if len(prior_editions) < required:
        raise ValueError(
            f"Need at least {required} prior World Cup editions before {reference_edition_year}; "
            f"found {len(prior_editions)}"
        )
    return int(prior_editions[-required])


def resolve_training_anchor_date(reference_edition_year: int, lookback_editions: int = 5) -> pd.Timestamp:
    """Resolve the start date of the anchor World Cup edition."""
    anchor_year = resolve_training_anchor_year(reference_edition_year, lookback_editions=lookback_editions)
    starts = load_world_cup_edition_start_dates()
    return pd.Timestamp(starts[anchor_year]).normalize()


def resolve_training_anchor_year_from_results(
    reference_edition_year: int,
    results_df: pd.DataFrame,
    lookback_editions: int = 5,
) -> int:
    starts = world_cup_edition_start_dates_from_results(results_df)
    prior_editions = [edition for edition in sorted(starts) if edition < int(reference_edition_year)]
    required = int(lookback_editions) + 1
    if len(prior_editions) < required:
        raise ValueError(
            f"Need at least {required} prior World Cup editions before {reference_edition_year}; "
            f"found {len(prior_editions)}"
        )
    return int(prior_editions[-required])


def resolve_training_anchor_date_from_results(
    reference_edition_year: int,
    results_df: pd.DataFrame,
    lookback_editions: int = 5,
) -> pd.Timestamp:
    anchor_year = resolve_training_anchor_year_from_results(
        reference_edition_year,
        results_df,
        lookback_editions=lookback_editions,
    )
    starts = world_cup_edition_start_dates_from_results(results_df)
    return pd.Timestamp(starts[anchor_year]).normalize()


def training_metadata_from_frame(
    training_df: pd.DataFrame,
    training_scope: str,
    anchor_year: int,
    anchor_date: pd.Timestamp,
) -> dict[str, object]:
    """Build common training-source metadata for validation artifacts and model bundles."""
    date_series = pd.to_datetime(training_df["date"], errors="coerce") if "date" in training_df.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "training_scope": normalize_training_scope(training_scope),
        "anchor_year": int(anchor_year),
        "anchor_date": pd.Timestamp(anchor_date).strftime("%Y-%m-%d"),
        "training_start_date": "" if date_series.dropna().empty else pd.Timestamp(date_series.min()).strftime("%Y-%m-%d"),
        "training_end_date": "" if date_series.dropna().empty else pd.Timestamp(date_series.max()).strftime("%Y-%m-%d"),
        "training_match_count": int(len(training_df)),
        "sample_weight_policy": SAMPLE_WEIGHT_POLICY,
    }


def select_prior_editions(
    edition_year: int,
    edition_weight_map: dict[int, int],
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> list[int]:
    """Return the most recent prior World Cup editions for one reference year."""
    earlier_editions = [year for year in sorted(edition_weight_map) if year < int(edition_year)]
    if edition_lookback > 0:
        return earlier_editions[-int(edition_lookback) :]
    return earlier_editions


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
        appearance_counts = pd.concat([group["home_team_id"], group["away_team_id"]]).value_counts()
        team_count = len(appearance_counts)
        expected_fixtures = team_count * (team_count - 1) // 2
        expected_appearances = max(team_count - 1, 0)
        if team_count < 2 or len(group) != expected_fixtures:
            raise ValueError(
                f"Expected complete round-robin fixtures for Group {group_code}, "
                f"found {len(group)} fixtures across {team_count} teams"
            )
        if not appearance_counts.eq(expected_appearances).all():
            raise ValueError(f"Expected each team in Group {group_code} to play {expected_appearances} fixtures")

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
    normalized = normalize_key(str(stage))
    return V2_STAGE_GROUP if "group" in normalized or normalized == "gs" else V2_STAGE_KNOCKOUT


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


def load_historical_world_cup_results(exclude_editions: Iterable[int] = ()) -> pd.DataFrame:
    """Load the historical match results used to train the v2 multinomial model."""
    excluded = set(normalize_excluded_editions(exclude_editions))
    rows: list[pd.DataFrame] = []
    for year in range(HISTORICAL_RESULTS_START_YEAR, HISTORICAL_RESULTS_END_YEAR + 1):
        if year in excluded:
            continue
        path = WORLD_CUP_ROOT / str(year) / "results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        parsed = normalize_match_level_results(df)
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


def normalize_match_level_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match from either legacy or team-perspective results."""
    if "home_team" in results_df.columns and "away_team" in results_df.columns:
        return results_df.copy()
    required = {"is_home", "team", "opponent", "team_score", "opponent_score"}
    if not required.issubset(results_df.columns):
        return results_df.copy()

    is_home = results_df["is_home"].astype(str).str.upper().eq("TRUE")
    home_rows = results_df.loc[is_home].copy()
    rename_map = {
        "team": "home_team",
        "opponent": "away_team",
        "team_id": "home_team_code",
        "opponent_id": "away_team_code",
        "team_score": "home_score",
        "opponent_score": "away_score",
        "team_elo_start": "home_elo_start",
        "opponent_elo_start": "away_elo_start",
        "team_elo_end": "home_elo_end",
        "opponent_elo_end": "away_elo_end",
        "team_elo_delta": "home_elo_delta",
    }
    parsed = home_rows.rename(columns={old: new for old, new in rename_map.items() if old in home_rows.columns})
    if "away_elo_delta" not in parsed.columns and "home_elo_delta" in parsed.columns:
        parsed["away_elo_delta"] = -pd.to_numeric(parsed["home_elo_delta"], errors="coerce")
    return parsed.reset_index(drop=True)


def lead_in_to_match_level_results(lead_in_df: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical team-perspective lead-in rows into deduplicated match rows."""
    df = lead_in_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column_name in ("team_score", "opponent_score"):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    is_home = df.get("is_home_perspective", False)
    is_home = is_home.astype(str).str.upper().isin({"TRUE", "1", "YES"}) if hasattr(is_home, "astype") else False
    out = pd.DataFrame(
        {
            "date": df["date"],
            "edition": pd.to_datetime(df["date"], errors="coerce").dt.year,
            "home_team_id": np.where(is_home, df["qualified_team_id"], df["opponent_team_id"]),
            "away_team_id": np.where(is_home, df["opponent_team_id"], df["qualified_team_id"]),
            "home_team": df.get("home_team_canonical", df.get("home_team", "")),
            "away_team": df.get("away_team_canonical", df.get("away_team", "")),
            "home_score": np.where(is_home, df["team_score"], df["opponent_score"]),
            "away_score": np.where(is_home, df["opponent_score"], df["team_score"]),
            "tournament": df.get("tournament", ""),
            "stage": df.get("stage", df.get("tournament", "")),
            "country": df.get("country", ""),
            "neutral": df.get("neutral", False),
            "match_key": df.get("match_key", ""),
        }
    )
    dedupe_columns = [column for column in ("match_key", "date", "home_team_id", "away_team_id") if column in out.columns]
    if dedupe_columns:
        out = out.drop_duplicates(dedupe_columns, keep="first")
    return out.dropna(subset=["date", "home_team_id", "away_team_id", "home_score", "away_score"]).reset_index(drop=True)


def world_cup_results_to_match_level_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Convert all-editions World Cup results into one row per match."""
    parsed = normalize_match_level_results(results_df)
    parsed = parsed.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    parsed["edition"] = pd.to_numeric(parsed.get("edition", parsed["date"].dt.year), errors="coerce")
    for column_name in ("home_score", "away_score"):
        parsed[column_name] = pd.to_numeric(parsed[column_name], errors="coerce")
    if "home_team_id" not in parsed.columns:
        parsed["home_team_id"] = parsed.get("home_team_code", parsed.get("team_id", parsed.get("home_team", "")))
    if "away_team_id" not in parsed.columns:
        parsed["away_team_id"] = parsed.get("away_team_code", parsed.get("opponent_id", parsed.get("away_team", "")))
    if "tournament" not in parsed.columns:
        parsed["tournament"] = "FIFA World Cup"
    if "country" not in parsed.columns:
        parsed["country"] = ""
    if "neutral" not in parsed.columns:
        parsed["neutral"] = True
    return parsed.dropna(subset=["date", "home_team_id", "away_team_id", "home_score", "away_score"]).reset_index(drop=True)


def _snapshot_position_score(position: object) -> float:
    value = pd.to_numeric(position, errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return 0.0
    return float(np.clip(1.0 - ((float(value) - 1.0) / 31.0), 0.0, 1.0))


def _snapshot_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return float(default if pd.isna(parsed) else parsed)


def _snapshot_team_values(snapshot_row: pd.Series) -> dict[str, float]:
    return {
        "elo": _snapshot_float(snapshot_row.get("start_elo", 0.0)),
        "results_form": _snapshot_float(snapshot_row.get("results_form", 0.0)),
        "gd_form": _snapshot_float(snapshot_row.get("gd_form", 0.0)),
        "perf_vs_exp": _snapshot_float(snapshot_row.get("perf_vs_exp", 0.0)),
        "goals_for": _snapshot_float(snapshot_row.get("goals_for", 0.0)),
        "goals_against": _snapshot_float(snapshot_row.get("goals_against", 0.0)),
        "placement": _snapshot_position_score(snapshot_row.get("best_position", np.nan)),
        "appearance": _snapshot_float(snapshot_row.get("appearance_count", 0.0)),
        "wc_l5_goal_diff": _snapshot_float(snapshot_row.get("wc_l5_goal_diff", 0.0)),
        "has_wc_l5_history": _snapshot_float(snapshot_row.get("has_wc_l5_history", 0.0)),
    }


def build_snapshot_training_frame(
    match_source: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    reference_edition_year: int,
    match_window: int,
    training_scope: str,
    model_version: str,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
    start_year: int | None = None,
    end_date: str | pd.Timestamp | None = None,
    exclude_tournament: Iterable[str] = (),
    quadratic_weights: bool = False,
) -> pd.DataFrame:
    """Build match-level model features from point-in-time team snapshots."""
    from .feature_snapshots import get_team_features_at_date

    source = match_source.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    for column_name in ("home_score", "away_score"):
        source[column_name] = pd.to_numeric(source[column_name], errors="coerce")
    source = source.dropna(subset=["date", "home_team_id", "away_team_id", "home_score", "away_score"]).copy()
    anchor_date = resolve_training_anchor_date_from_results(
        reference_edition_year,
        data["results"],
        lookback_editions=edition_lookback,
    )
    source = source[source["date"] >= anchor_date].copy()
    if start_year is not None:
        source = source[source["date"].dt.year >= int(start_year)].copy()
    if end_date is not None:
        source = source[source["date"] <= pd.Timestamp(end_date)].copy()
    excluded = {normalize_key(value) for value in exclude_tournament if normalize_key(value)}
    if excluded:
        if "tournament" not in source.columns:
            source["tournament"] = ""
        source["_tournament_key"] = source["tournament"].map(normalize_key)
        source = source[~source["_tournament_key"].isin(excluded)].copy()

    rows_before_snapshot_filter = int(len(source))
    known_team_ids = set(data["lead_in"]["qualified_team_id"].dropna().astype(str))
    known_team_ids.update(data["teams"]["team_id"].dropna().astype(str))
    source = source[
        source["home_team_id"].astype(str).isin(known_team_ids)
        & source["away_team_id"].astype(str).isin(known_team_ids)
    ].copy()

    source["_snapshot_date"] = source["date"].dt.normalize()
    teams_by_snapshot_date = {
        snapshot_date: sorted(
            set(group["home_team_id"].astype(str)).union(set(group["away_team_id"].astype(str)))
        )
        for snapshot_date, group in source.groupby("_snapshot_date", sort=False)
    }
    snapshot_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def snapshot_for(team_id: str, match_date: pd.Timestamp) -> pd.Series:
        cache_key = match_date.normalize()
        if cache_key not in snapshot_cache:
            team_ids_for_date = teams_by_snapshot_date.get(cache_key, [str(team_id)])
            snapshot_cache[cache_key] = get_team_features_at_date(
                team_ids_for_date,
                int(match_date.year),
                match_date.date(),
                data,
                k=match_window,
                quadratic_weights=quadratic_weights,
            )
        return snapshot_cache[cache_key].loc[str(team_id)]

    rows: list[dict[str, object]] = []
    for match in source.sort_values(["date", "home_team_id", "away_team_id"], kind="stable").itertuples(index=False):
        match_date = pd.Timestamp(match.date)
        home_id = str(match.home_team_id)
        away_id = str(match.away_team_id)
        home_values = _snapshot_team_values(snapshot_for(home_id, match_date))
        away_values = _snapshot_team_values(snapshot_for(away_id, match_date))
        neutral_site_flag = 1.0 if str(getattr(match, "neutral", "")).strip().lower() in {"true", "1", "yes", "y"} else 0.0
        country_key = normalize_historical_team_name(getattr(match, "country", ""))
        home_host_flag = 1.0 if not neutral_site_flag and country_key == normalize_historical_team_name(getattr(match, "home_team", "")) else 0.0
        away_host_flag = 1.0 if not neutral_site_flag and country_key == normalize_historical_team_name(getattr(match, "away_team", "")) else 0.0
        stage = str(getattr(match, "stage", ""))
        competition_importance = classify_competition_importance(getattr(match, "tournament", ""))
        row = {
            "date": match_date,
            "edition": int(getattr(match, "edition", match_date.year)) if pd.notna(getattr(match, "edition", match_date.year)) else int(match_date.year),
            "home_team": str(getattr(match, "home_team", home_id)),
            "away_team": str(getattr(match, "away_team", away_id)),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "tournament": str(getattr(match, "tournament", "")),
            "stage": stage,
            "stage_bucket": match_stage_bucket(stage),
            "home_score": int(match.home_score),
            "away_score": int(match.away_score),
            "outcome_label": outcome_label_from_scoreline(int(match.home_score), int(match.away_score)),
            "elo_diff": home_values["elo"] - away_values["elo"],
            "results_form_diff": home_values["results_form"] - away_values["results_form"],
            "goals_for_diff": home_values["goals_for"] - away_values["goals_for"],
            "goals_against_diff": home_values["goals_against"] - away_values["goals_against"],
            "placement_diff": home_values["placement"] - away_values["placement"],
            "appearance_diff": home_values["appearance"] - away_values["appearance"],
            "gd_form_diff": home_values["gd_form"] - away_values["gd_form"],
            "perf_vs_exp_diff": home_values["perf_vs_exp"] - away_values["perf_vs_exp"],
            "competition_importance": competition_importance,
            "neutral_site_flag": neutral_site_flag,
            "net_host_flag": home_host_flag - away_host_flag,
            "sample_weight": competition_importance,
        }
        if model_version == "v4":
            row.update(
                {
                    "home_wc_l5_goal_difference": home_values["wc_l5_goal_diff"],
                    "away_wc_l5_goal_difference": away_values["wc_l5_goal_diff"],
                    "home_has_wc_l5_history": home_values["has_wc_l5_history"],
                    "away_has_wc_l5_history": away_values["has_wc_l5_history"],
                    "is_knockout": 0.0 if match_stage_bucket(stage) == V2_STAGE_GROUP else 1.0,
                    "competition_weight": competition_importance,
                }
            )
        rows.append(row)

    training_df = pd.DataFrame(rows)
    if training_df.empty:
        raise ValueError(f"{model_version.upper()} snapshot training frame is empty")
    training_df["training_scope"] = normalize_training_scope(training_scope)
    training_df["anchor_year"] = int(
        resolve_training_anchor_year_from_results(
            reference_edition_year,
            data["results"],
            lookback_editions=edition_lookback,
        )
    )
    training_df["anchor_date"] = anchor_date.strftime("%Y-%m-%d")
    training_df.attrs["training_rows_before_snapshot_filter"] = rows_before_snapshot_filter
    training_df.attrs["training_rows_after_snapshot_filter"] = int(len(training_df))
    training_df.attrs["training_rows_dropped_snapshot_missing"] = rows_before_snapshot_filter - int(len(training_df))
    return training_df


def load_historical_team_history(data: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Load modeling-safe team history, goal history, Elo starts, and edition metadata."""
    if data is None:
        teams_df = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "teams.csv")
        results_df = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "results.csv")
        history_df = pd.read_csv(WORLD_CUP_ROOT / "fifa_world_cup_history.csv")
        edition_years = sorted(pd.to_numeric(history_df["Year"], errors="coerce").dropna().astype(int).tolist())
        edition_team_counts = {
            int(year): int(teams)
            for year, teams in zip(
                pd.to_numeric(history_df["Year"], errors="coerce"),
                pd.to_numeric(history_df["Teams"], errors="coerce"),
                strict=False,
            )
            if pd.notna(year) and pd.notna(teams)
        }
    else:
        teams_df = data["teams"].copy()
        results_df = data["results"].copy()
        edition_years = sorted(pd.to_numeric(teams_df["year"], errors="coerce").dropna().astype(int).unique().tolist())
        edition_team_counts = (
            teams_df.assign(edition=pd.to_numeric(teams_df["year"], errors="coerce"))
            .dropna(subset=["edition"])
            .groupby("edition")["team_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        edition_team_counts = {int(year): int(count) for year, count in edition_team_counts.items()}

    teams_df["edition"] = pd.to_numeric(teams_df["year"], errors="coerce").astype(int)
    teams_df["position"] = pd.to_numeric(teams_df["position"], errors="coerce")
    teams_df["matches_played"] = pd.to_numeric(teams_df.get("matches_played", 0), errors="coerce").fillna(0.0)
    teams_df["country"] = teams_df["team"]
    teams_df["team_key"] = teams_df["team"].map(normalize_historical_team_name)

    results = results_df.copy()
    results["edition"] = pd.to_numeric(results["edition"], errors="coerce")
    results["team_score"] = pd.to_numeric(results["team_score"], errors="coerce").fillna(0.0)
    results["opponent_score"] = pd.to_numeric(results["opponent_score"], errors="coerce").fillna(0.0)
    results["team_elo_start"] = pd.to_numeric(results["team_elo_start"], errors="coerce")
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["match_number"] = pd.to_numeric(results["match_number"], errors="coerce")
    result_summary = (
        results.dropna(subset=["edition", "team_id"])
        .sort_values(["edition", "team_id", "date", "match_number"], kind="stable")
        .groupby(["edition", "team_id"], as_index=False)
        .agg(
            gs=("team_score", "sum"),
            ga=("opponent_score", "sum"),
            start_elo=("team_elo_start", "first"),
        )
    )
    result_summary["edition"] = result_summary["edition"].astype(int)

    team_history = teams_df.merge(result_summary, on=["edition", "team_id"], how="left")
    for column_name in ("gs", "ga", "start_elo"):
        team_history[column_name] = pd.to_numeric(team_history[column_name], errors="coerce")

    edition_weight_map = {edition: (index + 1) ** 2 for index, edition in enumerate(edition_years)}
    return team_history, edition_team_counts, edition_weight_map


def compute_pre_tournament_history_features(
    team_key: str,
    edition_year: int,
    team_history_df: pd.DataFrame,
    edition_team_counts: dict[int, int],
    edition_weight_map: dict[int, int],
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> dict[str, float]:
    """Compute appearance and weighted placement features using only earlier editions."""
    earlier_editions = select_prior_editions(
        edition_year,
        edition_weight_map,
        edition_lookback=edition_lookback,
    )
    if not earlier_editions:
        return {"placement": 0.0, "appearance": 0.0}

    team_rows = team_history_df[(team_history_df["team_key"] == team_key) & (team_history_df["edition"] < int(edition_year))]
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
    team_history_df: pd.DataFrame,
    edition_team_counts: dict[int, int],
    edition_weight_map: dict[int, int],
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> dict[str, dict[str, float]]:
    """Build raw team features for every participant in one historical edition."""
    edition_year = int(edition_matches["edition"].iloc[0])
    edition_start = pd.to_datetime(edition_matches["date"], errors="coerce").min()
    edition_team_rows = team_history_df[team_history_df["edition"] == edition_year].copy()
    edition_team_rows["start_elo"] = pd.to_numeric(edition_team_rows["start_elo"], errors="coerce")
    team_start_elo_lookup = {
        str(row.country): float(row.start_elo)
        for row in edition_team_rows.dropna(subset=["start_elo"]).itertuples(index=False)
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
            team_history_df,
            edition_team_counts,
            edition_weight_map,
            edition_lookback=edition_lookback,
        )
        pre_tournament_elo = snapshot["pre_tournament_elo"]
        if pre_tournament_elo == 0.0:
            pre_tournament_elo = float(team_start_elo_lookup.get(team_name, 0.0))
        features[team_name] = {
            **snapshot,
            **history_features,
            "pre_tournament_elo": pre_tournament_elo,
        }
    return features


@lru_cache(maxsize=1)
def load_historical_confederation_lookup() -> dict[str, str]:
    """Map normalized historical team names onto confederations from per-country folders."""
    root = WORLD_CUP_ROOT / "by_confederation"
    lookup: dict[str, str] = {}
    if not root.exists():
        return lookup

    for confederation_dir in root.iterdir():
        if not confederation_dir.is_dir():
            continue
        confederation = str(confederation_dir.name).upper()
        for results_path in confederation_dir.glob("*/results.csv"):
            team_key = normalize_historical_team_name(results_path.parent.name.replace("_", " "))
            if team_key:
                lookup.setdefault(team_key, confederation)
            parsed = parse_country_results_file(results_path)
            if not parsed.empty and "team" in parsed.columns:
                team_values = parsed["team"].dropna().astype(str)
                if not team_values.empty:
                    lookup.setdefault(normalize_historical_team_name(team_values.iloc[0]), confederation)
    return lookup


def build_recent_history_feature_table(
    base_df: pd.DataFrame,
    reference_edition_year: int = 2026,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> pd.DataFrame:
    """Build a trailing-window World Cup history table for current teams."""
    team_history_df, edition_team_counts, edition_weight_map = load_historical_team_history()
    prior_editions = select_prior_editions(
        reference_edition_year,
        edition_weight_map,
        edition_lookback=edition_lookback,
    )
    history_total_weight = float(sum(edition_weight_map[edition] for edition in prior_editions))
    rows: list[dict[str, object]] = []

    for row in base_df.itertuples(index=False):
        team_name = ""
        for column_name in ("canonical_name", "display_name", "tournament_name"):
            value = getattr(row, column_name, "")
            if pd.notna(value) and str(value).strip():
                team_name = str(value)
                break

        existing_participations = pd.to_numeric(getattr(row, "world_cup_participations", np.nan), errors="coerce")
        existing_weighted_participations = pd.to_numeric(
            getattr(row, "weighted_world_cup_participations", np.nan),
            errors="coerce",
        )
        existing_weighted_placement = pd.to_numeric(
            getattr(row, "weighted_world_cup_placement_score", np.nan),
            errors="coerce",
        )

        team_key = normalize_historical_team_name(team_name)
        matched_rows = team_history_df[team_history_df["team_key"] == team_key]
        if team_key and not matched_rows.empty:
            history_features = compute_pre_tournament_history_features(
                team_key,
                reference_edition_year,
                team_history_df,
                edition_team_counts,
                edition_weight_map,
                edition_lookback=edition_lookback,
            )
            position_by_edition = {
                int(matched_row.edition): int(matched_row.position)
                for matched_row in matched_rows[matched_rows["edition"].isin(prior_editions)]
                .dropna(subset=["position"])
                .drop_duplicates(subset=["edition"], keep="first")
                .itertuples(index=False)
            }
            weighted_participations = float(sum(edition_weight_map[edition] for edition in position_by_edition))
            world_cup_participations = int(history_features["appearance"]) + 1
            weighted_placement = float(history_features["placement"])
            row_history_total_weight = history_total_weight
        else:
            weighted_participations = float(existing_weighted_participations) if pd.notna(existing_weighted_participations) else 0.0
            world_cup_participations = int(existing_participations) if pd.notna(existing_participations) else 1
            weighted_placement = float(existing_weighted_placement) if pd.notna(existing_weighted_placement) else 0.0
            row_history_total_weight = WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT

        rows.append(
            {
                "team_id": str(row.team_id),
                "world_cup_participations": world_cup_participations,
                "weighted_world_cup_participations": weighted_participations,
                "weighted_world_cup_placement_score": weighted_placement,
                "history_total_weight": float(row_history_total_weight),
            }
        )

    return pd.DataFrame(rows)


def build_2022_group_code_lookup(results_df: pd.DataFrame) -> dict[str, str]:
    """Derive 2022 group codes from the group-stage opponent graph."""
    group_matches = (
        results_df[results_df["stage"] == "Group Stage"]
        .sort_values(["match_number", "date"], kind="stable")
        .reset_index(drop=True)
    )
    if group_matches.empty:
        raise ValueError("2022 results are missing group-stage matches")

    adjacency: dict[str, set[str]] = {}
    first_match_by_team: dict[str, int] = {}
    for row in group_matches.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        adjacency.setdefault(home_team, set()).add(away_team)
        adjacency.setdefault(away_team, set()).add(home_team)
        match_number = int(row.match_number)
        first_match_by_team[home_team] = min(first_match_by_team.get(home_team, match_number), match_number)
        first_match_by_team[away_team] = min(first_match_by_team.get(away_team, match_number), match_number)

    components: list[list[str]] = []
    remaining = set(adjacency)
    while remaining:
        start_team = sorted(remaining, key=lambda team_name: (first_match_by_team.get(team_name, 999), team_name))[0]
        stack = [start_team]
        component: list[str] = []
        while stack:
            team_name = stack.pop()
            if team_name not in remaining:
                continue
            remaining.remove(team_name)
            component.append(team_name)
            for opponent in sorted(adjacency.get(team_name, set()), reverse=True):
                if opponent in remaining:
                    stack.append(opponent)
        components.append(sorted(component))

    components = sorted(
        components,
        key=lambda teams: min(first_match_by_team.get(team_name, 999) for team_name in teams),
    )
    if len(components) != len(BACKTEST_2022_GROUP_ORDER):
        raise ValueError(f"Expected 8 connected 2022 groups, found {len(components)}")
    if any(len(component) != 4 for component in components):
        raise ValueError("Expected each 2022 group component to contain exactly 4 teams")

    lookup: dict[str, str] = {}
    for group_code, teams in zip(BACKTEST_2022_GROUP_ORDER, components, strict=False):
        for team_name in teams:
            lookup[str(team_name)] = group_code
    return lookup


def build_historical_group_code_lookup(results_df: pd.DataFrame, group_order: Iterable[str] = BACKTEST_2022_GROUP_ORDER) -> dict[str, str]:
    """Derive group codes from the group-stage opponent graph for a 32-team World Cup."""
    group_matches = (
        results_df[results_df["stage"].map(match_stage_bucket).eq(V2_STAGE_GROUP)]
        .sort_values(["match_number", "date"], kind="stable")
        .reset_index(drop=True)
    )
    if group_matches.empty:
        raise ValueError("Expected group-stage results to derive group codes")

    adjacency: dict[str, set[str]] = {}
    first_match_by_team: dict[str, int] = {}
    for row in group_matches.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        match_number = int(row.match_number)
        adjacency.setdefault(home_team, set()).add(away_team)
        adjacency.setdefault(away_team, set()).add(home_team)
        first_match_by_team.setdefault(home_team, match_number)
        first_match_by_team.setdefault(away_team, match_number)

    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining, key=lambda team_name: first_match_by_team.get(team_name, 999))
        stack = [start]
        component: list[str] = []
        while stack:
            team_name = stack.pop()
            if team_name not in remaining:
                continue
            remaining.remove(team_name)
            component.append(team_name)
            for opponent in sorted(adjacency.get(team_name, set()), reverse=True):
                if opponent in remaining:
                    stack.append(opponent)
        components.append(sorted(component))

    group_order = list(group_order)
    components = sorted(
        components,
        key=lambda teams: min(first_match_by_team.get(team_name, 999) for team_name in teams),
    )
    if len(components) != len(group_order):
        raise ValueError(f"Expected {len(group_order)} connected groups, found {len(components)}")

    lookup: dict[str, str] = {}
    for group_code, teams in zip(group_order, components, strict=False):
        for team_name in teams:
            lookup[str(team_name)] = group_code
    return lookup


ROUND_CODE_BY_STAGE = {
    "group stage": "GS",
    "round of 32": "R32",
    "round of 16": "R16",
    "quarter final": "QF",
    "quarter finals": "QF",
    "quarter-final": "QF",
    "quarter-finals": "QF",
    "semi final": "SF",
    "semi finals": "SF",
    "semi-final": "SF",
    "semi-finals": "SF",
    "third place": "3P",
    "third-place": "3P",
    "final": "F",
}


BACKTEST_2022_KNOCKOUT_SLOT_LOOKUP = {
    49: ("1A", "2B"),
    50: ("1C", "2D"),
    51: ("1D", "2C"),
    52: ("1B", "2A"),
    53: ("1E", "2F"),
    54: ("1G", "2H"),
    55: ("1F", "2E"),
    56: ("1H", "2G"),
    57: ("W53", "W54"),
    58: ("W49", "W50"),
    59: ("W55", "W56"),
    60: ("W51", "W52"),
    61: ("W57", "W58"),
    62: ("W59", "W60"),
    63: ("L61", "L62"),
    64: ("W61", "W62"),
}


def round_code_from_stage(stage: object) -> str:
    """Normalize a World Cup stage label into the simulator's round code."""
    normalized = normalize_key(str(stage))
    if normalized in ROUND_CODE_BY_STAGE:
        return ROUND_CODE_BY_STAGE[normalized]
    if "group" in normalized:
        return "GS"
    if "32" in normalized:
        return "R32"
    if "16" in normalized:
        return "R16"
    if "quarter" in normalized:
        return "QF"
    if "semi" in normalized:
        return "SF"
    if "third" in normalized:
        return "3P"
    if "final" in normalized:
        return "F"
    raise ValueError(f"Unsupported World Cup stage label: {stage}")


def tournament_holdout_label(edition_year: int) -> str:
    """Return the public validation holdout label for one World Cup edition."""
    return f"{int(edition_year)} FIFA World Cup"


def validation_artifact_filenames(edition_year: int) -> dict[str, str]:
    """Return fold-aware validation artifact filenames for one holdout edition."""
    year = int(edition_year)
    return {
        "json": f"model_validation_{year}.json",
        "match_predictions": f"match_predictions_{year}.csv",
        "team_backtest": f"team_backtest_{year}.csv",
    }


def _actual_group_rank_lookup(
    results_df: pd.DataFrame,
    group_code_lookup: dict[str, str],
    final_position_lookup: dict[str, int],
) -> dict[str, tuple[int, str]]:
    """Rank actual group-stage tables and return team -> (rank, group_code)."""
    group_matches = results_df[results_df["stage"].map(match_stage_bucket).eq(V2_STAGE_GROUP)].copy()
    rank_lookup: dict[str, tuple[int, str]] = {}
    for group_code in sorted(set(group_code_lookup.values())):
        group_teams = sorted([team_name for team_name, code in group_code_lookup.items() if code == group_code])
        table = {
            team_name: {"points": 0, "goals_for": 0, "goals_against": 0}
            for team_name in group_teams
        }
        group_fixture_rows = group_matches[
            group_matches["home_team"].isin(group_teams) & group_matches["away_team"].isin(group_teams)
        ].sort_values(["match_number"], kind="stable")
        for match in group_fixture_rows.itertuples(index=False):
            home_team = str(match.home_team)
            away_team = str(match.away_team)
            home_score = int(match.home_score)
            away_score = int(match.away_score)
            table[home_team]["goals_for"] += home_score
            table[home_team]["goals_against"] += away_score
            table[away_team]["goals_for"] += away_score
            table[away_team]["goals_against"] += home_score
            if home_score > away_score:
                table[home_team]["points"] += 3
            elif away_score > home_score:
                table[away_team]["points"] += 3
            else:
                table[home_team]["points"] += 1
                table[away_team]["points"] += 1

        ranked = sorted(
            table,
            key=lambda team_name: (
                -table[team_name]["points"],
                -(table[team_name]["goals_for"] - table[team_name]["goals_against"]),
                -table[team_name]["goals_for"],
                final_position_lookup.get(team_name, 999),
                team_name,
            ),
        )
        for index, team_name in enumerate(ranked, start=1):
            rank_lookup[team_name] = (index, group_code)
    return rank_lookup


def _match_winner_loser(row: object) -> tuple[str, str]:
    """Return winner and loser team names for a played knockout match row."""
    home_team = str(getattr(row, "home_team"))
    away_team = str(getattr(row, "away_team"))
    home_score = int(getattr(row, "home_score"))
    away_score = int(getattr(row, "away_score"))
    if home_score > away_score:
        return home_team, away_team
    if away_score > home_score:
        return away_team, home_team
    shootout_winner = str(getattr(row, "shootout_winner", "") or "")
    if shootout_winner:
        if shootout_winner in {home_team, str(getattr(row, "home_team_code", ""))}:
            return home_team, away_team
        if shootout_winner in {away_team, str(getattr(row, "away_team_code", ""))}:
            return away_team, home_team
    return home_team, away_team


def derive_knockout_slot_lookup(
    results_df: pd.DataFrame,
    group_code_lookup: dict[str, str],
    final_position_lookup: dict[str, int],
) -> dict[int, tuple[str, str]]:
    """Derive bracket slot labels from actual group ranks and prior knockout results."""
    group_rank_lookup = _actual_group_rank_lookup(results_df, group_code_lookup, final_position_lookup)
    knockout_rows = (
        results_df[~results_df["stage"].map(match_stage_bucket).eq(V2_STAGE_GROUP)]
        .sort_values(["match_number"], kind="stable")
        .reset_index(drop=True)
    )
    if knockout_rows.empty:
        return {}

    participant_sources: dict[str, list[tuple[int, str]]] = {}
    slot_lookup: dict[int, tuple[str, str]] = {}
    for row in knockout_rows.itertuples(index=False):
        match_number = int(row.match_number)
        labels: list[str] = []
        for team_name in [str(row.home_team), str(row.away_team)]:
            prior_sources = participant_sources.get(team_name, [])
            if prior_sources:
                labels.append(prior_sources[-1][1])
            else:
                rank, group_code = group_rank_lookup[team_name]
                labels.append(f"{rank}{group_code}")
        slot_lookup[match_number] = (labels[0], labels[1])

        winner, loser = _match_winner_loser(row)
        participant_sources.setdefault(winner, []).append((match_number, f"W{match_number}"))
        participant_sources.setdefault(loser, []).append((match_number, f"L{match_number}"))
    return slot_lookup


def tournament_knockout_slot_lookup(
    edition_year: int,
    results_df: pd.DataFrame,
    group_code_lookup: dict[str, str],
    final_position_lookup: dict[str, int],
) -> dict[int, tuple[str, str]]:
    """Return simulator bracket slot labels, preserving legacy 2022 orientation."""
    derived_lookup = derive_knockout_slot_lookup(results_df, group_code_lookup, final_position_lookup)
    if int(edition_year) == 2022:
        return dict(BACKTEST_2022_KNOCKOUT_SLOT_LOOKUP)
    return derived_lookup


@lru_cache(maxsize=8)
def _build_historical_world_cup_backtest_data_cached(edition_year: int = 2022) -> dict[str, object]:
    return _build_historical_world_cup_backtest_data_uncached(edition_year, data=None)


def build_historical_world_cup_backtest_data(
    edition_year: int = 2022,
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Build one historical World Cup's leakage-free backtest inputs."""
    if data is None:
        return _build_historical_world_cup_backtest_data_cached(int(edition_year))
    return _build_historical_world_cup_backtest_data_uncached(int(edition_year), data=data)


def _build_historical_world_cup_backtest_data_uncached(
    edition_year: int = 2022,
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Build one historical World Cup's leakage-free backtest inputs."""
    edition_year = int(edition_year)
    if data is None:
        results_path = WORLD_CUP_ROOT / str(edition_year) / "results.csv"
        if not results_path.exists():
            raise ValueError(f"{edition_year} World Cup files are unavailable for backtesting")
        results_source = pd.read_csv(results_path)
    else:
        results_source = data["results"].copy()
        results_source["edition"] = pd.to_numeric(results_source["edition"], errors="coerce")
        results_source = results_source[results_source["edition"].eq(edition_year)].copy()
        if results_source.empty:
            raise ValueError(f"{edition_year} World Cup rows are unavailable in preloaded validation data")

    results_df = normalize_match_level_results(results_source)
    results_df["match_number"] = pd.to_numeric(results_df["match_number"], errors="coerce").astype(int)
    results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")

    group_code_lookup = build_historical_group_code_lookup(results_df)
    team_history_df, edition_team_counts, edition_weight_map = load_historical_team_history(data)
    teams_df = team_history_df[team_history_df["edition"].eq(edition_year)].copy()
    if teams_df.empty:
        raise ValueError(f"{edition_year} team history is unavailable for backtesting")
    teams_df["position"] = pd.to_numeric(teams_df["position"], errors="coerce").astype(int)
    teams_df["start_elo"] = pd.to_numeric(teams_df["start_elo"], errors="coerce")
    if data is None:
        confederation_lookup = load_historical_confederation_lookup()
    else:
        confederation_lookup = teams_df.set_index("team_key")["confederation"].fillna("").astype(str).to_dict()

    code_lookup: dict[str, str] = {}
    for row in results_df.itertuples(index=False):
        code_lookup[str(row.home_team)] = str(row.home_team_code)
        code_lookup[str(row.away_team)] = str(row.away_team_code)
    if "team_code" in teams_df.columns:
        for row in teams_df.dropna(subset=["team_code"]).itertuples(index=False):
            code_lookup[str(row.country)] = str(row.team_code)

    edition_start = pd.to_datetime(results_df["date"], errors="coerce").min()
    if pd.isna(edition_start):
        raise ValueError(f"{edition_year} results are missing valid match dates")

    base_rows: list[dict[str, object]] = []
    elo_ranks = (
        teams_df.loc[:, ["country", "start_elo"]]
        .sort_values(["start_elo", "country"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    elo_rank_lookup = {str(row.country): index + 1 for index, row in elo_ranks.iterrows()}

    for row in teams_df.sort_values(["position", "country"], kind="stable").itertuples(index=False):
        team_name = str(row.country)
        team_key = normalize_historical_team_name(team_name)
        prior_editions = select_prior_editions(
            edition_year,
            edition_weight_map,
            edition_lookback=V2_PREVIOUS_EDITION_LOOKBACK,
        )
        prior_rows = team_history_df[
            team_history_df["team_key"].eq(team_key) & team_history_df["edition"].isin(prior_editions)
        ]
        team_prior_editions = sorted({int(edition) for edition in prior_rows["edition"].dropna().astype(int).tolist()})
        weighted_participations = float(sum(edition_weight_map[int(edition)] for edition in prior_editions))
        history_features = compute_pre_tournament_history_features(
            team_key,
            edition_year,
            team_history_df,
            edition_team_counts,
            edition_weight_map,
            edition_lookback=V2_PREVIOUS_EDITION_LOOKBACK,
        )
        base_rows.append(
            {
                "team_id": code_lookup.get(team_name, getattr(row, "team_code", team_name)),
                "canonical_name": team_name,
                "display_name": team_name,
                "tournament_name": team_name,
                "flag_icon_code": "",
                "group_code": group_code_lookup[team_name],
                "confederation": confederation_lookup.get(team_key, ""),
                "world_rank": int(elo_rank_lookup.get(team_name, len(elo_rank_lookup) + 1)),
                "fifa_points": 0.0,
                "elo_rating": float(row.start_elo) if pd.notna(row.start_elo) else 0.0,
                "world_cup_participations": int(history_features["appearance"]) + 1,
                "weighted_world_cup_participations": float(sum(edition_weight_map[int(edition)] for edition in team_prior_editions)),
                "weighted_world_cup_placement_score": float(history_features["placement"]),
            }
        )

    base_df = pd.DataFrame(base_rows).sort_values(
        ["group_code", "display_name"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    if data is not None and "lead_in" in data:
        lead_in_df = data["lead_in"].copy()
        lead_in_df["date"] = pd.to_datetime(lead_in_df["date"], errors="coerce")
        lead_in_df = lead_in_df[
            lead_in_df["qualified_team_id"].astype(str).isin(set(base_df["team_id"].astype(str)))
            & (lead_in_df["date"] < edition_start)
        ].copy()
    else:
        lead_in_rows: list[dict[str, object]] = []
        country_results_lookup = load_historical_country_results_lookup()
        lead_in_counter = 1
        for team_row in base_df.itertuples(index=False):
            team_key = normalize_historical_team_name(str(team_row.canonical_name))
            team_results = country_results_lookup.get(team_key, pd.DataFrame()).copy()
            if team_results.empty:
                continue
            team_results["date"] = pd.to_datetime(team_results["date"], errors="coerce")
            team_results = team_results[team_results["date"] < edition_start].sort_values(["date"], kind="stable")
            for result_row in team_results.itertuples(index=False):
                lead_in_rows.append(
                    {
                        "lead_in_id": f"{edition_year}_lead_in_{lead_in_counter:06d}",
                        "date": pd.Timestamp(result_row.date).strftime("%Y-%m-%d"),
                        "qualified_team_id": str(team_row.team_id),
                        "qualified_team_name": str(team_row.canonical_name),
                        "opponent_name": str(getattr(result_row, "opponent", "")),
                        "team_score": getattr(result_row, "team_score", np.nan),
                        "opponent_score": getattr(result_row, "opponent_score", np.nan),
                        "goal_difference": (
                            pd.to_numeric(getattr(result_row, "team_score", np.nan), errors="coerce")
                            - pd.to_numeric(getattr(result_row, "opponent_score", np.nan), errors="coerce")
                        ),
                        "result": str(getattr(result_row, "result", "")),
                        "team_elo_start": getattr(result_row, "team_elo_start", np.nan),
                        "opponent_elo_start": getattr(result_row, "opponent_elo_start", np.nan),
                        "team_elo_delta": getattr(result_row, "team_elo_delta", np.nan),
                    }
                )
                lead_in_counter += 1
        lead_in_df = pd.DataFrame(lead_in_rows)

    snapshot_features_df = pd.DataFrame()
    if data is not None:
        from .feature_snapshots import get_team_features_at_date

        snapshot_features_df = get_team_features_at_date(
            base_df["team_id"].astype(str).tolist(),
            edition_year,
            pd.Timestamp(edition_start).date(),
            data,
        )
        base_df = base_df.copy()
        base_df["elo_rating"] = base_df["team_id"].astype(str).map(snapshot_features_df["start_elo"]).fillna(base_df["elo_rating"])

    final_position_lookup = {str(row.country): int(row.position) for row in teams_df.itertuples(index=False)}
    knockout_slot_lookup = tournament_knockout_slot_lookup(
        edition_year,
        results_df,
        group_code_lookup,
        final_position_lookup,
    )
    fixtures_rows: list[dict[str, object]] = []
    for row in results_df.sort_values(["match_number"], kind="stable").itertuples(index=False):
        match_number = int(row.match_number)
        home_slot_label, away_slot_label = knockout_slot_lookup.get(match_number, ("", ""))
        is_group_stage = match_stage_bucket(str(row.stage)) == V2_STAGE_GROUP
        fixtures_rows.append(
            {
                "match_id": f"{edition_year}_{match_number}",
                "match_number": match_number,
                "edition_year": edition_year,
                "round_code": round_code_from_stage(row.stage),
                "round_name": str(row.stage),
                "group_code": group_code_lookup.get(str(row.home_team), "") if is_group_stage else "",
                "kickoff_datetime_utc": f"{pd.Timestamp(row.date).strftime('%Y-%m-%d')}T00:00:00Z",
                "home_team_id": str(code_lookup[str(row.home_team)]),
                "away_team_id": str(code_lookup[str(row.away_team)]),
                "home_slot_label": home_slot_label,
                "away_slot_label": away_slot_label,
                "home_tournament_name": str(row.home_team),
                "away_tournament_name": str(row.away_team),
                "home_canonical_name": str(row.home_team),
                "away_canonical_name": str(row.away_team),
                "status": str(row.status),
                "home_score": int(row.home_score),
                "away_score": int(row.away_score),
            }
        )
    fixtures_df = pd.DataFrame(fixtures_rows)

    return {
        "base_df": base_df,
        "lead_in_df": lead_in_df,
        "fixtures_df": fixtures_df,
        "results_df": results_df,
        "teams_df": teams_df,
        "group_code_lookup": group_code_lookup,
        "snapshot_features_df": snapshot_features_df,
    }


@lru_cache(maxsize=1)
def build_2022_backtest_data() -> dict[str, object]:
    """Build the 2022 World Cup inputs needed for leakage-free backtests."""
    return build_historical_world_cup_backtest_data(2022)


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


def build_actual_group_standings(
    results_df: pd.DataFrame,
    group_code_lookup: dict[str, str],
    team_feature_df: pd.DataFrame,
    group_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Rank real group-stage tables using the simulator's tie-break logic."""
    actual_rows: list[pd.DataFrame] = []
    strength_lookup = team_feature_df.set_index("team_id")["team_strength"].astype(float).to_dict()
    name_to_team_id = team_feature_df.set_index("display_name")["team_id"].astype(str).to_dict()
    group_matches = results_df[results_df["stage"].map(match_stage_bucket).eq(V2_STAGE_GROUP)].copy()
    resolved_group_order = list(group_order) if group_order is not None else sorted(set(group_code_lookup.values()))

    for group_code in resolved_group_order:
        group_teams = sorted([team_name for team_name, code in group_code_lookup.items() if code == group_code])
        if not group_teams:
            continue
        table_rows = []
        fixture_rows = []
        for team_name in group_teams:
            team_id = name_to_team_id[team_name]
            table_rows.append(
                {
                    "team_id": team_id,
                    "display_name": team_name,
                    "group_code": group_code,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "team_strength": float(strength_lookup.get(team_id, 0.0)),
                }
            )

        table_index = {row["team_id"]: idx for idx, row in enumerate(table_rows)}
        group_fixture_rows = group_matches[
            group_matches["home_team"].isin(group_teams) & group_matches["away_team"].isin(group_teams)
        ].sort_values(["match_number"], kind="stable")
        for match in group_fixture_rows.itertuples(index=False):
            home_team_id = name_to_team_id[str(match.home_team)]
            away_team_id = name_to_team_id[str(match.away_team)]
            home_idx = table_index[home_team_id]
            away_idx = table_index[away_team_id]
            home_score = int(match.home_score)
            away_score = int(match.away_score)

            table_rows[home_idx]["goals_for"] += home_score
            table_rows[home_idx]["goals_against"] += away_score
            table_rows[away_idx]["goals_for"] += away_score
            table_rows[away_idx]["goals_against"] += home_score
            if home_score > away_score:
                table_rows[home_idx]["points"] += 3
            elif away_score > home_score:
                table_rows[away_idx]["points"] += 3
            else:
                table_rows[home_idx]["points"] += 1
                table_rows[away_idx]["points"] += 1

            fixture_rows.append(
                {
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_goals": home_score,
                    "away_goals": away_score,
                }
            )

        ranked_group = rank_group_standings(pd.DataFrame(table_rows), pd.DataFrame(fixture_rows))
        ranked_group["actual_group_rank"] = np.arange(1, len(ranked_group) + 1)
        actual_rows.append(ranked_group)

    return pd.concat(actual_rows, ignore_index=True)


def build_2022_actual_group_standings(
    results_df: pd.DataFrame,
    group_code_lookup: dict[str, str],
    team_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility wrapper for callers/tests that still use the 2022 name."""
    return build_actual_group_standings(
        results_df,
        group_code_lookup,
        team_feature_df,
        group_order=BACKTEST_2022_GROUP_ORDER,
    )


def stage_label_from_position(position: int) -> str:
    """Convert a placement position into the farthest stage reached label."""
    position = int(position)
    if position <= 1:
        return "Champion"
    if position <= 2:
        return "Final"
    if position <= 4:
        return "Semi-final"
    if position <= 8:
        return "Quarter-final"
    if position <= 16:
        return "Round of 16"
    return "Group Stage"


__all__ = [
    'normalize_weight_pair',
    'normalize_key',
    'zscore',
    'scale_to_range',
    'clip_scale',
    'compute_elo_expected_score',
    'normalize_weighted_form_result',
    'normalize_excluded_editions',
    'normalize_training_scope',
    'classify_competition_importance',
    'load_world_cup_edition_start_dates',
    'world_cup_edition_start_dates_from_results',
    'resolve_training_anchor_year',
    'resolve_training_anchor_date',
    'resolve_training_anchor_year_from_results',
    'resolve_training_anchor_date_from_results',
    'training_metadata_from_frame',
    'select_prior_editions',
    'extract_group_stage_fixtures',
    'extract_knockout_fixtures',
    'extract_main_bracket_fixtures',
    'build_historical_world_cup_backtest_data',
    'derive_knockout_slot_lookup',
    'tournament_knockout_slot_lookup',
    'round_code_from_stage',
    'tournament_holdout_label',
    'validation_artifact_filenames',
    'build_recent_form_metrics',
    'build_weighted_form_table',
    'build_team_strengths',
    'normalize_historical_team_name',
    'outcome_label_from_scoreline',
    'match_stage_bucket',
    'compute_history_placement_score',
    'parse_country_results_file',
    'load_historical_country_results_lookup',
    'latest_pre_tournament_elo',
    'compute_weighted_form_snapshot',
    'build_weighted_form_feature_lookup',
    'lead_in_to_match_level_results',
    'world_cup_results_to_match_level_results',
    'build_snapshot_training_frame',
    'load_historical_world_cup_results',
    'load_historical_team_history',
    'compute_pre_tournament_history_features',
    'build_pre_tournament_team_features_by_edition',
    'load_historical_confederation_lookup',
    'build_recent_history_feature_table',
    'build_2022_group_code_lookup',
    'build_2022_backtest_data',
    '_head_to_head_stats',
    '_rank_group_indices',
    'rank_best_third_place_teams',
    'rank_group_standings',
    'expected_goals_from_strengths',
    'resolve_knockout_slot',
    'stable_seed_from_tokens',
    'get_modal_group_rankings',
    'get_average_third_place_stats',
    'build_actual_group_standings',
    'build_2022_actual_group_standings',
    'stage_label_from_position'
]
