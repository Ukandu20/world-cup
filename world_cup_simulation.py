from __future__ import annotations

import base64
import json
import zlib
from collections import Counter
from collections.abc import Iterable

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
THIRD_PLACE_ROUTING_COMPRESSED = (
    "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"
)


def normalize_weight_pair(primary_weight: float, secondary_weight: float) -> tuple[float, float]:
    """Normalize a two-value weight pair so it sums to 1.0."""
    total = float(primary_weight) + float(secondary_weight)
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return float(primary_weight) / total, float(secondary_weight) / total


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
    ]
    team_metadata = base_df.loc[:, team_metadata_columns].drop_duplicates(subset=["team_id"], keep="first").copy()
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

    form_df["form"] = (
        results_weight * zscore(form_df["results_form"])
        + gd_weight * zscore(form_df["gd_form"])
        + perf_weight * zscore(form_df["perf_vs_exp"])
        + elo_delta_weight * zscore(form_df["elo_delta_form"])
    )

    form_df["schedule_difficulty"] = scale_to_range(form_df["difficulty"]).round(1)
    form_df["avg_opp_elo"] = form_df["avg_opp_elo"].round(1)
    form_df["avg_elo_gap"] = form_df["avg_elo_gap"].round(1)
    form_df["results_form"] = form_df["results_form"].round(3)
    form_df["gd_form"] = form_df["gd_form"].round(3)
    form_df["difficulty"] = form_df["difficulty"].round(3)
    form_df["expected_score"] = form_df["expected_score"].round(3)
    form_df["perf_vs_exp"] = form_df["perf_vs_exp"].round(3)
    form_df["elo_delta_form"] = form_df["elo_delta_form"].round(3)
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
