from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_GROUP_ORDER = tuple("ABCDEFGHIJKL")
RECENT_MATCH_WINDOW = 8
RESULT_POINTS = {"win": 3, "draw": 1, "loss": 0}
BASELINE_RATING_WEIGHTS = (0.65, 0.35)
FORM_COMPONENT_WEIGHTS = (0.7, 0.3)
STRENGTH_BLEND_WEIGHTS = (0.75, 0.25)
EXPECTED_GOALS_BASE = 1.20
EXPECTED_GOALS_SCALE = 0.40
EXPECTED_GOALS_MIN = 0.20
EXPECTED_GOALS_MAX = 3.00


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a numeric series while handling empty or constant inputs safely."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - series.mean()) / std


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


def build_recent_form_metrics(lead_in_df: pd.DataFrame, match_window: int = RECENT_MATCH_WINDOW) -> pd.DataFrame:
    """Build recent-form metrics from each team's most recent lead-in matches."""
    if match_window <= 0:
        raise ValueError("match_window must be positive")

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
    form["form_score"] = (
        FORM_COMPONENT_WEIGHTS[0] * form["points_form_z"] + FORM_COMPONENT_WEIGHTS[1] * form["goal_diff_form_z"]
    )
    return form


def build_team_strengths(base_df: pd.DataFrame, lead_in_df: pd.DataFrame, match_window: int = RECENT_MATCH_WINDOW) -> pd.DataFrame:
    """Blend ratings and recent form into one pre-tournament strength score."""
    df = base_df.copy()
    df["elo_rating"] = pd.to_numeric(df["elo_rating"], errors="coerce")
    df["fifa_points"] = pd.to_numeric(df["fifa_points"], errors="coerce")
    df["rating_score"] = (
        BASELINE_RATING_WEIGHTS[0] * zscore(df["elo_rating"]) + BASELINE_RATING_WEIGHTS[1] * zscore(df["fifa_points"])
    )

    form = build_recent_form_metrics(lead_in_df, match_window=match_window)
    df = df.merge(form, left_on="team_id", right_on="qualified_team_id", how="left")
    df["recent_matches"] = df["recent_matches"].fillna(0).astype(int)
    df["points_per_match"] = df["points_per_match"].fillna(0.0)
    df["goal_diff_per_match"] = df["goal_diff_per_match"].fillna(0.0)
    df["points_form_z"] = df["points_form_z"].fillna(0.0)
    df["goal_diff_form_z"] = df["goal_diff_form_z"].fillna(0.0)
    df["form_score"] = df["form_score"].fillna(0.0)
    df["team_strength"] = (
        STRENGTH_BLEND_WEIGHTS[0] * df["rating_score"] + STRENGTH_BLEND_WEIGHTS[1] * df["form_score"]
    )
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


def simulate_group_probabilities(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = DEFAULT_GROUP_ORDER,
) -> pd.DataFrame:
    """Simulate group-stage finishing probabilities from fixtures, ratings, and recent form."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    strengths_df = build_team_strengths(base_df, lead_in_df, match_window=RECENT_MATCH_WINDOW)
    group_fixtures = extract_group_stage_fixtures(fixtures_df, group_order=group_order)

    results: list[pd.DataFrame] = []
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

        finish_counts = np.zeros((len(team_ids), len(team_ids)), dtype=np.int32)
        for simulation_index in range(simulations):
            ranked_indices = _rank_group_indices(
                points=points[simulation_index],
                goals_for=goals_for[simulation_index],
                goals_against=goals_against[simulation_index],
                fixture_pairs=fixture_pairs,
                home_goals=simulated_home_goals[simulation_index],
                away_goals=simulated_away_goals[simulation_index],
                team_strength=team_strength,
            )
            for place, team_idx in enumerate(ranked_indices):
                finish_counts[team_idx, place] += 1

        probability_frame = pd.DataFrame(
            {
                f"prob_{place + 1}": finish_counts[:, place] / simulations * 100
                for place in range(len(team_ids))
            }
        )
        probability_frame["team_id"] = team_ids
        probability_frame["group_code"] = group_code
        results.append(probability_frame)

    probabilities_df = pd.concat(results, ignore_index=True)
    return strengths_df.merge(probabilities_df, on=["team_id", "group_code"], how="left")
