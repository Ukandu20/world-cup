from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_GROUP_ORDER = tuple("ABCDEFGHIJKL")
RECENT_MATCH_WINDOW = 10
RESULT_POINTS = {"win": 3, "draw": 1, "loss": 0}
BASELINE_RATING_WEIGHTS = (1.0, 0.0)
FORM_COMPONENT_WEIGHTS = (0.7, 0.3)
STRENGTH_BLEND_WEIGHTS = (0.5, 0.5)
EXPECTED_GOALS_BASE = 1.20
EXPECTED_GOALS_SCALE = 0.40
EXPECTED_GOALS_MIN = 0.20
EXPECTED_GOALS_MAX = 3.00
BEST_THIRD_QUALIFICATION_SLOTS = 8


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
    """Simulate group-stage finishing probabilities from fixtures, ratings, and recent form."""
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

    team_global_index = {team_id: idx for idx, team_id in enumerate(strengths_df["team_id"])}
    ko_counts = np.zeros(len(strengths_df), dtype=np.int32)
    group_simulations: dict[str, dict[str, np.ndarray | list[str]]] = {}
    finish_counts_by_group: dict[str, np.ndarray] = {}

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

    for simulation_index in range(simulations):
        third_place_rows: list[dict[str, object]] = []
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

            third_idx = ranked_indices[2]
            third_place_rows.append(
                {
                    "team_id": group_simulation["team_ids"][third_idx],
                    "team_global_index": int(team_global_indices[third_idx]),
                    "group_code": group_code,
                    "points": int(points[third_idx]),
                    "goal_difference": int(goals_for[third_idx] - goals_against[third_idx]),
                    "goals_for": int(goals_for[third_idx]),
                    "team_strength": float(group_simulation["team_strength"][third_idx]),
                }
            )

        if third_place_rows:
            ranked_third_place = rank_best_third_place_teams(pd.DataFrame(third_place_rows))
            for row in ranked_third_place[ranked_third_place["qualifies_as_best_third"]].itertuples(index=False):
                ko_counts[int(row.team_global_index)] += 1

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
    probabilities_df["ko_prob"] = probabilities_df["team_id"].map(
        {team_id: ko_counts[team_global_index[team_id]] / simulations * 100 for team_id in strengths_df["team_id"]}
    )
    return strengths_df.merge(probabilities_df, on=["team_id", "group_code"], how="left")
