from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import *  # noqa: F403
from .shared import *  # noqa: F403


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


__all__ = [
    'build_recent_form_metrics',
    'build_weighted_form_table',
    'build_team_strengths',
    'simulate_knockout_match',
    'predict_knockout_matchup',
    'build_deterministic_bracket',
    'simulate_group_probabilities',
]
