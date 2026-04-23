from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import *  # noqa: F403
from .shared import *  # noqa: F403


def build_v2_team_strengths(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    baseline_rating_weights: tuple[float, float] = BASELINE_RATING_WEIGHTS,
    form_composite_weights: tuple[float, float, float, float] = WEIGHTED_FORM_COMPOSITE_WEIGHTS,
    history_component_weights: tuple[float, float] = V2_HISTORY_COMPONENT_WEIGHTS,
    strength_blend_weights: tuple[float, float, float] = V2_STRENGTH_BLEND_WEIGHTS,
    reference_edition_year: int = 2026,
    history_edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
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
    recent_history_df = build_recent_history_feature_table(
        base_df,
        reference_edition_year=reference_edition_year,
        edition_lookback=history_edition_lookback,
    )
    df = df.drop(
        columns=[
            column_name
            for column_name in [
                "world_cup_participations",
                "weighted_world_cup_participations",
                "weighted_world_cup_placement_score",
            ]
            if column_name in df.columns
        ],
        errors="ignore",
    ).merge(
        recent_history_df,
        on="team_id",
        how="left",
    )

    for column_name in ["weighted_world_cup_participations", "weighted_world_cup_placement_score", "world_cup_participations"]:
        if column_name not in df.columns:
            df[column_name] = 0.0
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)
    if "history_total_weight" not in df.columns:
        df["history_total_weight"] = WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT
    df["history_total_weight"] = pd.to_numeric(df["history_total_weight"], errors="coerce").fillna(WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT)
    df["world_cup_participations"] = df["world_cup_participations"].astype(int)
    df["weighted_world_cup_participation_ratio"] = (
        df["weighted_world_cup_participations"] / df["history_total_weight"].replace(0.0, np.nan)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
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


def build_v2_training_frame(
    match_window: int = RECENT_MATCH_WINDOW,
    exclude_editions: Iterable[int] = (),
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> pd.DataFrame:
    """Build the historical match-level training frame for the v2 multinomial model."""
    historical_results = load_historical_world_cup_results(exclude_editions=exclude_editions)
    if historical_results.empty:
        raise ValueError("Historical World Cup results are unavailable for v2 training")

    if edition_lookback > 0:
        included_editions = sorted(historical_results["edition"].dropna().astype(int).unique().tolist())[-int(edition_lookback) :]
        historical_results = historical_results[historical_results["edition"].isin(included_editions)].copy()
    if historical_results.empty:
        raise ValueError("Historical World Cup training frame is empty after edition lookback filtering")

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
            edition_lookback=edition_lookback,
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


@lru_cache(maxsize=16)
def fit_v2_match_multinomial_model(
    match_window: int = RECENT_MATCH_WINDOW,
    exclude_editions: tuple[int, ...] = (),
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> dict[str, object]:
    """Fit the sklearn multinomial model plus empirical scoreline samplers for v2."""
    normalized_exclusions = normalize_excluded_editions(exclude_editions)
    training_df = build_v2_training_frame(
        match_window=match_window,
        exclude_editions=normalized_exclusions,
        edition_lookback=edition_lookback,
    )
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
        "exclude_editions": normalized_exclusions,
        "edition_lookback": int(edition_lookback),
        "scoreline_distributions": build_v2_scoreline_distributions(training_df),
    }


def build_v2_match_feature_table(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    reference_edition_year: int = 2026,
    history_edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> pd.DataFrame:
    """Build the current-team feature table consumed by the v2 matchup model."""
    team_features = build_v2_team_strengths(
        base_df,
        lead_in_df,
        match_window=match_window,
        reference_edition_year=reference_edition_year,
        history_edition_lookback=history_edition_lookback,
    ).copy()
    form_lookup = build_weighted_form_feature_lookup(lead_in_df, "qualified_team_id", match_window=match_window)

    feature_rows = []
    for row in team_features.itertuples(index=False):
        form_snapshot = form_lookup.get(str(row.team_id), {})
        appearance_count = max(int(getattr(row, "world_cup_participations", 1)) - 1, 0)
        world_rank = pd.to_numeric(getattr(row, "world_rank", np.nan), errors="coerce")
        elo_rating = pd.to_numeric(getattr(row, "elo_rating", np.nan), errors="coerce")
        placement_score = pd.to_numeric(getattr(row, "weighted_world_cup_placement_score", np.nan), errors="coerce")
        feature_rows.append(
            {
                "team_id": str(row.team_id),
                "display_name": str(row.display_name),
                "flag_icon_code": str(row.flag_icon_code) if pd.notna(row.flag_icon_code) else "",
                "group_code": str(row.group_code),
                "confederation": str(row.confederation),
                "world_rank": int(world_rank) if pd.notna(world_rank) else 999,
                "elo_rating": float(elo_rating) if pd.notna(elo_rating) else 0.0,
                "team_strength": float(getattr(row, "v2_strength", 0.0)),
                "v2_strength": float(getattr(row, "v2_strength", 0.0)),
                "results_form": float(form_snapshot.get("results_form", 0.0)),
                "gd_form": float(form_snapshot.get("gd_form", 0.0)),
                "perf_vs_exp": float(form_snapshot.get("perf_vs_exp", 0.0)),
                "goals_for": float(form_snapshot.get("goals_for", 0.0)),
                "goals_against": float(form_snapshot.get("goals_against", 0.0)),
                "placement": float(placement_score) if pd.notna(placement_score) else 0.0,
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


def simulate_group_probabilities_v2_32team(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = BACKTEST_2022_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
    exclude_editions: Iterable[int] = (),
) -> pd.DataFrame:
    """Simulate a 32-team tournament using the V2 multinomial model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    normalized_exclusions = normalize_excluded_editions(exclude_editions)
    model_bundle = fit_v2_match_multinomial_model(
        match_window=match_window,
        exclude_editions=normalized_exclusions,
    )
    feature_df = build_v2_match_feature_table(
        base_df,
        lead_in_df,
        match_window=match_window,
        reference_edition_year=2022,
    )
    group_fixtures = extract_group_stage_fixtures(fixtures_df, group_order=group_order)
    knockout_fixtures = (
        extract_knockout_fixtures(fixtures_df)
        .loc[lambda df: df["round_code"].isin(BACKTEST_2022_MAIN_BRACKET_ROUND_CODES)]
        .reset_index(drop=True)
    )

    team_global_index = {team_id: idx for idx, team_id in enumerate(feature_df["team_id"])}
    team_feature_lookup = feature_df.set_index("team_id").to_dict("index")
    r16_counts = np.zeros(len(feature_df), dtype=np.int32)
    qf_counts = np.zeros(len(feature_df), dtype=np.int32)
    sf_counts = np.zeros(len(feature_df), dtype=np.int32)
    final_counts = np.zeros(len(feature_df), dtype=np.int32)
    champion_counts = np.zeros(len(feature_df), dtype=np.int32)
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
                    r16_counts[int(team_global_indices[team_idx])] += 1
            group_rankings[group_code] = [group_simulation["team_ids"][team_idx] for team_idx in ranked_indices]
            group_order_counts_by_group[group_code][tuple(group_rankings[group_code])] += 1

        match_results: dict[int, dict[str, str]] = {}
        for match in knockout_fixtures.itertuples(index=False):
            match_number = int(match.match_number)
            home_team_id = resolve_knockout_slot(
                match.home_slot_label,
                match_number,
                group_rankings,
                match_results,
                {},
            )
            away_team_id = resolve_knockout_slot(
                match.away_slot_label,
                match_number,
                group_rankings,
                match_results,
                {},
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
            if match.round_code == "R16":
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
    result_df["actual_format"] = "32-team"
    result_df.attrs["modal_group_rankings"] = {
        group_code: list(sorted(order_counter.items(), key=lambda item: (-item[1], item[0]))[0][0])
        for group_code, order_counter in group_order_counts_by_group.items()
        if order_counter
    }
    result_df.attrs["exclude_editions"] = normalized_exclusions
    return result_df


def build_deterministic_bracket_v2_32team(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable 32-team knockout bracket from modal group rankings and V2 matchup odds."""
    modal_group_rankings = get_modal_group_rankings(simulation_df)
    main_bracket_fixtures = (
        extract_knockout_fixtures(fixtures_df)
        .loc[lambda df: df["round_code"].isin(BACKTEST_2022_MAIN_BRACKET_ROUND_CODES)]
        .reset_index(drop=True)
    )
    team_feature_lookup = team_feature_df.set_index("team_id").to_dict("index")

    match_results: dict[int, dict[str, str]] = {}
    round_matches: dict[str, list[dict[str, object]]] = {round_code: [] for round_code in BACKTEST_2022_MAIN_BRACKET_ROUND_CODES}
    for match in main_bracket_fixtures.itertuples(index=False):
        match_number = int(match.match_number)
        home_team_id = resolve_knockout_slot(
            match.home_slot_label,
            match_number,
            modal_group_rankings,
            match_results,
            {},
        )
        away_team_id = resolve_knockout_slot(
            match.away_slot_label,
            match_number,
            modal_group_rankings,
            match_results,
            {},
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
        round_matches[str(match.round_code)].append(
            {
                "match_number": match_number,
                "round_code": str(match.round_code),
                "round_label": ROUND_CODE_LABELS[str(match.round_code)],
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
        "rounds": [
            {
                "round_code": round_code,
                "round_label": ROUND_CODE_LABELS[round_code],
                "matches": round_matches[round_code],
            }
            for round_code in BACKTEST_2022_MAIN_BRACKET_ROUND_CODES
        ],
    }


def run_v2_backtest_2022(
    match_window: int = RECENT_MATCH_WINDOW,
    simulations: int = 20000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Run a leakage-free V2 backtest against the actual 2022 World Cup."""
    dataset = build_2022_backtest_data()
    base_df = dataset["base_df"]
    lead_in_df = dataset["lead_in_df"]
    fixtures_df = dataset["fixtures_df"]
    results_df = dataset["results_df"]
    placement_df = dataset["placement_df"]
    group_code_lookup = dataset["group_code_lookup"]

    model_bundle = fit_v2_match_multinomial_model(
        match_window=match_window,
        exclude_editions=(2022,),
    )
    feature_df = build_v2_match_feature_table(
        base_df,
        lead_in_df,
        match_window=match_window,
        reference_edition_year=2022,
    )
    simulation_df = simulate_group_probabilities_v2_32team(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        seed=seed,
        match_window=match_window,
        exclude_editions=(2022,),
    )
    deterministic_bracket = build_deterministic_bracket_v2_32team(
        simulation_df,
        fixtures_df,
        feature_df,
        model_bundle,
        head_to_head_simulations=min(max(int(simulations // 5), 200), 4000),
        seed=seed,
    )

    feature_lookup = feature_df.set_index("team_id").to_dict("index")
    name_to_team_id = feature_df.set_index("display_name")["team_id"].astype(str).to_dict()
    match_rows: list[dict[str, object]] = []
    actual_probability_rows: list[tuple[float, float, float, str]] = []
    epsilon = 1e-15
    for row in results_df.sort_values(["match_number"], kind="stable").itertuples(index=False):
        home_team_id = name_to_team_id[str(row.home_team)]
        away_team_id = name_to_team_id[str(row.away_team)]
        probability_map = predict_match_probabilities_v2(
            home_team_id,
            away_team_id,
            feature_lookup,
            model_bundle,
        )
        actual_outcome = outcome_label_from_scoreline(int(row.home_score), int(row.away_score))
        probability_triplet = (
            float(probability_map["home_win_prob"]),
            float(probability_map["draw_prob"]),
            float(probability_map["away_win_prob"]),
        )
        predicted_outcome = V2_OUTCOME_LABELS[int(np.argmax(probability_triplet))]
        actual_probability_rows.append((*probability_triplet, actual_outcome))
        match_rows.append(
            {
                "match_number": int(row.match_number),
                "stage": str(row.stage),
                "group_code": group_code_lookup.get(str(row.home_team), "") if str(row.stage) == "Group Stage" else "",
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_score": int(row.home_score),
                "away_score": int(row.away_score),
                "home_win_prob": probability_triplet[0],
                "draw_prob": probability_triplet[1],
                "away_win_prob": probability_triplet[2],
                "predicted_outcome": predicted_outcome,
                "actual_outcome": actual_outcome,
                "top1_correct": predicted_outcome == actual_outcome,
            }
        )
    match_predictions = pd.DataFrame(match_rows)

    y_true = np.array(
        [
            [1.0 if label == "home_win" else 0.0, 1.0 if label == "draw" else 0.0, 1.0 if label == "away_win" else 0.0]
            for _, _, _, label in actual_probability_rows
        ],
        dtype=float,
    )
    y_pred = np.array([[home, draw, away] for home, draw, away, _ in actual_probability_rows], dtype=float)
    true_class_indices = np.argmax(y_true, axis=1)
    multiclass_log_loss = float(-np.mean(np.log(np.clip(y_pred[np.arange(len(y_pred)), true_class_indices], epsilon, 1.0))))
    multiclass_brier_score = float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))
    top1_match_accuracy = float(match_predictions["top1_correct"].mean() * 100.0)

    actual_group_standings = build_2022_actual_group_standings(results_df, group_code_lookup, feature_df)
    actual_group_rank_lookup = actual_group_standings.set_index("team_id")["actual_group_rank"].astype(int).to_dict()
    modal_group_rankings = get_modal_group_rankings(simulation_df)
    modal_group_rank_lookup = {
        team_id: rank
        for _, ranked_team_ids in modal_group_rankings.items()
        for rank, team_id in enumerate(ranked_team_ids, start=1)
    }

    placement_df = placement_df.copy()
    placement_df["team_id"] = placement_df["country"].map(name_to_team_id)
    placement_df["actual_stage"] = placement_df["position"].map(stage_label_from_position)
    actual_stage_lookup = placement_df.set_index("team_id")["actual_stage"].astype(str).to_dict()
    actual_position_lookup = placement_df.set_index("team_id")["position"].astype(int).to_dict()
    actual_r16_team_ids = set(placement_df.loc[placement_df["position"] <= 16, "team_id"].dropna().astype(str))
    actual_semifinalist_team_ids = set(placement_df.loc[placement_df["position"] <= 4, "team_id"].dropna().astype(str))
    actual_finalist_team_ids = set(placement_df.loc[placement_df["position"] <= 2, "team_id"].dropna().astype(str))
    actual_champion_team_id = str(placement_df.loc[placement_df["position"] == 1, "team_id"].iloc[0])

    team_backtest_table = simulation_df.copy()
    team_backtest_table["actual_group_rank"] = team_backtest_table["team_id"].map(actual_group_rank_lookup)
    team_backtest_table["modal_group_rank"] = team_backtest_table["team_id"].map(modal_group_rank_lookup)
    team_backtest_table["actual_position"] = team_backtest_table["team_id"].map(actual_position_lookup)
    team_backtest_table["actual_stage"] = team_backtest_table["team_id"].map(actual_stage_lookup)
    team_backtest_table["actual_r16"] = team_backtest_table["team_id"].isin(actual_r16_team_ids)
    team_backtest_table["actual_sf"] = team_backtest_table["team_id"].isin(actual_semifinalist_team_ids)
    team_backtest_table["actual_final"] = team_backtest_table["team_id"].isin(actual_finalist_team_ids)
    team_backtest_table["actual_champion"] = team_backtest_table["team_id"].eq(actual_champion_team_id)

    group_backtest_table = team_backtest_table.loc[
        :,
        [
            "group_code",
            "team_id",
            "display_name",
            "prob_1",
            "prob_2",
            "prob_3",
            "prob_4",
            "modal_group_rank",
            "actual_group_rank",
        ],
    ].sort_values(["group_code", "actual_group_rank", "display_name"], kind="stable").reset_index(drop=True)

    predicted_r16_team_ids = set(
        team_backtest_table.sort_values(["r16_prob", "team_strength", "display_name"], ascending=[False, False, True], kind="stable")
        .head(16)["team_id"]
        .astype(str)
        .tolist()
    )
    predicted_semifinalist_team_ids = set(
        team_backtest_table.sort_values(["sf_prob", "team_strength", "display_name"], ascending=[False, False, True], kind="stable")
        .head(4)["team_id"]
        .astype(str)
        .tolist()
    )
    predicted_champion_team_id = str(
        team_backtest_table.sort_values(["champion_prob", "team_strength", "display_name"], ascending=[False, False, True], kind="stable")
        .iloc[0]["team_id"]
    )

    bracket_round_lookup = {round_data["round_code"]: round_data["matches"] for round_data in deterministic_bracket["rounds"]}
    predicted_finalists = sorted(
        {
            str(match["home_team_id"])
            for match in bracket_round_lookup.get("F", [])
        }.union(
            {
                str(match["away_team_id"])
                for match in bracket_round_lookup.get("F", [])
            }
        )
    )
    predicted_semifinalists = sorted(
        {
            str(match["home_team_id"])
            for match in bracket_round_lookup.get("SF", [])
        }.union(
            {
                str(match["away_team_id"])
                for match in bracket_round_lookup.get("SF", [])
            }
        )
    )
    bracket_summary = {
        "predicted_champion_team_id": str(bracket_round_lookup.get("F", [{}])[-1].get("winner_team_id", predicted_champion_team_id)),
        "predicted_finalist_team_ids": predicted_finalists,
        "predicted_semifinalist_team_ids": predicted_semifinalists,
        "actual_champion_team_id": actual_champion_team_id,
        "actual_finalist_team_ids": sorted(actual_finalist_team_ids),
        "actual_semifinalist_team_ids": sorted(actual_semifinalist_team_ids),
        "rounds": deterministic_bracket["rounds"],
    }

    summary_metrics = {
        "multiclass_log_loss": multiclass_log_loss,
        "multiclass_brier_score": multiclass_brier_score,
        "top1_match_accuracy": top1_match_accuracy,
        "exact_champion_hit": int(predicted_champion_team_id == actual_champion_team_id),
        "semifinal_hit_count": int(len(predicted_semifinalist_team_ids.intersection(actual_semifinalist_team_ids))),
        "round_of_16_hit_count": int(len(predicted_r16_team_ids.intersection(actual_r16_team_ids))),
        "predicted_champion_team_id": predicted_champion_team_id,
        "actual_champion_team_id": actual_champion_team_id,
    }

    return {
        "summary_metrics": summary_metrics,
        "match_predictions": match_predictions,
        "team_backtest_table": team_backtest_table.sort_values(
            ["champion_prob", "sf_prob", "qf_prob", "r16_prob", "display_name"],
            ascending=[False, False, False, False, True],
            kind="stable",
        ).reset_index(drop=True),
        "group_backtest_table": group_backtest_table,
        "bracket_summary": bracket_summary,
    }


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


__all__ = [
    'build_v2_team_strengths',
    'build_v2_training_frame',
    'build_v2_scoreline_distributions',
    'fit_v2_match_multinomial_model',
    'build_v2_match_feature_table',
    'predict_match_probabilities_v2',
    'sample_scoreline_v2',
    'simulate_knockout_match_v2',
    'predict_knockout_matchup_v2',
    'build_deterministic_bracket_v2',
    'simulate_group_probabilities_v2_32team',
    'build_deterministic_bracket_v2_32team',
    'run_v2_backtest_2022',
    'simulate_group_probabilities_v2',
]
