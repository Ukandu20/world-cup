from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import *  # noqa: F403
from .shared import *  # noqa: F403


def normalize_excluded_tournaments(exclude_tournament: str | Iterable[str] | None = None) -> tuple[str, ...]:
    """Normalize tournament exclusions into a stable tuple of text keys."""
    if exclude_tournament is None:
        return ()
    if isinstance(exclude_tournament, str):
        values = [exclude_tournament]
    else:
        values = list(exclude_tournament)
    return tuple(sorted({normalize_key(value) for value in values if normalize_key(value)}))


def classify_competition_importance(tournament: str | None) -> float:
    """Map a tournament name onto the fixed V3 competition-importance scale."""
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


def is_neutral_site(value: object) -> bool:
    """Interpret common boolean and text representations of a neutral-site flag."""
    if isinstance(value, bool):
        return value
    normalized = normalize_key(str(value))
    return normalized in {"true", "1", "yes", "y"}


def infer_v3_host_flag(
    team_id: str,
    display_name: str,
    canonical_name: str,
    reference_edition_year: int,
    explicit_is_host: object = None,
) -> float:
    """Infer whether a team is one of the known tournament hosts for the reference edition."""
    if pd.notna(explicit_is_host):
        normalized = normalize_key(str(explicit_is_host))
        if normalized in {"true", "1", "yes"}:
            return 1.0
        if normalized in {"false", "0", "no"}:
            return 0.0

    team_id_normalized = str(team_id).strip().upper()
    team_name_key = normalize_historical_team_name(display_name or canonical_name)
    if int(reference_edition_year) == 2026 and team_id_normalized in V3_2026_HOST_TEAM_IDS:
        return 1.0
    if int(reference_edition_year) == 2022 and (
        team_id_normalized in V3_2022_HOST_TEAM_IDS or team_name_key == "qatar"
    ):
        return 1.0
    return 0.0


def build_v3_strength_score(
    elo_rating: float,
    results_form: float,
    gd_form: float,
    perf_vs_exp: float,
    goals_for: float,
    goals_against: float,
    placement: float,
    appearance: float,
    host_flag: float,
) -> float:
    """Build a stable scalar fallback used for tie-breaks and display ordering."""
    return float(
        elo_rating
        + 120.0 * results_form
        + 18.0 * gd_form
        + 80.0 * perf_vs_exp
        + 22.0 * goals_for
        - 18.0 * goals_against
        + 60.0 * placement
        + 8.0 * appearance
        + 25.0 * host_flag
    )


def build_v3_team_feature_table(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    reference_date_or_edition: int | str | pd.Timestamp,
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> pd.DataFrame:
    """Build the current-team V3 feature table consumed by the Poisson matchup model."""
    if isinstance(reference_date_or_edition, pd.Timestamp):
        reference_year = int(reference_date_or_edition.year)
    else:
        try:
            reference_year = int(reference_date_or_edition)
        except (TypeError, ValueError):
            reference_year = int(pd.Timestamp(reference_date_or_edition).year)

    history_df = build_recent_history_feature_table(
        base_df,
        reference_edition_year=reference_year,
        edition_lookback=edition_lookback,
    )
    history_lookup = history_df.set_index("team_id").to_dict("index") if not history_df.empty else {}
    form_lookup = build_weighted_form_feature_lookup(lead_in_df, "qualified_team_id", match_window=match_window)

    rows: list[dict[str, object]] = []
    for row in base_df.itertuples(index=False):
        team_id = str(getattr(row, "team_id"))
        display_name = str(getattr(row, "display_name", getattr(row, "tournament_name", team_id)))
        canonical_name = str(getattr(row, "canonical_name", display_name))
        form_snapshot = form_lookup.get(team_id, {})
        history_snapshot = history_lookup.get(team_id, {})

        elo_rating = pd.to_numeric(getattr(row, "elo_rating", np.nan), errors="coerce")
        if pd.isna(elo_rating) or float(elo_rating) == 0.0:
            elo_rating = float(form_snapshot.get("pre_tournament_elo", 0.0))
        placement_score = pd.to_numeric(
            history_snapshot.get(
                "weighted_world_cup_placement_score",
                getattr(row, "weighted_world_cup_placement_score", 0.0),
            ),
            errors="coerce",
        )
        appearance_count = pd.to_numeric(
            history_snapshot.get(
                "world_cup_participations",
                getattr(row, "world_cup_participations", 1),
            ),
            errors="coerce",
        )
        host_flag = infer_v3_host_flag(
            team_id=team_id,
            display_name=display_name,
            canonical_name=canonical_name,
            reference_edition_year=reference_year,
            explicit_is_host=getattr(row, "is_host", None),
        )
        results_form = float(form_snapshot.get("results_form", 0.0))
        gd_form = float(form_snapshot.get("gd_form", 0.0))
        perf_vs_exp = float(form_snapshot.get("perf_vs_exp", 0.0))
        goals_for = float(form_snapshot.get("goals_for", 0.0))
        goals_against = float(form_snapshot.get("goals_against", 0.0))
        appearance = max(float(appearance_count) - 1.0, 0.0) if pd.notna(appearance_count) else 0.0
        placement = float(placement_score) if pd.notna(placement_score) else 0.0
        v3_strength = build_v3_strength_score(
            elo_rating=float(elo_rating) if pd.notna(elo_rating) else 0.0,
            results_form=results_form,
            gd_form=gd_form,
            perf_vs_exp=perf_vs_exp,
            goals_for=goals_for,
            goals_against=goals_against,
            placement=placement,
            appearance=appearance,
            host_flag=host_flag,
        )
        rows.append(
            {
                "team_id": team_id,
                "display_name": display_name,
                "flag_icon_code": str(getattr(row, "flag_icon_code", "")) if pd.notna(getattr(row, "flag_icon_code", "")) else "",
                "group_code": str(getattr(row, "group_code", "")),
                "confederation": str(getattr(row, "confederation", "")),
                "world_rank": int(pd.to_numeric(getattr(row, "world_rank", 999), errors="coerce") or 999),
                "elo_rating": float(elo_rating) if pd.notna(elo_rating) else 0.0,
                "team_strength": v3_strength,
                "v3_strength": v3_strength,
                "results_form": results_form,
                "gd_form": gd_form,
                "perf_vs_exp": perf_vs_exp,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "placement": placement,
                "appearance": appearance,
                "host_flag": host_flag,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["team_strength", "elo_rating", "display_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_v3_training_frame(
    results_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
    start_year: int = V3_MATCH_START_YEAR,
    end_date: str | pd.Timestamp | None = None,
    exclude_tournament: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build the historical match-level V3 training frame from broad international results."""
    if results_df.empty:
        raise ValueError("results_df must include historical international matches for V3")

    cutoff = pd.Timestamp(end_date) if end_date is not None else None
    excluded_tournaments = set(normalize_excluded_tournaments(exclude_tournament))

    training_source = results_df.copy()
    training_source["date"] = pd.to_datetime(training_source["date"], errors="coerce")
    training_source["home_score"] = pd.to_numeric(training_source["home_score"], errors="coerce")
    training_source["away_score"] = pd.to_numeric(training_source["away_score"], errors="coerce")
    training_source = training_source.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"]).copy()
    training_source = training_source[training_source["date"].dt.year >= int(start_year)].copy()
    if cutoff is not None:
        training_source = training_source[training_source["date"] <= cutoff].copy()
    if excluded_tournaments:
        training_source["_tournament_key"] = training_source["tournament"].map(normalize_key)
        training_source = training_source[~training_source["_tournament_key"].isin(excluded_tournaments)].copy()
    if training_source.empty:
        raise ValueError("V3 training frame is empty after date and tournament filtering")

    country_results_lookup = load_historical_country_results_lookup()
    placement_df, edition_team_counts, edition_weight_map = load_historical_placement_history()

    rows: list[dict[str, object]] = []
    for match in training_source.sort_values(["date", "home_team", "away_team"], kind="stable").itertuples(index=False):
        match_date = pd.Timestamp(match.date)
        home_key = normalize_historical_team_name(str(match.home_team))
        away_key = normalize_historical_team_name(str(match.away_team))
        home_results = country_results_lookup.get(home_key, pd.DataFrame())
        away_results = country_results_lookup.get(away_key, pd.DataFrame())
        home_prior = home_results[home_results["date"] < match_date].copy() if not home_results.empty else pd.DataFrame()
        away_prior = away_results[away_results["date"] < match_date].copy() if not away_results.empty else pd.DataFrame()

        home_form = compute_weighted_form_snapshot(home_prior, match_window=match_window)
        away_form = compute_weighted_form_snapshot(away_prior, match_window=match_window)
        home_history = compute_pre_tournament_history_features(
            home_key,
            int(match_date.year),
            placement_df,
            edition_team_counts,
            edition_weight_map,
            edition_lookback=edition_lookback,
        )
        away_history = compute_pre_tournament_history_features(
            away_key,
            int(match_date.year),
            placement_df,
            edition_team_counts,
            edition_weight_map,
            edition_lookback=edition_lookback,
        )

        neutral_site_flag = 1.0 if is_neutral_site(getattr(match, "neutral", False)) else 0.0
        match_country_key = normalize_historical_team_name(getattr(match, "country", ""))
        home_host_flag = 1.0 if not neutral_site_flag and match_country_key == home_key else 0.0
        away_host_flag = 1.0 if not neutral_site_flag and match_country_key == away_key else 0.0

        rows.append(
            {
                "date": match_date,
                "home_team": str(match.home_team),
                "away_team": str(match.away_team),
                "tournament": str(getattr(match, "tournament", "")),
                "home_score": int(match.home_score),
                "away_score": int(match.away_score),
                "outcome_label": outcome_label_from_scoreline(int(match.home_score), int(match.away_score)),
                "elo_diff": float(home_form.get("pre_tournament_elo", 0.0)) - float(away_form.get("pre_tournament_elo", 0.0)),
                "results_form_diff": float(home_form.get("results_form", 0.0)) - float(away_form.get("results_form", 0.0)),
                "goals_for_diff": float(home_form.get("goals_for", 0.0)) - float(away_form.get("goals_for", 0.0)),
                "goals_against_diff": float(home_form.get("goals_against", 0.0)) - float(away_form.get("goals_against", 0.0)),
                "placement_diff": float(home_history.get("placement", 0.0)) - float(away_history.get("placement", 0.0)),
                "appearance_diff": float(home_history.get("appearance", 0.0)) - float(away_history.get("appearance", 0.0)),
                "gd_form_diff": float(home_form.get("gd_form", 0.0)) - float(away_form.get("gd_form", 0.0)),
                "perf_vs_exp_diff": float(home_form.get("perf_vs_exp", 0.0)) - float(away_form.get("perf_vs_exp", 0.0)),
                "competition_importance": classify_competition_importance(getattr(match, "tournament", "")),
                "neutral_site_flag": neutral_site_flag,
                "net_host_flag": home_host_flag - away_host_flag,
                "sample_weight": classify_competition_importance(getattr(match, "tournament", "")),
            }
        )

    training_df = pd.DataFrame(rows)
    if training_df.empty:
        raise ValueError("V3 training frame is empty")
    return training_df


@lru_cache(maxsize=16)
def fit_v3_poisson_models(
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
    start_year: int = V3_MATCH_START_YEAR,
    end_date: str | None = None,
    exclude_tournament: str | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Fit and cache the pair of Poisson goal models used by V3."""
    results_path = INTERNATIONAL_RESULTS_PATH
    if not results_path.exists():
        raise ValueError("Historical international results are unavailable for V3 training")
    normalized_exclusions = normalize_excluded_tournaments(exclude_tournament)
    training_df = build_v3_training_frame(
        pd.read_csv(results_path),
        match_window=match_window,
        edition_lookback=edition_lookback,
        start_year=start_year,
        end_date=end_date,
        exclude_tournament=normalized_exclusions,
    )
    try:
        from sklearn.linear_model import PoissonRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
        raise ImportError("scikit-learn is required for the V3 Poisson simulator") from exc

    X = training_df.loc[:, list(V3_FEATURE_COLUMNS)].astype(float)
    sample_weight = training_df["sample_weight"].astype(float).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    home_goal_model = PoissonRegressor(alpha=0.1, max_iter=1000)
    away_goal_model = PoissonRegressor(alpha=0.1, max_iter=1000)
    home_goal_model.fit(X_scaled, training_df["home_score"].astype(float).to_numpy(), sample_weight=sample_weight)
    away_goal_model.fit(X_scaled, training_df["away_score"].astype(float).to_numpy(), sample_weight=sample_weight)

    return {
        "training_frame": training_df,
        "feature_columns": V3_FEATURE_COLUMNS,
        "scaler": scaler,
        "home_goal_model": home_goal_model,
        "away_goal_model": away_goal_model,
        "match_window": int(match_window),
        "edition_lookback": int(edition_lookback),
        "start_year": int(start_year),
        "end_date": end_date,
        "exclude_tournament": normalized_exclusions,
    }


def poisson_probability_vector(lambda_value: float, goal_cap: int = V3_POISSON_GOAL_CAP) -> np.ndarray:
    """Return Poisson probabilities 0..goal_cap, folding the tail into the final bucket."""
    lambda_value = float(np.clip(lambda_value, V3_LAMBDA_MIN, V3_LAMBDA_MAX))
    probabilities = np.zeros(goal_cap + 1, dtype=float)
    probabilities[0] = float(np.exp(-lambda_value))
    running_total = probabilities[0]
    for goals in range(1, goal_cap):
        probabilities[goals] = probabilities[goals - 1] * lambda_value / float(goals)
        running_total += probabilities[goals]
    probabilities[goal_cap] = max(0.0, 1.0 - running_total)
    probabilities /= probabilities.sum()
    return probabilities


def build_v3_probability_triplet(lambda_home: float, lambda_away: float) -> tuple[float, float, float]:
    """Convert home and away Poisson means into implied home/draw/away probabilities."""
    home_probabilities = poisson_probability_vector(lambda_home)
    away_probabilities = poisson_probability_vector(lambda_away)
    score_matrix = np.outer(home_probabilities, away_probabilities)
    draw_prob = float(np.trace(score_matrix))
    home_win_prob = float(np.tril(score_matrix, k=-1).sum())
    away_win_prob = float(np.triu(score_matrix, k=1).sum())
    total = home_win_prob + draw_prob + away_win_prob
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return home_win_prob / total, draw_prob / total, away_win_prob / total


def predict_match_lambdas_v3(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    neutral_site: bool = True,
) -> dict[str, float | str]:
    """Predict home and away expected goals plus implied W/D/L probabilities for one matchup."""
    home_features = team_feature_lookup[str(home_team_id)]
    away_features = team_feature_lookup[str(away_team_id)]
    feature_row = pd.DataFrame(
        [
            {
                "elo_diff": float(home_features["elo_rating"]) - float(away_features["elo_rating"]),
                "results_form_diff": float(home_features["results_form"]) - float(away_features["results_form"]),
                "goals_for_diff": float(home_features["goals_for"]) - float(away_features["goals_for"]),
                "goals_against_diff": float(home_features["goals_against"]) - float(away_features["goals_against"]),
                "placement_diff": float(home_features["placement"]) - float(away_features["placement"]),
                "appearance_diff": float(home_features["appearance"]) - float(away_features["appearance"]),
                "gd_form_diff": float(home_features["gd_form"]) - float(away_features["gd_form"]),
                "perf_vs_exp_diff": float(home_features["perf_vs_exp"]) - float(away_features["perf_vs_exp"]),
                "competition_importance": float(V3_COMPETITION_IMPORTANCE["world_cup_finals"]),
                "neutral_site_flag": 1.0 if neutral_site else 0.0,
                "net_host_flag": float(home_features.get("host_flag", 0.0)) - float(away_features.get("host_flag", 0.0)),
            }
        ],
        columns=list(model_bundle["feature_columns"]),
    )
    scaled = model_bundle["scaler"].transform(feature_row)
    lambda_home = float(np.clip(model_bundle["home_goal_model"].predict(scaled)[0], V3_LAMBDA_MIN, V3_LAMBDA_MAX))
    lambda_away = float(np.clip(model_bundle["away_goal_model"].predict(scaled)[0], V3_LAMBDA_MIN, V3_LAMBDA_MAX))
    home_win_prob, draw_prob, away_win_prob = build_v3_probability_triplet(lambda_home, lambda_away)
    return {
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
    }


def simulate_knockout_match_v3(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    rng: np.random.Generator,
    matchup_probability_cache: dict[tuple[str, str], dict[str, float | str]] | None = None,
) -> tuple[str, str]:
    """Simulate one knockout matchup from Poisson regulation and extra-time goal models."""
    cache_key = (str(home_team_id), str(away_team_id))
    if matchup_probability_cache is not None and cache_key in matchup_probability_cache:
        probability_map = matchup_probability_cache[cache_key]
    else:
        neutral_site = not (
            float(team_feature_lookup[str(home_team_id)].get("host_flag", 0.0))
            or float(team_feature_lookup[str(away_team_id)].get("host_flag", 0.0))
        )
        probability_map = predict_match_lambdas_v3(
            home_team_id,
            away_team_id,
            team_feature_lookup,
            model_bundle,
            neutral_site=neutral_site,
        )
        if matchup_probability_cache is not None:
            matchup_probability_cache[cache_key] = probability_map

    regulation_home = int(rng.poisson(float(probability_map["lambda_home"])))
    regulation_away = int(rng.poisson(float(probability_map["lambda_away"])))
    if regulation_home > regulation_away:
        return str(home_team_id), str(away_team_id)
    if regulation_away > regulation_home:
        return str(away_team_id), str(home_team_id)

    extra_home = int(rng.poisson(float(probability_map["lambda_home"]) * EXTRA_TIME_FACTOR))
    extra_away = int(rng.poisson(float(probability_map["lambda_away"]) * EXTRA_TIME_FACTOR))
    if extra_home > extra_away:
        return str(home_team_id), str(away_team_id)
    if extra_away > extra_home:
        return str(away_team_id), str(home_team_id)

    if bool(rng.integers(0, 2)):
        return str(home_team_id), str(away_team_id)
    return str(away_team_id), str(home_team_id)


def predict_knockout_matchup_v3(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, float | str]:
    """Estimate one knockout matchup with repeated V3 Poisson simulations."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    rng = np.random.default_rng(seed)
    matchup_probability_cache: dict[tuple[str, str], dict[str, float | str]] = {}
    home_wins = 0
    for _ in range(simulations):
        winner_team_id, _ = simulate_knockout_match_v3(
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
        winner_team_id = str(home_team_id)
        winner_win_prob = home_win_prob
    else:
        winner_team_id = str(away_team_id)
        winner_win_prob = away_win_prob
    return {
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "winner_team_id": winner_team_id,
        "winner_win_prob": winner_win_prob,
    }


def build_deterministic_bracket_v3(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable V3 knockout bracket from modal group rankings and Poisson matchup odds."""
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
        prediction = predict_knockout_matchup_v3(
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


def build_deterministic_bracket_v3_32team(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable 32-team V3 knockout bracket from modal group rankings."""
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
        home_team_id = resolve_knockout_slot(match.home_slot_label, match_number, modal_group_rankings, match_results, {})
        away_team_id = resolve_knockout_slot(match.away_slot_label, match_number, modal_group_rankings, match_results, {})
        prediction = predict_knockout_matchup_v3(
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


def simulate_group_probabilities_v3(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = DEFAULT_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
) -> pd.DataFrame:
    """Simulate the 2026 tournament using the V3 Poisson expected-goals model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    model_bundle = fit_v3_poisson_models(match_window=match_window)
    feature_df = build_v3_team_feature_table(base_df, lead_in_df, reference_date_or_edition=2026, match_window=match_window)
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
        fixture_pairs = np.array([(team_index[row.home_team_id], team_index[row.away_team_id]) for row in fixtures.itertuples(index=False)], dtype=int)

        rng = np.random.default_rng(seed + ord(group_code))
        simulated_home_goals = np.zeros((simulations, len(fixtures)), dtype=np.int16)
        simulated_away_goals = np.zeros((simulations, len(fixtures)), dtype=np.int16)
        points = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_for = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_against = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        simulation_indices = np.arange(simulations)

        for match_index, match in enumerate(fixtures.itertuples(index=False)):
            neutral_site = not (
                float(team_feature_lookup[str(match.home_team_id)].get("host_flag", 0.0))
                or float(team_feature_lookup[str(match.away_team_id)].get("host_flag", 0.0))
            )
            probability_map = predict_match_lambdas_v3(
                str(match.home_team_id),
                str(match.away_team_id),
                team_feature_lookup,
                model_bundle,
                neutral_site=neutral_site,
            )
            home_scores = rng.poisson(float(probability_map["lambda_home"]), size=simulations).astype(np.int16)
            away_scores = rng.poisson(float(probability_map["lambda_away"]), size=simulations).astype(np.int16)
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
                home_team_id = resolve_knockout_slot(match.home_slot_label, match_number, group_rankings, match_results, third_place_routing)
                away_team_id = resolve_knockout_slot(match.away_slot_label, match_number, group_rankings, match_results, third_place_routing)
                winner_team_id, loser_team_id = simulate_knockout_match_v3(
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
        probability_frame = pd.DataFrame({f"prob_{place + 1}": finish_counts[:, place] / simulations * 100.0 for place in range(len(team_ids))})
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
    result_df.attrs["modal_group_rankings"] = {
        group_code: list(sorted(order_counter.items(), key=lambda item: (-item[1], item[0]))[0][0])
        for group_code, order_counter in group_order_counts_by_group.items()
        if order_counter
    }
    result_df.attrs["average_third_place_stats"] = {
        team_id: {
            "points": third_place_points_sum[global_index] / finish_count,
            "goal_difference": third_place_gd_sum[global_index] / finish_count,
            "goals_for": third_place_gf_sum[global_index] / finish_count,
            "team_strength": float(team_strength_lookup[team_id]),
        }
        for team_id, global_index in team_global_index.items()
        for finish_count in [int(third_place_finish_counts[global_index])]
        if finish_count > 0
    }
    return result_df


def simulate_group_probabilities_v3_32team(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = BACKTEST_2022_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
    training_end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Simulate a 32-team tournament using the V3 Poisson expected-goals model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    model_bundle = fit_v3_poisson_models(match_window=match_window, end_date=None if training_end_date is None else str(pd.Timestamp(training_end_date).date()))
    feature_df = build_v3_team_feature_table(base_df, lead_in_df, reference_date_or_edition=2022, match_window=match_window)
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
        fixture_pairs = np.array([(team_index[row.home_team_id], team_index[row.away_team_id]) for row in fixtures.itertuples(index=False)], dtype=int)

        rng = np.random.default_rng(seed + ord(group_code))
        simulated_home_goals = np.zeros((simulations, len(fixtures)), dtype=np.int16)
        simulated_away_goals = np.zeros((simulations, len(fixtures)), dtype=np.int16)
        points = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_for = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        goals_against = np.zeros((simulations, len(team_ids)), dtype=np.int16)
        simulation_indices = np.arange(simulations)

        for match_index, match in enumerate(fixtures.itertuples(index=False)):
            neutral_site = not (
                float(team_feature_lookup[str(match.home_team_id)].get("host_flag", 0.0))
                or float(team_feature_lookup[str(match.away_team_id)].get("host_flag", 0.0))
            )
            probability_map = predict_match_lambdas_v3(
                str(match.home_team_id),
                str(match.away_team_id),
                team_feature_lookup,
                model_bundle,
                neutral_site=neutral_site,
            )
            home_scores = rng.poisson(float(probability_map["lambda_home"]), size=simulations).astype(np.int16)
            away_scores = rng.poisson(float(probability_map["lambda_away"]), size=simulations).astype(np.int16)
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
            home_team_id = resolve_knockout_slot(match.home_slot_label, match_number, group_rankings, match_results, {})
            away_team_id = resolve_knockout_slot(match.away_slot_label, match_number, group_rankings, match_results, {})
            winner_team_id, loser_team_id = simulate_knockout_match_v3(
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
        probability_frame = pd.DataFrame({f"prob_{place + 1}": finish_counts[:, place] / simulations * 100.0 for place in range(len(team_ids))})
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
    return result_df


def run_v3_2022_backtest(
    match_window: int = RECENT_MATCH_WINDOW,
    simulations: int = 20000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Run a leakage-free V3 backtest against the actual 2022 World Cup."""
    dataset = build_2022_backtest_data()
    base_df = dataset["base_df"]
    lead_in_df = dataset["lead_in_df"]
    fixtures_df = dataset["fixtures_df"]
    results_df = dataset["results_df"]
    placement_df = dataset["placement_df"]
    group_code_lookup = dataset["group_code_lookup"]
    edition_start = pd.to_datetime(pd.DataFrame(results_df)["date"], errors="coerce").min()
    training_end_date = None if pd.isna(edition_start) else str((pd.Timestamp(edition_start) - pd.Timedelta(days=1)).date())

    model_bundle = fit_v3_poisson_models(
        match_window=match_window,
        end_date=training_end_date,
    )
    feature_df = build_v3_team_feature_table(
        base_df,
        lead_in_df,
        reference_date_or_edition=2022,
        match_window=match_window,
    )
    simulation_df = simulate_group_probabilities_v3_32team(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        seed=seed,
        match_window=match_window,
        training_end_date=training_end_date,
    )
    deterministic_bracket = build_deterministic_bracket_v3_32team(
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
    for row in pd.DataFrame(results_df).sort_values(["match_number"], kind="stable").itertuples(index=False):
        home_team_id = name_to_team_id[str(row.home_team)]
        away_team_id = name_to_team_id[str(row.away_team)]
        neutral_site = not (
            float(feature_lookup[home_team_id].get("host_flag", 0.0))
            or float(feature_lookup[away_team_id].get("host_flag", 0.0))
        )
        probability_map = predict_match_lambdas_v3(
            home_team_id,
            away_team_id,
            feature_lookup,
            model_bundle,
            neutral_site=neutral_site,
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
                "lambda_home": float(probability_map["lambda_home"]),
                "lambda_away": float(probability_map["lambda_away"]),
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
        [[1.0 if label == "home_win" else 0.0, 1.0 if label == "draw" else 0.0, 1.0 if label == "away_win" else 0.0] for _, _, _, label in actual_probability_rows],
        dtype=float,
    )
    y_pred = np.array([[home, draw, away] for home, draw, away, _ in actual_probability_rows], dtype=float)
    true_class_indices = np.argmax(y_true, axis=1)
    multiclass_log_loss = float(-np.mean(np.log(np.clip(y_pred[np.arange(len(y_pred)), true_class_indices], epsilon, 1.0))))
    multiclass_brier_score = float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))
    top1_match_accuracy = float(match_predictions["top1_correct"].mean() * 100.0)
    actual_draw_rate = float(y_true[:, 1].mean() * 100.0)
    predicted_draw_rate = float(y_pred[:, 1].mean() * 100.0)

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
        ["group_code", "team_id", "display_name", "prob_1", "prob_2", "prob_3", "prob_4", "modal_group_rank", "actual_group_rank"],
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
    predicted_finalists = sorted({str(match["home_team_id"]) for match in bracket_round_lookup.get("F", [])}.union({str(match["away_team_id"]) for match in bracket_round_lookup.get("F", [])}))
    predicted_semifinalists = sorted({str(match["home_team_id"]) for match in bracket_round_lookup.get("SF", [])}.union({str(match["away_team_id"]) for match in bracket_round_lookup.get("SF", [])}))
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
        "draw_rate_actual": actual_draw_rate,
        "draw_rate_predicted": predicted_draw_rate,
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


__all__ = [
    'normalize_excluded_tournaments',
    'classify_competition_importance',
    'is_neutral_site',
    'infer_v3_host_flag',
    'build_v3_strength_score',
    'build_v3_team_feature_table',
    'build_v3_training_frame',
    'fit_v3_poisson_models',
    'poisson_probability_vector',
    'build_v3_probability_triplet',
    'predict_match_lambdas_v3',
    'simulate_knockout_match_v3',
    'predict_knockout_matchup_v3',
    'build_deterministic_bracket_v3',
    'build_deterministic_bracket_v3_32team',
    'simulate_group_probabilities_v3',
    'simulate_group_probabilities_v3_32team',
    'run_v3_2022_backtest',
]
