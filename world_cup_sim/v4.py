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


def is_neutral_site(value: object) -> bool:
    """Interpret common boolean and text representations of a neutral-site flag."""
    if isinstance(value, bool):
        return value
    normalized = normalize_key(str(value))
    return normalized in {"true", "1", "yes", "y"}


def quadratic_recency_weights(length: int) -> np.ndarray:
    """Return quadratic recency weights 1^2..n^2 for oldest-to-newest rows."""
    if length <= 0:
        return np.array([], dtype=float)
    values = np.arange(1, int(length) + 1, dtype=float)
    return values * values


def compute_quadratic_form_snapshot(results_df: pd.DataFrame, match_window: int = RECENT_MATCH_WINDOW) -> dict[str, float]:
    """Summarize the last k Elo-rated matches with V4 quadratic recency weights."""
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
    for column_name in ("team_score", "opponent_score", "team_elo_start", "opponent_elo_start"):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df = df.dropna(subset=["date", "team_score", "opponent_score", "team_elo_start", "opponent_elo_start"]).copy()
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

    weights = quadratic_recency_weights(len(df))
    total_weight = float(weights.sum())
    df["goal_difference"] = df["team_score"] - df["opponent_score"]
    df["gd_capped"] = df["goal_difference"].clip(
        lower=-WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
        upper=WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
    )
    df["actual_score"] = df["normalized_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0}).astype(float)
    df["expected_score"] = compute_elo_expected_score(df["team_elo_start"], df["opponent_elo_start"]).astype(float)
    df["perf_vs_exp"] = df["actual_score"] - df["expected_score"]
    return {
        "results_form": float(np.dot(df["actual_score"], weights) / total_weight),
        "gd_form": float(np.dot(df["gd_capped"], weights) / total_weight),
        "perf_vs_exp": float(np.dot(df["perf_vs_exp"], weights) / total_weight),
        "goals_for": float(np.dot(df["team_score"], weights) / total_weight),
        "goals_against": float(np.dot(df["opponent_score"], weights) / total_weight),
        "pre_tournament_elo": latest_pre_tournament_elo(df),
    }


def build_quadratic_form_feature_lookup(
    results_df: pd.DataFrame,
    team_key_column: str,
    match_window: int = RECENT_MATCH_WINDOW,
) -> dict[str, dict[str, float]]:
    """Build V4 quadratic weighted form snapshots keyed by team id/name."""
    if team_key_column not in results_df.columns:
        raise ValueError(f"results_df is missing {team_key_column}")
    lookup: dict[str, dict[str, float]] = {}
    grouped = results_df.sort_values(["date"], kind="stable").groupby(team_key_column, dropna=False)
    for team_key, matches in grouped:
        lookup[str(team_key)] = compute_quadratic_form_snapshot(matches, match_window=match_window)
    return lookup


def compute_wc_l5_goal_history(
    team_key: str,
    edition_year: int,
    placement_df: pd.DataFrame,
    edition_weight_map: dict[int, int],
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> dict[str, float]:
    """Return prior-World-Cup last-5 goal-difference signal for one team."""
    earlier_editions = select_prior_editions(
        edition_year,
        edition_weight_map,
        edition_lookback=edition_lookback,
    )
    if not earlier_editions:
        return {"wc_l5_goal_difference": np.nan, "wc_l5_goal_diff_norm": 0.5, "has_wc_l5_history": 0.0}

    team_rows = placement_df[
        (placement_df["team_key"] == team_key)
        & (placement_df["edition"].astype(int).isin([int(edition) for edition in earlier_editions]))
    ].copy()
    if team_rows.empty:
        return {"wc_l5_goal_difference": np.nan, "wc_l5_goal_diff_norm": 0.5, "has_wc_l5_history": 0.0}
    for column_name in ("gs", "ga", "matches_played"):
        if column_name not in team_rows.columns:
            team_rows[column_name] = 0.0
        team_rows[column_name] = pd.to_numeric(team_rows[column_name], errors="coerce").fillna(0.0)
    goal_difference = float((team_rows["gs"] - team_rows["ga"]).sum())
    matches_played = float(team_rows["matches_played"].sum())
    per_match = goal_difference / matches_played if matches_played > 0 else 0.0
    return {
        "wc_l5_goal_difference": goal_difference,
        "wc_l5_goal_diff_norm": float(np.clip((per_match + 3.0) / 6.0, 0.0, 1.0)),
        "has_wc_l5_history": 1.0,
    }


def v4_stage_key(stage: object) -> str:
    """Normalize a stage label or round code to a V4 stage multiplier key."""
    normalized = normalize_key(str(stage))
    if normalized in {"gs", "group", "group stage"} or "group" in normalized:
        return "group"
    if normalized in {"r32", "round of 32"}:
        return "round_of_16"
    if normalized in {"r16", "round of 16", "last 16"} or "16" in normalized:
        return "round_of_16"
    if normalized in {"qf", "quarter finals", "quarter final", "quarterfinals", "quarterfinal"} or "quarter" in normalized:
        return "quarter"
    if normalized in {"sf", "semi finals", "semi final", "semifinals", "semifinal"} or "semi" in normalized:
        return "semi"
    if normalized in {"f", "final"} or normalized.endswith("final"):
        return "final"
    return "group"


def compute_v4_stage_multipliers() -> dict[str, float]:
    """Compute World Cup stage goal multipliers with conservative fallbacks."""
    multipliers = dict(V4_STAGE_MULTIPLIER_FALLBACKS)
    historical_results = load_historical_world_cup_results(exclude_editions=())
    if historical_results.empty:
        return multipliers
    df = historical_results.copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "stage"]).copy()
    if df.empty:
        return multipliers
    df["stage_key"] = df["stage"].map(v4_stage_key)
    df["total_goals"] = df["home_score"] + df["away_score"]
    stage_avg = df.groupby("stage_key")["total_goals"].mean()
    group_avg = float(stage_avg.get("group", 0.0))
    if group_avg <= 0:
        return multipliers
    for key in multipliers:
        if key == "group":
            multipliers[key] = 1.0
        elif key in stage_avg and pd.notna(stage_avg[key]):
            multipliers[key] = float(np.clip(stage_avg[key] / group_avg, 0.65, 1.15))
    return multipliers


def dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Return the Dixon-Coles low-score correction factor."""
    h = int(home_goals)
    a = int(away_goals)
    rho = float(rho)
    if h == 0 and a == 0:
        return 1.0 - float(lambda_home) * float(lambda_away) * rho
    if h == 0 and a == 1:
        return 1.0 + float(lambda_home) * rho
    if h == 1 and a == 0:
        return 1.0 + float(lambda_away) * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def poisson_probability_vector_v4(lambda_value: float, goal_cap: int = V4_POISSON_GOAL_CAP) -> np.ndarray:
    """Return Poisson probabilities 0..goal_cap, folding the tail into the final bucket."""
    lambda_value = float(np.clip(lambda_value, V4_LAMBDA_MIN, V4_LAMBDA_MAX))
    probabilities = np.zeros(goal_cap + 1, dtype=float)
    probabilities[0] = float(np.exp(-lambda_value))
    running_total = probabilities[0]
    for goals in range(1, goal_cap):
        probabilities[goals] = probabilities[goals - 1] * lambda_value / float(goals)
        running_total += probabilities[goals]
    probabilities[goal_cap] = max(0.0, 1.0 - running_total)
    probabilities /= probabilities.sum()
    return probabilities


def build_v4_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    goal_cap: int = V4_POISSON_GOAL_CAP,
) -> np.ndarray:
    """Build a normalized Dixon-Coles corrected score probability matrix."""
    home_probabilities = poisson_probability_vector_v4(lambda_home, goal_cap=goal_cap)
    away_probabilities = poisson_probability_vector_v4(lambda_away, goal_cap=goal_cap)
    score_matrix = np.outer(home_probabilities, away_probabilities)
    for home_goals in range(min(2, goal_cap + 1)):
        for away_goals in range(min(2, goal_cap + 1)):
            tau = dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            score_matrix[home_goals, away_goals] *= max(float(tau), 0.0)
    matrix_sum = float(score_matrix.sum())
    if matrix_sum <= 0:
        return np.outer(home_probabilities, away_probabilities)
    return score_matrix / matrix_sum


def build_v4_probability_triplet(lambda_home: float, lambda_away: float, rho: float = 0.0) -> tuple[float, float, float]:
    """Convert V4 lambdas into home/draw/away probabilities."""
    score_matrix = build_v4_score_matrix(lambda_home, lambda_away, rho=rho)
    draw_prob = float(np.trace(score_matrix))
    home_win_prob = float(np.tril(score_matrix, k=-1).sum())
    away_win_prob = float(np.triu(score_matrix, k=1).sum())
    total = home_win_prob + draw_prob + away_win_prob
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return home_win_prob / total, draw_prob / total, away_win_prob / total


def sample_scores_from_matrix(score_matrix: np.ndarray, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample score pairs from a normalized score matrix."""
    flat = np.asarray(score_matrix, dtype=float).ravel()
    flat = flat / flat.sum()
    choices = rng.choice(len(flat), size=int(size), p=flat)
    home_scores, away_scores = np.unravel_index(choices, score_matrix.shape)
    return home_scores.astype(np.int16), away_scores.astype(np.int16)


def strength_weighted_penalty_probability(home_strength: float, away_strength: float) -> float:
    """Resolve shootouts with a weak strength tilt instead of a pure coin flip."""
    raw = 1.0 / (1.0 + np.exp(-((float(home_strength) - float(away_strength)) / 250.0)))
    return float(np.clip(raw, 0.35, 0.65))


def infer_v4_host_flag(
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
    if int(reference_edition_year) == 2026 and team_id_normalized in V4_2026_HOST_TEAM_IDS:
        return 1.0
    if int(reference_edition_year) == 2022 and (
        team_id_normalized in V4_2022_HOST_TEAM_IDS or team_name_key == "qatar"
    ):
        return 1.0
    return 0.0


def build_v4_strength_score(
    elo_rating: float,
    results_form: float,
    gd_form: float,
    perf_vs_exp: float,
    goals_for: float,
    goals_against: float,
    placement: float,
    appearance: float,
    host_flag: float,
    wc_l5_goal_diff_norm: float = 0.5,
) -> float:
    """Build a stable scalar fallback used for tie-breaks and display ordering."""
    return float(
        elo_rating
        + 200.0 * results_form
        + 180.0 * perf_vs_exp
        + 150.0 * placement
        + 100.0 * wc_l5_goal_diff_norm
        + 50.0 * gd_form
        + 30.0 * goals_for
        - 25.0 * goals_against
        + 20.0 * appearance
        + 15.0 * host_flag
    )


def build_v4_team_feature_table(
    base_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    reference_date_or_edition: int | str | pd.Timestamp,
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
) -> pd.DataFrame:
    """Build the current-team V4 feature table consumed by the Poisson matchup model."""
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
    form_lookup = build_quadratic_form_feature_lookup(lead_in_df, "qualified_team_id", match_window=match_window)
    placement_df, _, edition_weight_map = load_historical_placement_history()

    rows: list[dict[str, object]] = []
    for row in base_df.itertuples(index=False):
        team_id = str(getattr(row, "team_id"))
        display_name = str(getattr(row, "display_name", getattr(row, "tournament_name", team_id)))
        canonical_name = str(getattr(row, "canonical_name", display_name))
        form_snapshot = form_lookup.get(team_id, {})
        history_snapshot = history_lookup.get(team_id, {})
        team_key = normalize_historical_team_name(display_name or canonical_name)
        wc_l5_snapshot = compute_wc_l5_goal_history(
            team_key,
            reference_year,
            placement_df,
            edition_weight_map,
            edition_lookback=edition_lookback,
        )

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
        host_flag = infer_v4_host_flag(
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
        v4_strength = build_v4_strength_score(
            elo_rating=float(elo_rating) if pd.notna(elo_rating) else 0.0,
            results_form=results_form,
            gd_form=gd_form,
            perf_vs_exp=perf_vs_exp,
            goals_for=goals_for,
            goals_against=goals_against,
            placement=placement,
            appearance=appearance,
            host_flag=host_flag,
            wc_l5_goal_diff_norm=float(wc_l5_snapshot.get("wc_l5_goal_diff_norm", 0.5)),
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
                "team_strength": v4_strength,
                "v4_strength": v4_strength,
                "results_form": results_form,
                "gd_form": gd_form,
                "perf_vs_exp": perf_vs_exp,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "placement": placement,
                "appearance": appearance,
                "host_flag": host_flag,
                "wc_l5_goal_difference": float(wc_l5_snapshot.get("wc_l5_goal_difference", 0.0))
                if pd.notna(wc_l5_snapshot.get("wc_l5_goal_difference", np.nan))
                else 0.0,
                "wc_l5_goal_diff_norm": float(wc_l5_snapshot.get("wc_l5_goal_diff_norm", 0.5)),
                "has_wc_l5_history": float(wc_l5_snapshot.get("has_wc_l5_history", 0.0)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["team_strength", "elo_rating", "display_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_v4_training_frame(
    results_df: pd.DataFrame,
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
    start_year: int = V4_MATCH_START_YEAR,
    end_date: str | pd.Timestamp | None = None,
    exclude_tournament: str | Iterable[str] | None = None,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
    reference_edition_year: int = 2026,
) -> pd.DataFrame:
    """Build the historical match-level V4 training frame for the selected training scope."""
    scope = normalize_training_scope(training_scope)
    anchor_year = resolve_training_anchor_year(reference_edition_year, lookback_editions=edition_lookback)
    anchor_date = resolve_training_anchor_date(reference_edition_year, lookback_editions=edition_lookback)
    if results_df.empty and scope == TRAINING_SCOPE_ALL_INTERNATIONAL:
        raise ValueError("results_df must include historical international matches for V4")

    cutoff = pd.Timestamp(end_date) if end_date is not None else None
    excluded_tournaments = set(normalize_excluded_tournaments(exclude_tournament))

    if scope == TRAINING_SCOPE_WORLD_CUP_ONLY:
        training_source = load_historical_world_cup_results(exclude_editions=())
        training_source["tournament"] = "FIFA World Cup"
        training_source["neutral"] = True
        if "country" not in training_source.columns:
            training_source["country"] = ""
    else:
        training_source = results_df.copy()
    training_source["date"] = pd.to_datetime(training_source["date"], errors="coerce")
    training_source["home_score"] = pd.to_numeric(training_source["home_score"], errors="coerce")
    training_source["away_score"] = pd.to_numeric(training_source["away_score"], errors="coerce")
    training_source = training_source.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"]).copy()
    training_source = training_source[training_source["date"] >= anchor_date].copy()
    training_source = training_source[training_source["date"].dt.year >= int(start_year)].copy()
    if cutoff is not None:
        training_source = training_source[training_source["date"] <= cutoff].copy()
    if excluded_tournaments:
        training_source["_tournament_key"] = training_source["tournament"].map(normalize_key)
        training_source = training_source[~training_source["_tournament_key"].isin(excluded_tournaments)].copy()
    if training_source.empty:
        raise ValueError("V4 training frame is empty after date and tournament filtering")

    country_results_lookup = load_historical_country_results_lookup()
    placement_df, edition_team_counts, edition_weight_map = load_historical_placement_history()
    empty_form_snapshot = {
        "results_form": 0.0,
        "gd_form": 0.0,
        "perf_vs_exp": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "pre_tournament_elo": 0.0,
    }
    prepared_country_results: dict[str, dict[str, np.ndarray]] = {}
    for team_key, team_results in country_results_lookup.items():
        if team_results.empty:
            continue
        prepared = team_results.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        for column_name in ("team_score", "opponent_score", "team_elo_start", "opponent_elo_start", "team_elo_end"):
            if column_name in prepared.columns:
                prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")
        if "team_elo_end" not in prepared.columns:
            prepared["team_elo_end"] = np.nan
        prepared = prepared.dropna(subset=["date"]).sort_values(["date"], kind="stable").reset_index(drop=True)
        if prepared.empty:
            continue
        normalized_result = normalize_weighted_form_result(
            prepared["result"],
            prepared["team_score"],
            prepared["opponent_score"],
        )
        actual_score = normalized_result.map({"win": 1.0, "draw": 0.5, "loss": 0.0})
        goal_difference = prepared["team_score"] - prepared["opponent_score"]
        perf_vs_exp = actual_score - compute_elo_expected_score(
            prepared["team_elo_start"],
            prepared["opponent_elo_start"],
        ).astype(float)
        eligible_mask = (
            prepared["team_score"].notna()
            & prepared["opponent_score"].notna()
            & prepared["team_elo_start"].notna()
            & prepared["opponent_elo_start"].notna()
            & normalized_result.notna()
        )
        prepared_country_results[team_key] = {
            "date": prepared["date"].to_numpy(dtype="datetime64[ns]"),
            "eligible_index": np.flatnonzero(eligible_mask.to_numpy()),
            "elo_end_index": np.flatnonzero(prepared["team_elo_end"].notna().to_numpy()),
            "elo_start_index": np.flatnonzero(prepared["team_elo_start"].notna().to_numpy()),
            "actual_score": actual_score.fillna(0.0).to_numpy(dtype=float),
            "gd_capped": goal_difference.clip(
                lower=-WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
                upper=WEIGHTED_FORM_GOAL_DIFFERENCE_CAP,
            ).fillna(0.0).to_numpy(dtype=float),
            "perf_vs_exp": perf_vs_exp.fillna(0.0).to_numpy(dtype=float),
            "goals_for": prepared["team_score"].fillna(0.0).to_numpy(dtype=float),
            "goals_against": prepared["opponent_score"].fillna(0.0).to_numpy(dtype=float),
            "team_elo_end": prepared["team_elo_end"].fillna(0.0).to_numpy(dtype=float),
            "team_elo_start": prepared["team_elo_start"].fillna(0.0).to_numpy(dtype=float),
        }

    form_snapshot_cache: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    history_snapshot_cache: dict[tuple[str, int], dict[str, float]] = {}

    def form_snapshot_before(team_key: str, match_date: pd.Timestamp) -> dict[str, float]:
        cache_key = (team_key, match_date)
        if cache_key not in form_snapshot_cache:
            prepared = prepared_country_results.get(team_key)
            if prepared is None:
                form_snapshot_cache[cache_key] = empty_form_snapshot.copy()
                return form_snapshot_cache[cache_key]

            position = int(np.searchsorted(prepared["date"], np.datetime64(match_date), side="left"))
            eligible_indices = prepared["eligible_index"]
            recent_indices = eligible_indices[eligible_indices < position][-match_window:]

            pre_tournament_elo = 0.0
            elo_end_indices = prepared["elo_end_index"]
            prior_elo_end_indices = elo_end_indices[elo_end_indices < position]
            if len(prior_elo_end_indices):
                pre_tournament_elo = float(prepared["team_elo_end"][prior_elo_end_indices[-1]])
            else:
                elo_start_indices = prepared["elo_start_index"]
                prior_elo_start_indices = elo_start_indices[elo_start_indices < position]
                if len(prior_elo_start_indices):
                    pre_tournament_elo = float(prepared["team_elo_start"][prior_elo_start_indices[-1]])

            if len(recent_indices) == 0:
                snapshot = empty_form_snapshot.copy()
                snapshot["pre_tournament_elo"] = pre_tournament_elo
                form_snapshot_cache[cache_key] = snapshot
                return snapshot

            recency_weight = quadratic_recency_weights(len(recent_indices))
            total_weight = float(recency_weight.sum())
            form_snapshot_cache[cache_key] = {
                "results_form": float(np.dot(prepared["actual_score"][recent_indices], recency_weight) / total_weight),
                "gd_form": float(np.dot(prepared["gd_capped"][recent_indices], recency_weight) / total_weight),
                "perf_vs_exp": float(np.dot(prepared["perf_vs_exp"][recent_indices], recency_weight) / total_weight),
                "goals_for": float(np.dot(prepared["goals_for"][recent_indices], recency_weight) / total_weight),
                "goals_against": float(np.dot(prepared["goals_against"][recent_indices], recency_weight) / total_weight),
                "pre_tournament_elo": pre_tournament_elo,
            }
        return form_snapshot_cache[cache_key]

    def history_snapshot_before(team_key: str, edition_year: int) -> dict[str, float]:
        cache_key = (team_key, int(edition_year))
        if cache_key not in history_snapshot_cache:
            base_history = compute_pre_tournament_history_features(
                team_key,
                int(edition_year),
                placement_df,
                edition_team_counts,
                edition_weight_map,
                edition_lookback=edition_lookback,
            )
            wc_l5_history = compute_wc_l5_goal_history(
                team_key,
                int(edition_year),
                placement_df,
                edition_weight_map,
                edition_lookback=edition_lookback,
            )
            history_snapshot_cache[cache_key] = {**base_history, **wc_l5_history}
        return history_snapshot_cache[cache_key]

    rows: list[dict[str, object]] = []
    for match in training_source.sort_values(["date", "home_team", "away_team"], kind="stable").itertuples(index=False):
        match_date = pd.Timestamp(match.date)
        home_key = normalize_historical_team_name(str(match.home_team))
        away_key = normalize_historical_team_name(str(match.away_team))
        edition_year = int(match_date.year)

        home_form = form_snapshot_before(home_key, match_date)
        away_form = form_snapshot_before(away_key, match_date)
        home_history = history_snapshot_before(home_key, edition_year)
        away_history = history_snapshot_before(away_key, edition_year)
        home_wc_l5_goal_difference = home_history.get("wc_l5_goal_difference", np.nan)
        away_wc_l5_goal_difference = away_history.get("wc_l5_goal_difference", np.nan)

        neutral_site_flag = 1.0 if is_neutral_site(getattr(match, "neutral", False)) else 0.0
        match_country_key = normalize_historical_team_name(getattr(match, "country", ""))
        home_host_flag = 1.0 if not neutral_site_flag and match_country_key == home_key else 0.0
        away_host_flag = 1.0 if not neutral_site_flag and match_country_key == away_key else 0.0

        rows.append(
            {
                "date": match_date,
                "edition": int(getattr(match, "edition", match_date.year)) if pd.notna(getattr(match, "edition", match_date.year)) else int(match_date.year),
                "home_team": str(match.home_team),
                "away_team": str(match.away_team),
                "tournament": str(getattr(match, "tournament", "")),
                "stage": str(getattr(match, "stage", "")),
                "stage_bucket": match_stage_bucket(getattr(match, "stage", "")),
                "home_score": int(match.home_score),
                "away_score": int(match.away_score),
                "outcome_label": outcome_label_from_scoreline(int(match.home_score), int(match.away_score)),
                "elo_diff": float(home_form.get("pre_tournament_elo", 0.0)) - float(away_form.get("pre_tournament_elo", 0.0)),
                "results_form_diff": float(home_form.get("results_form", 0.0)) - float(away_form.get("results_form", 0.0)),
                "goals_for_diff": float(home_form.get("goals_for", 0.0)) - float(away_form.get("goals_for", 0.0)),
                "goals_against_diff": float(home_form.get("goals_against", 0.0)) - float(away_form.get("goals_against", 0.0)),
                "placement_diff": float(home_history.get("placement", 0.0)) - float(away_history.get("placement", 0.0)),
                "appearance_diff": float(home_history.get("appearance", 0.0)) - float(away_history.get("appearance", 0.0)),
                "home_wc_l5_goal_difference": home_wc_l5_goal_difference,
                "away_wc_l5_goal_difference": away_wc_l5_goal_difference,
                "home_has_wc_l5_history": float(home_history.get("has_wc_l5_history", 0.0)),
                "away_has_wc_l5_history": float(away_history.get("has_wc_l5_history", 0.0)),
                "gd_form_diff": float(home_form.get("gd_form", 0.0)) - float(away_form.get("gd_form", 0.0)),
                "perf_vs_exp_diff": float(home_form.get("perf_vs_exp", 0.0)) - float(away_form.get("perf_vs_exp", 0.0)),
                "competition_importance": classify_competition_importance(getattr(match, "tournament", "")),
                "neutral_site_flag": neutral_site_flag,
                "net_host_flag": home_host_flag - away_host_flag,
                "is_knockout": 0.0 if match_stage_bucket(getattr(match, "stage", "")) == V2_STAGE_GROUP else 1.0,
                "competition_weight": classify_competition_importance(getattr(match, "tournament", "")),
                "sample_weight": classify_competition_importance(getattr(match, "tournament", "")),
            }
        )

    training_df = pd.DataFrame(rows)
    if training_df.empty:
        raise ValueError("V4 training frame is empty")
    observed_wc_values = pd.concat(
        [
            pd.to_numeric(training_df["home_wc_l5_goal_difference"], errors="coerce"),
            pd.to_numeric(training_df["away_wc_l5_goal_difference"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    wc_l5_impute = float(observed_wc_values.mean()) if not observed_wc_values.empty else 0.0
    training_df["home_wc_l5_goal_difference"] = pd.to_numeric(training_df["home_wc_l5_goal_difference"], errors="coerce").fillna(wc_l5_impute)
    training_df["away_wc_l5_goal_difference"] = pd.to_numeric(training_df["away_wc_l5_goal_difference"], errors="coerce").fillna(wc_l5_impute)
    training_df["wc_l5_goal_diff_diff"] = training_df["home_wc_l5_goal_difference"] - training_df["away_wc_l5_goal_difference"]
    training_df["has_wc_l5_history_diff"] = (
        pd.to_numeric(training_df["home_has_wc_l5_history"], errors="coerce").fillna(0.0)
        - pd.to_numeric(training_df["away_has_wc_l5_history"], errors="coerce").fillna(0.0)
    )
    cutoff_for_weights = pd.Timestamp(end_date) if end_date is not None else pd.to_datetime(training_df["date"], errors="coerce").max()
    days_before_cutoff = (cutoff_for_weights - pd.to_datetime(training_df["date"], errors="coerce")).dt.days.clip(lower=0)
    half_life_days = float(V4_DEFAULT_TIME_DECAY_HALFLIFE_DAYS)
    training_df["time_weight"] = np.power(0.5, days_before_cutoff / half_life_days)
    training_df["sample_weight"] = pd.to_numeric(training_df["competition_weight"], errors="coerce").fillna(1.0) * training_df["time_weight"]
    training_df.attrs["wc_l5_goal_difference_impute"] = wc_l5_impute
    training_df.attrs["time_decay_halflife_days"] = half_life_days
    training_df["training_scope"] = scope
    training_df["anchor_year"] = int(anchor_year)
    training_df["anchor_date"] = anchor_date.strftime("%Y-%m-%d")
    return training_df


@lru_cache(maxsize=16)
def fit_v4_poisson_models(
    match_window: int = RECENT_MATCH_WINDOW,
    edition_lookback: int = V2_PREVIOUS_EDITION_LOOKBACK,
    start_year: int = V4_MATCH_START_YEAR,
    end_date: str | None = None,
    exclude_tournament: str | tuple[str, ...] | None = None,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
    reference_edition_year: int = 2026,
) -> dict[str, object]:
    """Fit and cache the pair of Poisson goal models used by V4."""
    results_path = INTERNATIONAL_RESULTS_PATH
    if not results_path.exists():
        raise ValueError("Historical international results are unavailable for V4 training")
    normalized_exclusions = normalize_excluded_tournaments(exclude_tournament)
    scope = normalize_training_scope(training_scope)
    training_df = build_v4_training_frame(
        pd.read_csv(results_path),
        match_window=match_window,
        edition_lookback=edition_lookback,
        start_year=start_year,
        end_date=end_date,
        exclude_tournament=normalized_exclusions,
        training_scope=scope,
        reference_edition_year=reference_edition_year,
    )
    try:
        from sklearn.linear_model import PoissonRegressor
        from sklearn.metrics import mean_poisson_deviance
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
        raise ImportError("scikit-learn is required for the V4 Poisson simulator") from exc

    training_df = training_df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    X = training_df.loc[:, list(V4_FEATURE_COLUMNS)].astype(float)
    sample_weight = training_df["sample_weight"].astype(float).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_home = training_df["home_score"].astype(float).to_numpy()
    y_away = training_df["away_score"].astype(float).to_numpy()
    alpha_scores: list[dict[str, float]] = []
    selected_alpha = 0.1
    alpha_source = "fallback"
    if len(training_df) >= 60:
        splits = min(5, max(2, len(training_df) // 40))
        try:
            tscv = TimeSeriesSplit(n_splits=splits)
            best_score = np.inf
            for alpha in V4_ALPHA_GRID:
                fold_scores = []
                for train_idx, test_idx in tscv.split(X_scaled):
                    home_model = PoissonRegressor(alpha=float(alpha), max_iter=1000)
                    away_model = PoissonRegressor(alpha=float(alpha), max_iter=1000)
                    home_model.fit(X_scaled[train_idx], y_home[train_idx], sample_weight=sample_weight[train_idx])
                    away_model.fit(X_scaled[train_idx], y_away[train_idx], sample_weight=sample_weight[train_idx])
                    home_pred = np.clip(home_model.predict(X_scaled[test_idx]), V4_LAMBDA_MIN, V4_LAMBDA_MAX)
                    away_pred = np.clip(away_model.predict(X_scaled[test_idx]), V4_LAMBDA_MIN, V4_LAMBDA_MAX)
                    score = 0.5 * (
                        mean_poisson_deviance(y_home[test_idx], home_pred)
                        + mean_poisson_deviance(y_away[test_idx], away_pred)
                    )
                    fold_scores.append(float(score))
                mean_score = float(np.mean(fold_scores))
                alpha_scores.append({"alpha": float(alpha), "mean_poisson_deviance": mean_score})
                if mean_score < best_score:
                    best_score = mean_score
                    selected_alpha = float(alpha)
                    alpha_source = "time_series_cv"
        except Exception:
            alpha_scores = []
            selected_alpha = 0.1
            alpha_source = "fallback"

    home_goal_model = PoissonRegressor(alpha=selected_alpha, max_iter=1000)
    away_goal_model = PoissonRegressor(alpha=selected_alpha, max_iter=1000)
    home_goal_model.fit(X_scaled, training_df["home_score"].astype(float).to_numpy(), sample_weight=sample_weight)
    away_goal_model.fit(X_scaled, training_df["away_score"].astype(float).to_numpy(), sample_weight=sample_weight)
    rho, rho_source = fit_v4_dixon_coles_rho(
        training_df,
        X_scaled,
        home_goal_model,
        away_goal_model,
        sample_weight=sample_weight,
    )
    stage_multipliers = compute_v4_stage_multipliers()

    anchor_year = int(training_df["anchor_year"].iloc[0])
    anchor_date = pd.Timestamp(training_df["anchor_date"].iloc[0])
    return {
        "training_frame": training_df,
        "feature_columns": V4_FEATURE_COLUMNS,
        "scaler": scaler,
        "home_goal_model": home_goal_model,
        "away_goal_model": away_goal_model,
        "alpha": selected_alpha,
        "alpha_source": alpha_source,
        "alpha_cv_scores": alpha_scores,
        "rho": rho,
        "rho_source": rho_source,
        "stage_multipliers": stage_multipliers,
        "wc_l5_goal_difference_impute": float(training_df.attrs.get("wc_l5_goal_difference_impute", 0.0)),
        "time_decay_halflife_days": int(training_df.attrs.get("time_decay_halflife_days", V4_DEFAULT_TIME_DECAY_HALFLIFE_DAYS)),
        "match_window": int(match_window),
        "edition_lookback": int(edition_lookback),
        "start_year": int(start_year),
        "end_date": end_date,
        "exclude_tournament": normalized_exclusions,
        **training_metadata_from_frame(training_df, scope, anchor_year, anchor_date),
    }


poisson_probability_vector = poisson_probability_vector_v4


def fit_v4_dixon_coles_rho(
    training_df: pd.DataFrame,
    X_scaled: np.ndarray,
    home_goal_model: object,
    away_goal_model: object,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, str]:
    """Fit rho by a conservative grid search over weighted score log likelihood."""
    try:
        lambda_home = np.clip(home_goal_model.predict(X_scaled), V4_LAMBDA_MIN, V4_LAMBDA_MAX)
        lambda_away = np.clip(away_goal_model.predict(X_scaled), V4_LAMBDA_MIN, V4_LAMBDA_MAX)
        home_scores = pd.to_numeric(training_df["home_score"], errors="coerce").fillna(0).clip(0, V4_POISSON_GOAL_CAP).astype(int).to_numpy()
        away_scores = pd.to_numeric(training_df["away_score"], errors="coerce").fillna(0).clip(0, V4_POISSON_GOAL_CAP).astype(int).to_numpy()
        weights = np.ones(len(training_df), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        best_rho = 0.0
        best_log_likelihood = -np.inf
        for rho in np.linspace(V4_RHO_BOUNDS[0], V4_RHO_BOUNDS[1], 41):
            log_likelihood = 0.0
            valid = True
            for idx, (home_score, away_score) in enumerate(zip(home_scores, away_scores, strict=False)):
                matrix = build_v4_score_matrix(float(lambda_home[idx]), float(lambda_away[idx]), rho=float(rho))
                probability = float(matrix[int(home_score), int(away_score)])
                if probability <= 0 or not np.isfinite(probability):
                    valid = False
                    break
                log_likelihood += float(weights[idx]) * float(np.log(probability))
            if valid and log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_rho = float(rho)
        return best_rho, "grid_search"
    except Exception:
        return 0.0, "fallback"


def predict_match_lambdas_v4(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    neutral_site: bool = True,
    stage: str = "group",
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
                "wc_l5_goal_diff_diff": float(home_features.get("wc_l5_goal_difference", 0.0)) - float(away_features.get("wc_l5_goal_difference", 0.0)),
                "has_wc_l5_history_diff": float(home_features.get("has_wc_l5_history", 0.0)) - float(away_features.get("has_wc_l5_history", 0.0)),
                "gd_form_diff": float(home_features["gd_form"]) - float(away_features["gd_form"]),
                "perf_vs_exp_diff": float(home_features["perf_vs_exp"]) - float(away_features["perf_vs_exp"]),
                "competition_importance": float(V4_COMPETITION_IMPORTANCE["world_cup_finals"]),
                "neutral_site_flag": 1.0 if neutral_site else 0.0,
                "net_host_flag": float(home_features.get("host_flag", 0.0)) - float(away_features.get("host_flag", 0.0)),
                "is_knockout": 0.0 if v4_stage_key(stage) == "group" else 1.0,
            }
        ],
        columns=list(model_bundle["feature_columns"]),
    )
    scaled = model_bundle["scaler"].transform(feature_row)
    lambda_home = float(model_bundle["home_goal_model"].predict(scaled)[0])
    lambda_away = float(model_bundle["away_goal_model"].predict(scaled)[0])
    stage_key = v4_stage_key(stage)
    stage_multiplier = float(model_bundle.get("stage_multipliers", V4_STAGE_MULTIPLIER_FALLBACKS).get(stage_key, 1.0))
    lambda_home_adj = float(np.clip(lambda_home * stage_multiplier, V4_LAMBDA_MIN, V4_LAMBDA_MAX))
    lambda_away_adj = float(np.clip(lambda_away * stage_multiplier, V4_LAMBDA_MIN, V4_LAMBDA_MAX))
    rho = float(model_bundle.get("rho", 0.0))
    home_win_prob, draw_prob, away_win_prob = build_v4_probability_triplet(lambda_home_adj, lambda_away_adj, rho=rho)
    return {
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "lambda_home": float(np.clip(lambda_home, V4_LAMBDA_MIN, V4_LAMBDA_MAX)),
        "lambda_away": float(np.clip(lambda_away, V4_LAMBDA_MIN, V4_LAMBDA_MAX)),
        "lambda_home_adj": lambda_home_adj,
        "lambda_away_adj": lambda_away_adj,
        "rho": rho,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
    }


def simulate_knockout_match_v4(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    rng: np.random.Generator,
    matchup_probability_cache: dict[tuple[str, str, str], dict[str, float | str]] | None = None,
    stage: str = "knockout",
) -> tuple[str, str]:
    """Simulate one knockout matchup from corrected V4 score matrices."""
    cache_key = (str(home_team_id), str(away_team_id), v4_stage_key(stage))
    if matchup_probability_cache is not None and cache_key in matchup_probability_cache:
        probability_map = matchup_probability_cache[cache_key]
    else:
        neutral_site = not (
            float(team_feature_lookup[str(home_team_id)].get("host_flag", 0.0))
            or float(team_feature_lookup[str(away_team_id)].get("host_flag", 0.0))
        )
        probability_map = predict_match_lambdas_v4(
            home_team_id,
            away_team_id,
            team_feature_lookup,
            model_bundle,
            neutral_site=neutral_site,
            stage=stage,
        )
        if matchup_probability_cache is not None:
            matchup_probability_cache[cache_key] = probability_map

    rho = float(probability_map.get("rho", 0.0))
    regulation_matrix = build_v4_score_matrix(
        float(probability_map["lambda_home_adj"]),
        float(probability_map["lambda_away_adj"]),
        rho=rho,
    )
    regulation_home_array, regulation_away_array = sample_scores_from_matrix(regulation_matrix, rng, 1)
    regulation_home = int(regulation_home_array[0])
    regulation_away = int(regulation_away_array[0])
    if regulation_home > regulation_away:
        return str(home_team_id), str(away_team_id)
    if regulation_away > regulation_home:
        return str(away_team_id), str(home_team_id)

    extra_matrix = build_v4_score_matrix(
        max(float(probability_map["lambda_home_adj"]) * EXTRA_TIME_FACTOR, V4_LAMBDA_MIN),
        max(float(probability_map["lambda_away_adj"]) * EXTRA_TIME_FACTOR, V4_LAMBDA_MIN),
        rho=rho,
    )
    extra_home_array, extra_away_array = sample_scores_from_matrix(extra_matrix, rng, 1)
    extra_home = int(extra_home_array[0])
    extra_away = int(extra_away_array[0])
    if extra_home > extra_away:
        return str(home_team_id), str(away_team_id)
    if extra_away > extra_home:
        return str(away_team_id), str(home_team_id)

    penalty_home_prob = strength_weighted_penalty_probability(
        float(team_feature_lookup[str(home_team_id)].get("team_strength", 0.0)),
        float(team_feature_lookup[str(away_team_id)].get("team_strength", 0.0)),
    )
    if float(rng.random()) < penalty_home_prob:
        return str(home_team_id), str(away_team_id)
    return str(away_team_id), str(home_team_id)


def predict_knockout_matchup_v4(
    home_team_id: str,
    away_team_id: str,
    team_feature_lookup: dict[str, dict[str, float]],
    model_bundle: dict[str, object],
    simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, float | str]:
    """Estimate one knockout matchup with repeated V4 Poisson simulations."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    rng = np.random.default_rng(seed)
    matchup_probability_cache: dict[tuple[str, str, str], dict[str, float | str]] = {}
    home_wins = 0
    for _ in range(simulations):
        winner_team_id, _ = simulate_knockout_match_v4(
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


def build_deterministic_bracket_v4(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable V4 knockout bracket from modal group rankings and Poisson matchup odds."""
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
        prediction = predict_knockout_matchup_v4(
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


def build_deterministic_bracket_v4_32team(
    simulation_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
    head_to_head_simulations: int = 1000,
    seed: int = 20260403,
) -> dict[str, object]:
    """Build one stable 32-team V4 knockout bracket from modal group rankings."""
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
        prediction = predict_knockout_matchup_v4(
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


def simulate_group_probabilities_v4(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = DEFAULT_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
) -> pd.DataFrame:
    """Simulate the 2026 tournament using the V4 Poisson expected-goals model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    scope = normalize_training_scope(training_scope)
    model_bundle = fit_v4_poisson_models(
        match_window=match_window,
        training_scope=scope,
        reference_edition_year=2026,
    )
    feature_df = build_v4_team_feature_table(base_df, lead_in_df, reference_date_or_edition=2026, match_window=match_window)
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
            probability_map = predict_match_lambdas_v4(
                str(match.home_team_id),
                str(match.away_team_id),
                team_feature_lookup,
                model_bundle,
                neutral_site=neutral_site,
                stage="group",
            )
            score_matrix = build_v4_score_matrix(
                float(probability_map["lambda_home_adj"]),
                float(probability_map["lambda_away_adj"]),
                rho=float(probability_map.get("rho", 0.0)),
            )
            home_scores, away_scores = sample_scores_from_matrix(score_matrix, rng, simulations)
            home_scores = home_scores.astype(np.int16)
            away_scores = away_scores.astype(np.int16)
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
    knockout_probability_cache: dict[tuple[str, str, str], dict[str, float | str]] = {}
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
                winner_team_id, loser_team_id = simulate_knockout_match_v4(
                    home_team_id,
                    away_team_id,
                    team_feature_lookup,
                    model_bundle,
                    knockout_rng,
                    matchup_probability_cache=knockout_probability_cache,
                    stage=str(match.round_code),
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


def simulate_group_probabilities_v4_32team(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int,
    seed: int = 20260403,
    group_order: Iterable[str] = BACKTEST_2022_GROUP_ORDER,
    match_window: int = RECENT_MATCH_WINDOW,
    training_end_date: str | pd.Timestamp | None = None,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
) -> pd.DataFrame:
    """Simulate a 32-team tournament using the V4 Poisson expected-goals model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    group_order = list(group_order)
    scope = normalize_training_scope(training_scope)
    model_bundle = fit_v4_poisson_models(
        match_window=match_window,
        end_date=None if training_end_date is None else str(pd.Timestamp(training_end_date).date()),
        training_scope=scope,
        reference_edition_year=2022,
    )
    feature_df = build_v4_team_feature_table(base_df, lead_in_df, reference_date_or_edition=2022, match_window=match_window)
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
            probability_map = predict_match_lambdas_v4(
                str(match.home_team_id),
                str(match.away_team_id),
                team_feature_lookup,
                model_bundle,
                neutral_site=neutral_site,
                stage="group",
            )
            score_matrix = build_v4_score_matrix(
                float(probability_map["lambda_home_adj"]),
                float(probability_map["lambda_away_adj"]),
                rho=float(probability_map.get("rho", 0.0)),
            )
            home_scores, away_scores = sample_scores_from_matrix(score_matrix, rng, simulations)
            home_scores = home_scores.astype(np.int16)
            away_scores = away_scores.astype(np.int16)
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
    knockout_probability_cache: dict[tuple[str, str, str], dict[str, float | str]] = {}
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
            winner_team_id, loser_team_id = simulate_knockout_match_v4(
                home_team_id,
                away_team_id,
                team_feature_lookup,
                model_bundle,
                knockout_rng,
                matchup_probability_cache=knockout_probability_cache,
                stage=str(match.round_code),
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


def run_v4_2022_backtest(
    match_window: int = RECENT_MATCH_WINDOW,
    simulations: int = 20000,
    seed: int = 20260403,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
) -> dict[str, object]:
    """Run a leakage-free V4 backtest against the actual 2022 World Cup."""
    dataset = build_2022_backtest_data()
    base_df = dataset["base_df"]
    lead_in_df = dataset["lead_in_df"]
    fixtures_df = dataset["fixtures_df"]
    results_df = dataset["results_df"]
    placement_df = dataset["placement_df"]
    group_code_lookup = dataset["group_code_lookup"]
    edition_start = pd.to_datetime(pd.DataFrame(results_df)["date"], errors="coerce").min()
    training_end_date = None if pd.isna(edition_start) else str((pd.Timestamp(edition_start) - pd.Timedelta(days=1)).date())

    scope = normalize_training_scope(training_scope)
    model_bundle = fit_v4_poisson_models(
        match_window=match_window,
        end_date=training_end_date,
        training_scope=scope,
        reference_edition_year=2022,
    )
    feature_df = build_v4_team_feature_table(
        base_df,
        lead_in_df,
        reference_date_or_edition=2022,
        match_window=match_window,
    )
    simulation_df = simulate_group_probabilities_v4_32team(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        seed=seed,
        match_window=match_window,
        training_end_date=training_end_date,
        training_scope=scope,
    )
    deterministic_bracket = build_deterministic_bracket_v4_32team(
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
        probability_map = predict_match_lambdas_v4(
            home_team_id,
            away_team_id,
            feature_lookup,
            model_bundle,
            neutral_site=neutral_site,
            stage=str(row.stage),
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
                "lambda_home_adj": float(probability_map["lambda_home_adj"]),
                "lambda_away_adj": float(probability_map["lambda_away_adj"]),
                "rho": float(probability_map["rho"]),
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
        "training_metadata": {
            key: model_bundle.get(key)
            for key in [
                "training_scope",
                "anchor_year",
                "anchor_date",
                "training_start_date",
                "training_end_date",
                "training_match_count",
                "sample_weight_policy",
                "time_decay_halflife_days",
                "alpha",
                "alpha_source",
                "alpha_cv_scores",
                "rho",
                "rho_source",
                "stage_multipliers",
                "wc_l5_goal_difference_impute",
            ]
        },
    }


def run_v4_rolling_backtest(
    match_window: int = RECENT_MATCH_WINDOW,
    simulations: int = 20000,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
    holdout_years: Iterable[int] = (2014, 2018, 2022),
) -> dict[str, object]:
    """Run the V4 rolling holdout summary.

    The current project has a complete leakage-free tournament reconstruction for
    2022. Earlier folds are represented explicitly so the dashboard can expose
    rolling validation status without pretending unavailable fixture builders
    have been completed.
    """
    fold_rows: list[dict[str, object]] = []
    fold_results: dict[int, dict[str, object]] = {}
    metric_names = [
        "multiclass_log_loss",
        "multiclass_brier_score",
        "top1_match_accuracy",
        "draw_rate_actual",
        "draw_rate_predicted",
        "round_of_16_hit_count",
        "semifinal_hit_count",
        "exact_champion_hit",
    ]
    for holdout_year in holdout_years:
        if int(holdout_year) != 2022:
            fold_rows.append(
                {
                    "holdout_year": int(holdout_year),
                    "status": "not_available",
                    "reason": "A generic leakage-free historical tournament fixture builder is not implemented yet.",
                    **{metric_name: np.nan for metric_name in metric_names},
                }
            )
            continue
        result = run_v4_2022_backtest(
            match_window=match_window,
            simulations=simulations,
            training_scope=training_scope,
        )
        metrics = dict(result["summary_metrics"])
        fold_rows.append(
            {
                "holdout_year": 2022,
                "status": "ok",
                "reason": "",
                **{metric_name: metrics.get(metric_name, np.nan) for metric_name in metric_names},
            }
        )
        fold_results[2022] = result

    folds_df = pd.DataFrame(fold_rows)
    ok_folds = folds_df.loc[folds_df["status"].eq("ok")]
    aggregate_rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        values = pd.to_numeric(ok_folds.get(metric_name, pd.Series(dtype=float)), errors="coerce").dropna()
        aggregate_rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()) if not values.empty else np.nan,
                "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0 if len(values) == 1 else np.nan,
                "fold_count": int(len(values)),
            }
        )
    return {
        "folds": folds_df,
        "aggregate_metrics": pd.DataFrame(aggregate_rows),
        "fold_results": fold_results,
    }


__all__ = [
    'quadratic_recency_weights',
    'compute_quadratic_form_snapshot',
    'dixon_coles_tau',
    'poisson_probability_vector_v4',
    'build_v4_score_matrix',
    'normalize_excluded_tournaments',
    'classify_competition_importance',
    'is_neutral_site',
    'infer_v4_host_flag',
    'build_v4_strength_score',
    'build_v4_team_feature_table',
    'build_v4_training_frame',
    'fit_v4_poisson_models',
    'poisson_probability_vector',
    'build_v4_probability_triplet',
    'predict_match_lambdas_v4',
    'simulate_knockout_match_v4',
    'predict_knockout_matchup_v4',
    'build_deterministic_bracket_v4',
    'build_deterministic_bracket_v4_32team',
    'simulate_group_probabilities_v4',
    'simulate_group_probabilities_v4_32team',
    'run_v4_2022_backtest',
    'run_v4_rolling_backtest',
    'strength_weighted_penalty_probability',
]
