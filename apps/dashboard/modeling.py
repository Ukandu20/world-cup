from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from .config import (
    DEFAULT_RECENT_MATCH_WINDOW,
    DEFAULT_SIMULATION_LABEL,
    DEFAULT_V2_TRAINING_SCOPE,
    DEFAULT_V3_TRAINING_SCOPE,
    DEFAULT_V4_TRAINING_SCOPE,
    GROUP_ORDER,
    SIMULATION_COUNT,
    WEIGHTED_FORM_COMPOSITE_WEIGHTS,
    fit_v2_match_multinomial_model,
    fit_v3_poisson_models,
    fit_v4_poisson_models,
    run_v2_backtest_2022,
    run_v3_2022_backtest,
    run_v4_2022_backtest,
    run_v4_rolling_backtest,
    simulate_group_probabilities,
    simulate_group_probabilities_v2,
    simulate_group_probabilities_v3,
    simulate_group_probabilities_v4,
)

@st.cache_data(show_spinner=False)
def simulate_probabilities(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    baseline_rating_weights: tuple[float, float] = (1.0, 0.0),
    form_component_weights: tuple[float, float] = (0.7, 0.3),
    strength_blend_weights: tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """Estimate group finishing probabilities from the fixture-based Monte Carlo model."""
    simulator_kwargs = {
        "base_df": base_df,
        "fixtures_df": fixtures_df,
        "lead_in_df": lead_in_df,
        "simulations": simulations,
        "group_order": GROUP_ORDER,
    }
    try:
        simulator_signature = inspect.signature(simulate_group_probabilities)
    except (TypeError, ValueError):
        simulator_signature = None
    if simulator_signature is None:
        simulator_kwargs["match_window"] = match_window
        return simulate_group_probabilities(**simulator_kwargs)

    optional_kwargs = {
        "match_window": match_window,
        "baseline_rating_weights": baseline_rating_weights,
        "form_component_weights": form_component_weights,
        "strength_blend_weights": strength_blend_weights,
    }
    for key, value in optional_kwargs.items():
        if key in simulator_signature.parameters:
            simulator_kwargs[key] = value
    return simulate_group_probabilities(**simulator_kwargs)


@st.cache_resource(show_spinner=False)
def load_v2_match_model(
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V2_TRAINING_SCOPE,
) -> dict[str, object]:
    """Fit and cache the v2 multinomial model artifacts for the active form window."""
    return fit_v2_match_multinomial_model(match_window=form_match_window, training_scope=training_scope)


@st.cache_resource(show_spinner=False)
def load_v3_poisson_model(
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V3_TRAINING_SCOPE,
) -> dict[str, object]:
    """Fit and cache the v3 Poisson model artifacts for the active form window."""
    return fit_v3_poisson_models(
        match_window=form_match_window,
        training_scope=training_scope,
        reference_edition_year=2026,
    )


@st.cache_resource(show_spinner=False)
def load_v4_poisson_model(
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
) -> dict[str, object]:
    """Fit and cache the v4 enhanced Poisson artifacts for the active form window."""
    return fit_v4_poisson_models(
        match_window=form_match_window,
        training_scope=training_scope,
        reference_edition_year=2026,
    )


@st.cache_data(show_spinner=False)
def simulate_probabilities_v2_dashboard(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V2_TRAINING_SCOPE,
) -> pd.DataFrame:
    """Estimate tournament probabilities from the v2 multinomial simulator."""
    return simulate_group_probabilities_v2(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
        training_scope=training_scope,
    )


@st.cache_data(show_spinner=False)
def simulate_probabilities_v3_dashboard(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V3_TRAINING_SCOPE,
    seed: int = 20260403,
) -> pd.DataFrame:
    """Estimate tournament probabilities from the v3 Poisson simulator."""
    return simulate_group_probabilities_v3(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
        training_scope=training_scope,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def simulate_probabilities_v4_dashboard(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
    seed: int = 20260403,
) -> pd.DataFrame:
    """Estimate tournament probabilities from the v4 enhanced Poisson simulator."""
    return simulate_group_probabilities_v4(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
        training_scope=training_scope,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def run_v2_backtest_2022_dashboard(
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V2_TRAINING_SCOPE,
) -> dict[str, object]:
    """Run and cache the 2022 holdout backtest for the active UI settings."""
    return run_v2_backtest_2022(
        match_window=match_window,
        simulations=simulations,
        training_scope=training_scope,
    )


@st.cache_data(show_spinner=False)
def run_v3_backtest_2022_dashboard(
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V3_TRAINING_SCOPE,
) -> dict[str, object]:
    """Run and cache the 2022 holdout backtest for the active V3 UI settings."""
    return run_v3_2022_backtest(
        match_window=match_window,
        simulations=simulations,
        training_scope=training_scope,
    )


@st.cache_data(show_spinner=False)
def run_v4_backtest_2022_dashboard(
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
) -> dict[str, object]:
    """Run and cache the 2022 holdout backtest for the active V4 UI settings."""
    return run_v4_2022_backtest(
        match_window=match_window,
        simulations=simulations,
        training_scope=training_scope,
    )


@st.cache_data(show_spinner=False)
def run_v4_rolling_backtest_dashboard(
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
) -> dict[str, object]:
    """Run and cache the V4 rolling World Cup holdout backtest."""
    return run_v4_rolling_backtest(
        match_window=match_window,
        simulations=simulations,
        training_scope=training_scope,
    )


def ensure_dashboard_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee probability columns expected by the dashboard exist on the dataframe."""
    normalized = df.copy()
    for column_name in ("prob_1", "prob_2", "prob_3", "prob_4"):
        if column_name not in normalized.columns:
            normalized[column_name] = 0.0
    for column_name in ("top8_third_prob", "r32_prob", "r16_prob", "qf_prob", "sf_prob", "final_prob", "champion_prob"):
        if column_name not in normalized.columns:
            normalized[column_name] = 0.0
    if "ko_prob" not in normalized.columns:
        normalized["ko_prob"] = (
            normalized["prob_1"].fillna(0.0)
            + normalized["prob_2"].fillna(0.0)
            + normalized["top8_third_prob"].fillna(0.0)
        )
    if "r32_prob" in normalized.columns:
        normalized["r32_prob"] = normalized["r32_prob"].where(
            normalized["r32_prob"].notna() & normalized["r32_prob"].ne(0.0),
            normalized["ko_prob"],
        )
    return normalized


def default_simulation_settings() -> dict[str, str | int]:
    """Return the default simulation settings for the dashboard."""
    return {
        "simulation_label": DEFAULT_SIMULATION_LABEL,
        "form_match_window": DEFAULT_RECENT_MATCH_WINDOW,
        "v2_results_weight": int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[0] * 100)),
        "v2_gd_weight": int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[1] * 100)),
        "v2_perf_weight": int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[2] * 100)),
        "v2_elo_delta_weight": int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[3] * 100)),
    }

