from __future__ import annotations

from typing import Any

import pandas as pd

from .config import (
    BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    DEFAULT_RECENT_MATCH_WINDOW,
    DEFAULT_V3_TRAINING_SCOPE,
    DEFAULT_V4_TRAINING_SCOPE,
    V3_MODEL_VERSION,
    V4_MODEL_VERSION,
    build_deterministic_bracket_v3,
    build_deterministic_bracket_v4,
)
from .modeling import (
    load_v3_poisson_model,
    load_v4_poisson_model,
    simulate_probabilities_v3_dashboard,
    simulate_probabilities_v4_dashboard,
)
from .simulation_store import DEFAULT_SIMULATION_SEED


def _v4_calibration_metadata(model_bundle: dict[str, Any]) -> dict[str, Any]:
    """Return V4 display metadata that must be available from cached artifacts."""
    return {
        "alpha": model_bundle.get("alpha"),
        "rho": model_bundle.get("rho"),
        "time_decay_halflife_days": model_bundle.get("time_decay_halflife_days"),
        "stage_multipliers": model_bundle.get("stage_multipliers"),
        "alpha_source": model_bundle.get("alpha_source"),
        "rho_source": model_bundle.get("rho_source"),
    }


def build_v3_probability_artifact(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    *,
    simulations: int,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V3_TRAINING_SCOPE,
    seed: int = DEFAULT_SIMULATION_SEED,
    bracket_head_to_head_simulations: int = BRACKET_HEAD_TO_HEAD_SIMULATIONS,
) -> dict[str, Any]:
    """Run the V3 probability simulation and return a persistable artifact payload."""
    model_bundle = load_v3_poisson_model(match_window, training_scope)
    dashboard_df = simulate_probabilities_v3_dashboard(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
        training_scope=training_scope,
        seed=seed,
    )
    bracket_data = build_deterministic_bracket_v3(
        dashboard_df,
        fixtures_df,
        dashboard_df,
        model_bundle,
        head_to_head_simulations=bracket_head_to_head_simulations,
        seed=seed,
    )
    return {
        "dashboard_df": dashboard_df,
        "bracket_data": bracket_data,
        "metadata": {
            "model_id": "v3",
            "model_version": V3_MODEL_VERSION,
        },
    }


def build_v4_probability_artifact(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    *,
    simulations: int,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    training_scope: str = DEFAULT_V4_TRAINING_SCOPE,
    seed: int = DEFAULT_SIMULATION_SEED,
    bracket_head_to_head_simulations: int = BRACKET_HEAD_TO_HEAD_SIMULATIONS,
) -> dict[str, Any]:
    """Run the V4 probability simulation and return a persistable artifact payload."""
    model_bundle = load_v4_poisson_model(match_window, training_scope)
    dashboard_df = simulate_probabilities_v4_dashboard(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
        training_scope=training_scope,
        seed=seed,
    )
    bracket_data = build_deterministic_bracket_v4(
        dashboard_df,
        fixtures_df,
        dashboard_df,
        model_bundle,
        head_to_head_simulations=bracket_head_to_head_simulations,
        seed=seed,
    )
    return {
        "dashboard_df": dashboard_df,
        "bracket_data": bracket_data,
        "metadata": {
            "model_id": "v4",
            "model_version": V4_MODEL_VERSION,
            **_v4_calibration_metadata(model_bundle),
        },
    }
