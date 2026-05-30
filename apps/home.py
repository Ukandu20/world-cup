from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.dashboard.config import *  # noqa: F401,F403
from apps.dashboard.data import *  # noqa: F401,F403
from apps.dashboard.modeling import *  # noqa: F401,F403
from apps.dashboard.rendering import *  # noqa: F401,F403
from apps.dashboard.export import (
    bracket_to_download_frame,
    combine_table_download_frames,
    dataframe_to_csv_bytes,
    table_to_download_frame,
    tables_to_download_frame,
)
from apps.dashboard.pages import *  # noqa: F401,F403


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
    """Facade wrapper for the V1 simulator so tests can monkeypatch the simulator symbol."""
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


def build_navigation_pages() -> dict[str, list[st.Page]]:
    """Build grouped Streamlit pages for the dashboard navigation."""
    return {
        "Home": [
            st.Page(render_home_page, title="Dashboard", icon=":material/home:", default=True),
        ],
        "Reports": [
            st.Page(render_analysis_page, title="Analysis", icon=":material/analytics:"),
            st.Page(
                render_team_report_card_navigation_page,
                title="Team Report Card",
                icon=":material/assignment:",
            ),
        ],
        "Models": [
            st.Page(
                render_v4_probabilities_navigation_page,
                title="V4 Enhanced Poisson",
                icon=":material/looks_4:",
            ),
        ],
        "Legacy": [
            st.Page(render_v1_navigation_page, title="V1 Team Strength", icon=":material/looks_one:"),
            st.Page(render_v2_form_navigation_page, title="V2 Form", icon=":material/looks_two:"),
            st.Page(
                render_v2_probabilities_navigation_page,
                title="V2 Probabilities",
                icon=":material/functions:",
            ),
            st.Page(
                render_v3_probabilities_navigation_page,
                title="V3 Poisson Regression",
                icon=":material/looks_3:",
            ),
        ],
        "Backtests": [
            st.Page(render_v2_backtest_navigation_page, title="V2 2022 Backtest", icon=":material/history:"),
            st.Page(render_v3_backtest_navigation_page, title="V3 2022 Backtest", icon=":material/history:"),
            st.Page(render_v4_backtest_navigation_page, title="V4 2022 Backtest", icon=":material/history:"),
            st.Page(render_v4_rolling_backtest_navigation_page, title="V4 Rolling Backtest", icon=":material/query_stats:"),
        ],
    }


def main() -> None:
    """Run the grouped Streamlit navigation entrypoint."""
    configure_page("World Cup 2026 Prediction Dashboard")
    selected_page = st.navigation(build_navigation_pages(), position="top")
    selected_page.run()


if __name__ == "__main__":
    main()

