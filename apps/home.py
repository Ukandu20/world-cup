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
    build_export_stem,
    build_screenshot_command,
    estimate_export_column_count,
    estimate_export_viewport_size,
    export_bracket_png,
    export_document_png,
    generate_export_suffix,
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


def export_current_view(
    view_mode: str,
    selected_group: str,
    tables: list[dict[str, object]],
    bracket_data: dict[str, object] | None = None,
    metadata_lookup: dict[str, dict[str, str]] | None = None,
    simulation_count: int | None = None,
) -> Path:
    """Export the currently visible dashboard view as one PNG file."""
    export_suffix = generate_export_suffix()
    if view_mode == "Single group":
        return export_document_png(
            f"group_{selected_group.lower()}_view",
            f"Group {selected_group} View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    if view_mode == "All groups":
        return export_document_png(
            "all_groups_view",
            "All Groups View",
            tables,
            multi_column=True,
            export_suffix=export_suffix,
        )
    if view_mode == "Bracket":
        if bracket_data is None or metadata_lookup is None:
            raise ValueError("Bracket export requires bracket_data and metadata_lookup")
        return export_bracket_png(
            "bracket_view",
            "Bracket View",
            bracket_data,
            metadata_lookup,
            simulation_count=simulation_count,
            export_suffix=export_suffix,
        )
    if view_mode == "Form":
        return export_document_png(
            "form_view",
            "Form View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    return export_document_png(
        "all_Countries_view",
        "All Countries View",
        tables,
        multi_column=False,
        export_suffix=export_suffix,
    )


def export_all_tables(
    probability_df: pd.DataFrame | None = None,
    form_df: pd.DataFrame | None = None,
    simulation_count: int | None = None,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> list[Path]:
    """Export the probability tables and optionally the form table as PNG files."""
    exported_paths: list[Path] = []
    export_suffix = generate_export_suffix()
    if probability_df is not None:
        for group_code in GROUP_ORDER:
            group_df = projected_group_table_frame(probability_df, group_code)
            if group_df.empty:
                continue
            exported_paths.append(
                export_document_png(
                    f"group_{group_code.lower()}",
                    f"Group {group_code}",
                    [
                        {
                            "title": f"Group {group_code}",
                            "frame": group_df,
                            "include_group_column": False,
                            "include_ko_column": False,
                            "card_subtitle": chart_subtitle("Bracket-Aligned Projected Order", simulation_count),
                            "group_pill_label": group_code,
                            "table_kind": "probability",
                        }
                    ],
                    multi_column=False,
                    export_suffix=export_suffix,
                )
            )

        combined = all_teams_table_frame(probability_df)
        exported_paths.append(
            export_document_png(
                "all_Countries",
                "All Countries",
                [
                    {
                        "title": "All Countries",
                        "frame": combined,
                        "include_group_column": True,
                        "include_ko_column": True,
                        "card_subtitle": chart_subtitle("Pre-Tournament Probability Table", simulation_count),
                        "group_pill_label": None,
                        "table_kind": "probability",
                    }
                ],
                multi_column=False,
                export_suffix=export_suffix,
            )
        )
    if form_df is not None:
        all_countries_tables = current_form_view_tables(
            form_df,
            "All Countries",
            "",
            form_match_window=form_match_window,
        )
        all_confederations_tables = current_form_view_tables(
            form_df,
            "All confederations",
            "",
            form_match_window=form_match_window,
        )
        exported_paths.append(
            export_document_png(
                "form_all_countries",
                "All Countries",
                all_countries_tables,
                multi_column=False,
                export_suffix=export_suffix,
            )
        )
        exported_paths.append(
            export_document_png(
                "form_all_confederations",
                "All Confederations",
                all_confederations_tables,
                multi_column=False,
                export_suffix=export_suffix,
            )
        )
        for table in all_confederations_tables:
            exported_paths.append(
                export_document_png(
                    str(table["stem"]),
                    str(table["title"]),
                    [table],
                    multi_column=False,
                    export_suffix=export_suffix,
                )
            )
    return exported_paths


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
            st.Page(render_v1_navigation_page, title="V1 Team Strength", icon=":material/looks_one:"),
            st.Page(render_v2_form_navigation_page, title="V2 Form", icon=":material/looks_two:"),
            st.Page(
                render_v2_probabilities_navigation_page,
                title="V2 Probabilities",
                icon=":material/functions:",
            ),
            st.Page(
                render_v3_probabilities_navigation_page,
                title="V3 Poisson Regression (Legacy)",
                icon=":material/looks_3:",
            ),
            st.Page(
                render_v4_probabilities_navigation_page,
                title="V4 Enhanced Poisson",
                icon=":material/looks_4:",
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

