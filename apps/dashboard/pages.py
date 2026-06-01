from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .config import (
    BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    DEFAULT_RECENT_MATCH_WINDOW,
    DEFAULT_SIMULATION_LABEL,
    DEFAULT_V2_TRAINING_SCOPE,
    DEFAULT_V3_TRAINING_SCOPE,
    DEFAULT_V4_TRAINING_SCOPE,
    FORM_WINDOW_MAX,
    FORM_WINDOW_MIN,
    GROUP_ORDER,
    MODEL_LABEL,
    MODEL_SUMMARY,
    MODEL_VERSION,
    SIMULATION_COUNT,
    SIMULATION_OPTIONS,
    TRAINING_SCOPE_ALL_INTERNATIONAL,
    TRAINING_SCOPE_LABEL_BY_VALUE,
    TRAINING_SCOPE_LABELS,
    V1_STATE_KEY,
    V1_VIEW_OPTIONS,
    V2_BACKTEST_2022_STATE_KEY,
    V2_MODEL_LABEL,
    V2_MODEL_SUMMARY,
    V2_MODEL_VERSION,
    V2_PROB_STATE_KEY,
    V2_PROB_VIEW_OPTIONS,
    V2_STATE_KEY,
    V2_VIEW_OPTIONS,
    V3_BACKTEST_2022_STATE_KEY,
    V3_MODEL_LABEL,
    V3_MODEL_SUMMARY,
    V3_MODEL_VERSION,
    V3_PROB_STATE_KEY,
    V4_BACKTEST_2022_STATE_KEY,
    V4_MODEL_LABEL,
    V4_MODEL_SUMMARY,
    V4_MODEL_VERSION,
    V4_PROB_STATE_KEY,
    WEIGHTED_FORM_COMPOSITE_WEIGHTS,
    build_v2_team_strengths,
    build_v3_team_feature_table,
    build_weighted_form_table,
    fit_v2_match_multinomial_model,
    fit_v3_poisson_models,
)
from .data import load_data, load_validation_artifacts, load_world_cup_logo_data_uri
from .export import (
    combine_table_download_frames,
    dataframe_to_csv_bytes,
    tables_to_download_frame,
)
from .modeling import (
    default_simulation_settings,
    ensure_dashboard_probability_columns,
    load_v3_poisson_model,
    load_v4_poisson_model,
    run_v2_backtest_2022_dashboard,
    run_v3_backtest_2022_dashboard,
    run_v4_backtest_2022_dashboard,
    simulate_probabilities,
    simulate_probabilities_v3_dashboard,
    simulate_probabilities_v4_dashboard,
)
from .model_registry import PRIMARY_MODEL
from .projection_jobs import build_v2_probability_artifact, build_v3_probability_artifact, build_v4_probability_artifact
from .rendering import (
    all_teams_table_frame,
    build_confederation_form_tables,
    build_form_view_tables,
    build_table_html,
    chart_subtitle,
    current_form_view_tables,
    current_view_tables,
    format_decimal,
    format_percent,
    form_table_frame,
    get_first_kickoff_details,
    inject_styles,
    ordered_confederations,
    render_countdown_timer,
    render_dashboard_header,
    render_filter_bar,
    render_tables,
)
from .simulation_store import (
    DEFAULT_SIMULATION_SEED,
    ArtifactLoadResult,
    ArtifactSettings,
    load_artifact,
    load_or_create_artifact,
)

def build_home_metric_card(label: str, value: str, detail: str) -> str:
    """Return one compact home-page metric card."""
    return (
        '<div class="wc-home-metric">'
        f'<div class="wc-home-metric-label">{html.escape(label)}</div>'
        f'<div class="wc-home-metric-value">{html.escape(value)}</div>'
        f'<div class="wc-home-metric-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def build_home_route_card(title: str, destination: str, description: str) -> str:
    """Return one home-page navigation card."""
    return (
        '<div class="wc-home-route-card">'
        f'<div class="wc-home-route-title">{html.escape(title)}</div>'
        f'<div class="wc-home-route-destination">{html.escape(destination)}</div>'
        f'<p class="wc-home-route-copy">{html.escape(description)}</p>'
        "</div>"
    )


def build_home_model_card(version: str, title: str, summary: str, recommended: bool = False) -> str:
    """Return one compact model comparison card."""
    recommended_badge = '<span class="wc-home-badge">Recommended</span>' if recommended else ""
    card_class = "wc-home-model-card wc-home-model-card-recommended" if recommended else "wc-home-model-card"
    return (
        f'<div class="{card_class}">'
        f'<div class="wc-home-model-version">{html.escape(version)} {recommended_badge}</div>'
        f'<div class="wc-home-model-title">{html.escape(title)}</div>'
        f'<p class="wc-home-model-copy">{html.escape(summary)}</p>'
        "</div>"
    )


def display_artifact_status(load_result: ArtifactLoadResult, model_label: str) -> None:
    """Render cache status and non-fatal artifact warnings."""
    for warning in load_result.warnings:
        st.warning(warning)
    if load_result.artifact is None:
        return
    artifact = load_result.artifact
    created_at = artifact.created_at_utc or "unknown time"
    source_label = "runtime" if artifact.source == "runtime" else "official"
    if load_result.created:
        st.caption(f"Fresh {source_label} {model_label} simulation run saved at {created_at}.")
    else:
        st.caption(f"Using {source_label} cached {model_label} simulation run from {created_at}.")


def _validation_filter_options(frame: pd.DataFrame, column: str) -> list:
    if frame.empty or column not in frame.columns:
        return []
    return sorted(frame[column].dropna().unique().tolist())


def _filter_validation_frame(
    frame: pd.DataFrame,
    *,
    model_families: list[str] | None = None,
    scopes: list[str] | None = None,
    fold_years: list[int] | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if model_families and "model_family" in filtered.columns:
        filtered = filtered[filtered["model_family"].isin(model_families)]
    if scopes and "scope" in filtered.columns:
        filtered = filtered[filtered["scope"].isin(scopes)]
    if fold_years and "fold_year" in filtered.columns:
        filtered = filtered[filtered["fold_year"].isin(fold_years)]
    return filtered


def render_validation_findings_cards(headline_findings: dict[str, str]) -> None:
    """Render the multi-fold validation headline cards."""
    cards = st.columns(4)
    cards[0].metric("Best mean log loss", headline_findings.get("best_log_loss", "Unavailable"))
    cards[1].metric("Best mean Brier", headline_findings.get("best_brier", "Unavailable"))
    cards[2].metric("Best top-1 accuracy", headline_findings.get("best_top1", "Unavailable"))
    cards[3].metric("Dominance conclusion", headline_findings.get("dominance", "Unavailable"))


def render_drilldown_backtest_banner() -> None:
    """Point users from one-year backtest drilldowns to the current multi-fold findings."""
    st.info(
        "For the current 2014/2018/2022 multi-fold validation findings, open Backtests > Multi-Fold Backtests. "
        "This page remains an interactive 2022 single-holdout drilldown."
    )


def render_table_data_downloads(
    *,
    key_prefix: str,
    current_frame: pd.DataFrame,
    current_filename: str,
    all_frame: pd.DataFrame | None = None,
    all_filename: str | None = None,
) -> None:
    """Render a collapsed table preview with CSV download buttons."""
    with st.expander("Download table data", expanded=False):
        if current_frame.empty:
            st.caption("No table data is available for this view.")
        else:
            st.table(current_frame)
            st.download_button(
                "Download current view CSV",
                data=dataframe_to_csv_bytes(current_frame),
                file_name=current_filename,
                mime="text/csv",
                width="stretch",
                key=f"{key_prefix}_current_table_csv",
            )

        if all_frame is not None and all_filename is not None and not all_frame.empty:
            st.download_button(
                "Download all tables CSV",
                data=dataframe_to_csv_bytes(all_frame),
                file_name=all_filename,
                mime="text/csv",
                width="stretch",
                key=f"{key_prefix}_all_tables_csv",
            )


def probability_all_tables_download_frame(
    dashboard_df: pd.DataFrame,
    simulation_count: int | None = None,
) -> pd.DataFrame:
    """Return group tables plus the all-country table as one CSV-ready frame."""
    group_tables = current_view_tables(
        dashboard_df,
        "All groups",
        "",
        simulation_count=simulation_count,
    )
    all_countries_table = current_view_tables(
        dashboard_df,
        "All Countries",
        "",
        simulation_count=simulation_count,
    )
    return combine_table_download_frames(group_tables + all_countries_table, section_column="section")


def form_all_tables_download_frame(form_df: pd.DataFrame, form_match_window: int) -> pd.DataFrame:
    """Return all V2 form tables as one CSV-ready frame."""
    all_countries_table = current_form_view_tables(
        form_df,
        "All Countries",
        "",
        form_match_window=form_match_window,
    )
    confederation_tables = current_form_view_tables(
        form_df,
        "All confederations",
        "",
        form_match_window=form_match_window,
    )
    return combine_table_download_frames(all_countries_table + confederation_tables, section_column="section")


def load_or_run_probability_artifact(
    settings: ArtifactSettings,
    build_artifact,
    *,
    spinner_label: str,
    force_refresh: bool,
) -> ArtifactLoadResult:
    """Load a cached probability artifact or run and save a fresh runtime artifact."""
    loaded = ArtifactLoadResult(artifact=None)
    if not force_refresh:
        loaded = load_artifact(settings)
        if loaded.artifact is not None:
            return loaded

    with st.spinner(spinner_label):
        created = load_or_create_artifact(
            settings,
            build_artifact,
            force_refresh=True,
            write_tier="runtime",
        )
    return ArtifactLoadResult(
        artifact=created.artifact,
        warnings=loaded.warnings + created.warnings,
        created=True,
    )


def migrate_default_training_scope(
    state_key: str,
    *,
    default_scope: str,
    widget_key: str | None = None,
) -> None:
    """Reset stale pre-default-change training scope state to the active default."""
    current_settings = st.session_state.get(state_key)
    if (
        isinstance(current_settings, dict)
        and str(current_settings.get("training_scope", "")) == TRAINING_SCOPE_ALL_INTERNATIONAL
        and default_scope != TRAINING_SCOPE_ALL_INTERNATIONAL
    ):
        st.session_state[state_key] = {
            **current_settings,
            "training_scope": default_scope,
        }
        if widget_key is not None:
            st.session_state.pop(widget_key, None)


def first_team_by_metric(df: pd.DataFrame, metric_column: str) -> pd.Series:
    """Return the leading team row for a numeric probability metric."""
    sortable = df.copy()
    sortable[metric_column] = pd.to_numeric(sortable[metric_column], errors="coerce").fillna(0.0)
    return sortable.sort_values(
        [metric_column, "elo_rating", "world_rank"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]


def team_value_detail(row: pd.Series, metric_column: str) -> tuple[str, str]:
    """Return a display name and probability detail for a team metric row."""
    return str(row["display_name"]), f'{format_percent(float(row[metric_column]))} probability'


def render_v1_dashboard() -> None:
    """Render the version 1 probability dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V1_STATE_KEY not in st.session_state:
        st.session_state[V1_STATE_KEY] = default_simulation_settings()
    current_settings = dict(st.session_state[V1_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(st.session_state.get("v1_simulation_label", current_settings["simulation_label"]))
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2026 V1",
    )
    with render_filter_bar():
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v1_simulation_label",
        )
        view_mode = st.radio("View", V1_VIEW_OPTIONS, horizontal=True, key="v1_view_mode")
        selected_group = (
            st.selectbox("Group", GROUP_ORDER, index=0, key="v1_selected_group")
            if view_mode == "Single group"
            else GROUP_ORDER[0]
        )

    st.session_state[V1_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": DEFAULT_RECENT_MATCH_WINDOW,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Model {MODEL_VERSION}: {MODEL_SUMMARY}. "
        "Probabilities come from a fixture-by-fixture group simulation using the real 2026 schedule, "
        "an Elo-only baseline (100% / 0%), "
        f"recent form from the last {DEFAULT_RECENT_MATCH_WINDOW} matches, "
        "built from points vs goal difference (70% / 30%), "
        "and a ratings-vs-form blend (50% / 50%). "
        "Top 8 3rd% is the share of runs where a team finishes third and still advances. "
        "KO% means reaching the Round of 32; R16%, QF%, SF%, Final%, and Champion% track deeper knockout progression. "
        "This page is isolated to the original probability model."
    )
    with st.spinner(f"Running {simulation_count:,} simulations..."):
        dashboard_df = simulate_probabilities(
            base_df=base_df,
            fixtures_df=fixtures_df,
            lead_in_df=lead_in_df,
            simulations=simulation_count,
        )
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    tables = current_view_tables(
        dashboard_df,
        view_mode,
        selected_group,
        simulation_count=simulation_count,
    )
    multi_column = view_mode == "All groups"

    current_download_frame = tables_to_download_frame(tables, section_column="group")
    render_table_data_downloads(
        key_prefix="v1",
        current_frame=current_download_frame,
        current_filename=f"v1_{view_mode.lower().replace(' ', '_')}.csv",
        all_frame=probability_all_tables_download_frame(dashboard_df, simulation_count=simulation_count),
        all_filename="v1_probability_tables.csv",
    )

    render_tables(tables, multi_column=multi_column)


def render_v2_dashboard() -> None:
    """Render the version 2 weighted-form dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V2_STATE_KEY not in st.session_state:
        st.session_state[V2_STATE_KEY] = default_simulation_settings()
    current_settings = dict(st.session_state[V2_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(st.session_state.get("v2_simulation_label", current_settings["simulation_label"]))
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2026 V2",
    )
    filter_bar = render_filter_bar()
    with filter_bar:
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v2_simulation_label",
        )
        form_match_window = int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW))
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, form_match_window)),
                key="v2_form_match_window",
            )
        )
        weight_cols = st.columns(4)
        with weight_cols[0]:
            results_weight = int(
                st.slider(
                    "Results weight",
                    min_value=0,
                    max_value=100,
                    value=int(current_settings.get("v2_results_weight", int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[0] * 100)))),
                    key="v2_results_weight",
                )
            )
        with weight_cols[1]:
            gd_weight = int(
                st.slider(
                    "GD weight",
                    min_value=0,
                    max_value=100,
                    value=int(current_settings.get("v2_gd_weight", int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[1] * 100)))),
                    key="v2_gd_weight",
                )
            )
        with weight_cols[2]:
            perf_weight = int(
                st.slider(
                    "PoE weight",
                    min_value=0,
                    max_value=100,
                    value=int(current_settings.get("v2_perf_weight", int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[2] * 100)))),
                    key="v2_perf_weight",
                )
            )
        with weight_cols[3]:
            elo_delta_weight = int(
                st.slider(
                    "Elo-delta weight",
                    min_value=0,
                    max_value=100,
                    value=int(current_settings.get("v2_elo_delta_weight", int(round(WEIGHTED_FORM_COMPOSITE_WEIGHTS[3] * 100)))),
                    key="v2_elo_delta_weight",
                )
            )
        view_mode = st.radio("View", V2_VIEW_OPTIONS, horizontal=True, key="v2_view_mode")
    form_composite_weights = (
        results_weight,
        gd_weight,
        perf_weight,
        elo_delta_weight,
    )
    st.session_state[V2_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "v2_results_weight": results_weight,
        "v2_gd_weight": gd_weight,
        "v2_perf_weight": perf_weight,
        "v2_elo_delta_weight": elo_delta_weight,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"V2 isolates the history-aware model from V1. This page ranks all 48 teams using rating (40%), weighted lead-in form (40%), "
        f"and World Cup history (20%). Form covers the last {form_match_window} Elo-rated matches with component weights: "
        f"Results {results_weight}, GD {gd_weight}, PoE {perf_weight}, Elo Delta {elo_delta_weight}. "
        "History blends weighted World Cup placement (70%) with weighted appearance count (30%) across the previous 5 World Cup editions, "
        "with DNQ editions scored as zero."
    )
    with st.spinner(f"Computing V2 history-aware strength for the last {form_match_window} matches..."):
        form_df = build_v2_team_strengths(
            base_df,
            lead_in_df,
            match_window=form_match_window,
            form_composite_weights=form_composite_weights,
        )
    available_confederations = ordered_confederations(form_df)
    with filter_bar:
        selected_confederation = (
            st.selectbox(
                "Confederation",
                available_confederations,
                index=0,
                key="v2_selected_confederation",
            )
            if view_mode == "Single confederation" and available_confederations
            else ""
        )
    tables = current_form_view_tables(
        form_df,
        view_mode,
        selected_confederation,
        form_match_window=form_match_window,
    )

    render_table_data_downloads(
        key_prefix="v2_form",
        current_frame=tables_to_download_frame(tables, section_column="confederation"),
        current_filename=f"v2_form_{view_mode.lower().replace(' ', '_')}.csv",
        all_frame=form_all_tables_download_frame(form_df, form_match_window=form_match_window),
        all_filename="v2_form_tables.csv",
    )
    render_tables(tables, multi_column=False)


def render_v2_probabilities_dashboard() -> None:
    """Render the version 2 multinomial probability dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V2_PROB_STATE_KEY not in st.session_state:
        st.session_state[V2_PROB_STATE_KEY] = default_simulation_settings()
    current_settings = dict(st.session_state[V2_PROB_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(st.session_state.get("v2_prob_simulation_label", current_settings["simulation_label"]))
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2026 V2 Probabilities",
        model_version=V2_MODEL_VERSION,
        model_label=V2_MODEL_LABEL,
    )
    with render_filter_bar("Model Filters"):
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v2_prob_simulation_label",
        )
        form_match_window = int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW))
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, form_match_window)),
                key="v2_prob_form_match_window",
            )
        )
        current_training_scope = str(current_settings.get("training_scope", DEFAULT_V2_TRAINING_SCOPE))
        training_scope_label = st.radio(
            "Training data",
            tuple(TRAINING_SCOPE_LABELS.keys()),
            index=tuple(TRAINING_SCOPE_LABELS.keys()).index(
                TRAINING_SCOPE_LABEL_BY_VALUE.get(current_training_scope, "World Cup only")
            ),
            horizontal=True,
            key="v2_prob_training_scope",
        )
        training_scope = TRAINING_SCOPE_LABELS[training_scope_label]
        view_mode = st.radio("View", V2_PROB_VIEW_OPTIONS, horizontal=True, key="v2_prob_view_mode")
        selected_group = (
            st.selectbox("Group", GROUP_ORDER, index=0, key="v2_prob_selected_group")
            if view_mode == "Single group"
            else GROUP_ORDER[0]
        )
        rerun_simulations = st.button("Rerun V2 simulations", key="v2_prob_rerun_simulations")

    st.session_state[V2_PROB_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "training_scope": training_scope,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Legacy comparison model {V2_MODEL_VERSION}: {V2_MODEL_SUMMARY}. "
        f"The v2 page trains a three-class multinomial regression using `{training_scope}` training data, "
        f"then simulates 2026 tournament outcomes using pre-tournament Elo, weighted form from the last {form_match_window} Elo-rated matches, "
        "and prior-5-edition World Cup history features. Knockout draws are interpreted using the local historical file semantics: "
        "level before penalties, then resolved by the model's non-draw split."
    )
    artifact_settings = ArtifactSettings(
        model_id="v2",
        model_version=V2_MODEL_VERSION,
        data_build_date=str(metadata.get("build_date", "")),
        simulations=simulation_count,
        match_window=form_match_window,
        training_scope=training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    load_result = load_or_run_probability_artifact(
        artifact_settings,
        lambda: build_v2_probability_artifact(
            base_df=base_df,
            fixtures_df=fixtures_df,
            lead_in_df=lead_in_df,
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
            seed=DEFAULT_SIMULATION_SEED,
            bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        ),
        spinner_label=f"Training v2 model and running {simulation_count:,} simulations...",
        force_refresh=rerun_simulations,
    )
    display_artifact_status(load_result, "V2 legacy")
    if load_result.artifact is None:
        st.error("V2 probability artifact could not be loaded or created.")
        return
    dashboard_df = load_result.artifact.dashboard_df
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    tables = current_view_tables(
        dashboard_df,
        view_mode,
        selected_group,
        simulation_count=simulation_count,
    )
    multi_column = view_mode == "All groups"

    current_download_frame = tables_to_download_frame(tables, section_column="group")
    render_table_data_downloads(
        key_prefix="v2_prob",
        current_frame=current_download_frame,
        current_filename=f"v2_probability_{view_mode.lower().replace(' ', '_')}.csv",
        all_frame=probability_all_tables_download_frame(dashboard_df, simulation_count=simulation_count),
        all_filename="v2_probability_tables.csv",
    )

    render_tables(tables, multi_column=multi_column)


def render_v3_probabilities_dashboard() -> None:
    """Render the version 3 Poisson probability dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V3_PROB_STATE_KEY not in st.session_state:
        st.session_state[V3_PROB_STATE_KEY] = default_simulation_settings()
    migrate_default_training_scope(
        V3_PROB_STATE_KEY,
        default_scope=DEFAULT_V3_TRAINING_SCOPE,
        widget_key="v3_prob_training_scope",
    )
    current_settings = dict(st.session_state[V3_PROB_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(st.session_state.get("v3_prob_simulation_label", current_settings["simulation_label"]))
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2026 V3 Probabilities",
        model_version=V3_MODEL_VERSION,
        model_label=V3_MODEL_LABEL,
    )
    with render_filter_bar("Model Filters"):
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v3_prob_simulation_label",
        )
        form_match_window = int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW))
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, form_match_window)),
                key="v3_prob_form_match_window",
            )
        )
        current_training_scope = str(current_settings.get("training_scope", DEFAULT_V3_TRAINING_SCOPE))
        training_scope_label = st.radio(
            "Training data",
            tuple(TRAINING_SCOPE_LABELS.keys()),
            index=tuple(TRAINING_SCOPE_LABELS.keys()).index(
                TRAINING_SCOPE_LABEL_BY_VALUE.get(current_training_scope, "World Cup only")
            ),
            horizontal=True,
            key="v3_prob_training_scope",
        )
        training_scope = TRAINING_SCOPE_LABELS[training_scope_label]
        view_mode = st.radio("View", V2_PROB_VIEW_OPTIONS, horizontal=True, key="v3_prob_view_mode")
        selected_group = (
            st.selectbox("Group", GROUP_ORDER, index=0, key="v3_prob_selected_group")
            if view_mode == "Single group"
            else GROUP_ORDER[0]
        )
        rerun_simulations = st.button("Rerun V3 simulations", key="v3_prob_rerun_simulations")

    st.session_state[V3_PROB_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "training_scope": training_scope,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Legacy comparison model {V3_MODEL_VERSION}: {V3_MODEL_SUMMARY}. "
        f"The v3 page trains paired Poisson regressors using `{training_scope}` training data, "
        f"then simulates 2026 tournament outcomes using pre-tournament Elo, weighted form from the last {form_match_window} Elo-rated matches, "
        "prior-5-edition World Cup pedigree, competition-importance weighting, and host flags for Canada, Mexico, and the United States."
    )
    artifact_settings = ArtifactSettings(
        model_id="v3",
        model_version=V3_MODEL_VERSION,
        data_build_date=str(metadata.get("build_date", "")),
        simulations=simulation_count,
        match_window=form_match_window,
        training_scope=training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    load_result = load_or_run_probability_artifact(
        artifact_settings,
        lambda: build_v3_probability_artifact(
            base_df,
            fixtures_df,
            lead_in_df,
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
            seed=DEFAULT_SIMULATION_SEED,
            bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        ),
        spinner_label=f"Training v3 model and running {simulation_count:,} simulations...",
        force_refresh=rerun_simulations,
    )
    display_artifact_status(load_result, "V3")
    if load_result.artifact is None:
        st.error("V3 probability artifact could not be loaded or created.")
        return
    dashboard_df = load_result.artifact.dashboard_df
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    tables = current_view_tables(
        dashboard_df,
        view_mode,
        selected_group,
        simulation_count=simulation_count,
    )
    multi_column = view_mode == "All groups"

    current_download_frame = tables_to_download_frame(tables, section_column="group")
    render_table_data_downloads(
        key_prefix="v3_prob",
        current_frame=current_download_frame,
        current_filename=f"v3_probability_{view_mode.lower().replace(' ', '_')}.csv",
        all_frame=probability_all_tables_download_frame(dashboard_df, simulation_count=simulation_count),
        all_filename="v3_probability_tables.csv",
    )

    render_tables(tables, multi_column=multi_column)


def render_v4_probabilities_dashboard() -> None:
    """Render the version 4 enhanced Poisson probability dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V4_PROB_STATE_KEY not in st.session_state:
        st.session_state[V4_PROB_STATE_KEY] = default_simulation_settings()
    migrate_default_training_scope(
        V4_PROB_STATE_KEY,
        default_scope=DEFAULT_V4_TRAINING_SCOPE,
        widget_key="v4_prob_training_scope",
    )
    current_settings = dict(st.session_state[V4_PROB_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(st.session_state.get("v4_prob_simulation_label", current_settings["simulation_label"]))
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2026 V4 Probabilities",
        model_version=V4_MODEL_VERSION,
        model_label=V4_MODEL_LABEL,
    )
    with render_filter_bar("Model Filters"):
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v4_prob_simulation_label",
        )
        form_match_window = int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW))
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, form_match_window)),
                key="v4_prob_form_match_window",
            )
        )
        current_training_scope = str(current_settings.get("training_scope", DEFAULT_V4_TRAINING_SCOPE))
        training_scope_label = st.radio(
            "Training data",
            tuple(TRAINING_SCOPE_LABELS.keys()),
            index=tuple(TRAINING_SCOPE_LABELS.keys()).index(
                TRAINING_SCOPE_LABEL_BY_VALUE.get(current_training_scope, "World Cup only")
            ),
            horizontal=True,
            key="v4_prob_training_scope",
        )
        training_scope = TRAINING_SCOPE_LABELS[training_scope_label]
        view_mode = st.radio("View", V2_PROB_VIEW_OPTIONS, horizontal=True, key="v4_prob_view_mode")
        selected_group = (
            st.selectbox("Group", GROUP_ORDER, index=0, key="v4_prob_selected_group")
            if view_mode == "Single group"
            else GROUP_ORDER[0]
        )
        rerun_simulations = st.button("Rerun V4 simulations", key="v4_prob_rerun_simulations")

    st.session_state[V4_PROB_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "training_scope": training_scope,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Primary model {V4_MODEL_VERSION}: {V4_MODEL_SUMMARY}. "
        f"V4 is the primary enhanced Poisson model using quadratic last-{form_match_window} form, "
        "World Cup last-5 goal history, Dixon-Coles low-score correction, stage lambda multipliers, "
        f"and `{training_scope}` training data."
    )
    artifact_settings = ArtifactSettings(
        model_id="v4",
        model_version=V4_MODEL_VERSION,
        data_build_date=str(metadata.get("build_date", "")),
        simulations=simulation_count,
        match_window=form_match_window,
        training_scope=training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    load_result = load_or_run_probability_artifact(
        artifact_settings,
        lambda: build_v4_probability_artifact(
            base_df,
            fixtures_df,
            lead_in_df,
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
            seed=DEFAULT_SIMULATION_SEED,
            bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        ),
        spinner_label=f"Training v4 model and running {simulation_count:,} simulations...",
        force_refresh=rerun_simulations,
    )
    display_artifact_status(load_result, "V4")
    if load_result.artifact is None:
        st.error("V4 probability artifact could not be loaded or created.")
        return
    dashboard_df = load_result.artifact.dashboard_df
    artifact_metadata = load_result.artifact.metadata
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    st.caption(
        " | ".join(
            [
                f"alpha={artifact_metadata.get('alpha')}",
                f"rho={artifact_metadata.get('rho')}",
                f"half-life={artifact_metadata.get('time_decay_halflife_days')} days",
                f"scope={artifact_metadata.get('training_scope')}",
            ]
        )
    )
    with st.expander("V4 calibration metadata", expanded=False):
        st.json(
            {
                "stage_multipliers": artifact_metadata.get("stage_multipliers"),
                "alpha_source": artifact_metadata.get("alpha_source"),
                "rho_source": artifact_metadata.get("rho_source"),
                "quadratic_form_window": form_match_window,
            }
        )
    tables = current_view_tables(
        dashboard_df,
        view_mode,
        selected_group,
        simulation_count=simulation_count,
    )

    current_download_frame = tables_to_download_frame(tables, section_column="group")
    render_table_data_downloads(
        key_prefix="v4_prob",
        current_frame=current_download_frame,
        current_filename=f"v4_probability_{view_mode.lower().replace(' ', '_')}.csv",
        all_frame=probability_all_tables_download_frame(dashboard_df, simulation_count=simulation_count),
        all_filename="v4_probability_tables.csv",
    )

    render_tables(tables, multi_column=view_mode == "All groups")


def render_v2_2022_backtest_dashboard() -> None:
    """Render the 2022 holdout backtest page for the V2 model."""
    inject_styles()

    _, fixtures_df, _, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V2_BACKTEST_2022_STATE_KEY not in st.session_state:
        st.session_state[V2_BACKTEST_2022_STATE_KEY] = default_simulation_settings()
    current_settings = dict(st.session_state[V2_BACKTEST_2022_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(
        st.session_state.get("v2_backtest_2022_simulation_label", current_settings["simulation_label"])
    )
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2022 V2 Backtest",
        model_version=V2_MODEL_VERSION,
        model_label=V2_MODEL_LABEL,
    )
    with render_filter_bar("Model Filters"):
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v2_backtest_2022_simulation_label",
        )
        form_match_window = int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW))
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, form_match_window)),
                key="v2_backtest_2022_form_match_window",
            )
        )
        current_training_scope = str(current_settings.get("training_scope", DEFAULT_V2_TRAINING_SCOPE))
        training_scope_label = st.radio(
            "Training data",
            tuple(TRAINING_SCOPE_LABELS.keys()),
            index=tuple(TRAINING_SCOPE_LABELS.keys()).index(
                TRAINING_SCOPE_LABEL_BY_VALUE.get(current_training_scope, "World Cup only")
            ),
            horizontal=True,
            key="v2_backtest_2022_training_scope",
        )
        training_scope = TRAINING_SCOPE_LABELS[training_scope_label]
    st.session_state[V2_BACKTEST_2022_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "training_scope": training_scope,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Model {V2_MODEL_VERSION}: {V2_MODEL_SUMMARY}. "
        f"This page trains the V2 multinomial model with 2022 excluded from training using `{training_scope}`, then backtests the actual 2022 World Cup "
        f"using pre-tournament Elo, weighted form from the last {form_match_window} Elo-rated matches, "
        "and prior-5-edition World Cup history features. "
        "It reports match-level calibration plus tournament-level hit rates."
    )
    render_drilldown_backtest_banner()

    with st.spinner(f"Running the 2022 holdout backtest with {simulation_count:,} simulations..."):
        backtest = run_v2_backtest_2022_dashboard(
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
        )

    summary_metrics = dict(backtest["summary_metrics"])
    match_predictions = pd.DataFrame(backtest["match_predictions"]).copy()
    team_backtest_table = pd.DataFrame(backtest["team_backtest_table"]).copy()
    group_backtest_table = pd.DataFrame(backtest["group_backtest_table"]).copy()
    bracket_summary = dict(backtest["bracket_summary"])

    team_name_lookup = (
        team_backtest_table.loc[:, ["team_id", "display_name"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .set_index("team_id")["display_name"]
        .astype(str)
        .to_dict()
    )
    predicted_champion_name = team_name_lookup.get(summary_metrics["predicted_champion_team_id"], str(summary_metrics["predicted_champion_team_id"]))
    actual_champion_name = team_name_lookup.get(summary_metrics["actual_champion_team_id"], str(summary_metrics["actual_champion_team_id"]))

    metric_cols = st.columns(6)
    metric_cols[0].metric("Log Loss", format_decimal(summary_metrics["multiclass_log_loss"], 4))
    metric_cols[1].metric("Brier Score", format_decimal(summary_metrics["multiclass_brier_score"], 4))
    metric_cols[2].metric("Top-1 Accuracy", format_percent(summary_metrics["top1_match_accuracy"]))
    metric_cols[3].metric("Champion Hit", "Yes" if summary_metrics["exact_champion_hit"] else "No")
    metric_cols[4].metric("Semi-final Hits", f"{int(summary_metrics['semifinal_hit_count'])}/4")
    metric_cols[5].metric("R16 Hits", f"{int(summary_metrics['round_of_16_hit_count'])}/16")

    st.markdown(
        f"**Champion**  Predicted: `{predicted_champion_name}` | Actual: `{actual_champion_name}`"
    )

    bracket_cols = st.columns(2)
    with bracket_cols[0]:
        predicted_finalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["predicted_finalist_team_ids"])
        predicted_semifinalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["predicted_semifinalist_team_ids"])
        st.markdown("**Predicted Bracket Summary**")
        st.write(f"Champion: {team_name_lookup.get(bracket_summary['predicted_champion_team_id'], str(bracket_summary['predicted_champion_team_id']))}")
        st.write(f"Finalists: {predicted_finalists}")
        st.write(f"Semi-finalists: {predicted_semifinalists}")
    with bracket_cols[1]:
        actual_finalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["actual_finalist_team_ids"])
        actual_semifinalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["actual_semifinalist_team_ids"])
        st.markdown("**Actual 2022 Outcome**")
        st.write(f"Champion: {team_name_lookup.get(bracket_summary['actual_champion_team_id'], str(bracket_summary['actual_champion_team_id']))}")
        st.write(f"Finalists: {actual_finalists}")
        st.write(f"Semi-finalists: {actual_semifinalists}")

    st.markdown("**Match Predictions**")
    st.dataframe(
        match_predictions.loc[
            :,
            [
                "match_number",
                "stage",
                "group_code",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "home_win_prob",
                "draw_prob",
                "away_win_prob",
                "predicted_outcome",
                "actual_outcome",
                "top1_correct",
            ],
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Group Finish Backtest**")
    st.dataframe(
        group_backtest_table,
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Team Advancement Backtest**")
    st.dataframe(
        team_backtest_table.loc[
            :,
            [
                "group_code",
                "display_name",
                "actual_stage",
                "actual_group_rank",
                "modal_group_rank",
                "r16_prob",
                "qf_prob",
                "sf_prob",
                "final_prob",
                "champion_prob",
            ],
        ],
        width="stretch",
        hide_index=True,
    )


def render_v3_2022_backtest_dashboard() -> None:
    """Render the 2022 holdout backtest page for the V3 model."""
    inject_styles()

    _, fixtures_df, _, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V3_BACKTEST_2022_STATE_KEY not in st.session_state:
        st.session_state[V3_BACKTEST_2022_STATE_KEY] = default_simulation_settings()
    migrate_default_training_scope(
        V3_BACKTEST_2022_STATE_KEY,
        default_scope=DEFAULT_V3_TRAINING_SCOPE,
        widget_key="v3_backtest_2022_training_scope",
    )
    current_settings = dict(st.session_state[V3_BACKTEST_2022_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(
        st.session_state.get("v3_backtest_2022_simulation_label", current_settings["simulation_label"])
    )
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2022 V3 Backtest",
        model_version=V3_MODEL_VERSION,
        model_label=V3_MODEL_LABEL,
    )
    with render_filter_bar("Model Filters"):
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v3_backtest_2022_simulation_label",
        )
        form_match_window = int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW))
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, form_match_window)),
                key="v3_backtest_2022_form_match_window",
            )
        )
        current_training_scope = str(current_settings.get("training_scope", DEFAULT_V3_TRAINING_SCOPE))
        training_scope_label = st.radio(
            "Training data",
            tuple(TRAINING_SCOPE_LABELS.keys()),
            index=tuple(TRAINING_SCOPE_LABELS.keys()).index(
                TRAINING_SCOPE_LABEL_BY_VALUE.get(current_training_scope, "World Cup only")
            ),
            horizontal=True,
            key="v3_backtest_2022_training_scope",
        )
        training_scope = TRAINING_SCOPE_LABELS[training_scope_label]
    st.session_state[V3_BACKTEST_2022_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "training_scope": training_scope,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Model {V3_MODEL_VERSION}: {V3_MODEL_SUMMARY}. "
        f"This page trains the V3 Poisson goal model using `{training_scope}` through the eve of the 2022 World Cup, "
        f"then backtests the actual tournament using weighted form from the last {form_match_window} Elo-rated matches and prior-5-edition pedigree features. "
        "It reports match-level calibration plus tournament-level hit rates."
    )
    render_drilldown_backtest_banner()

    with st.spinner(f"Running the 2022 V3 backtest with {simulation_count:,} simulations..."):
        backtest = run_v3_backtest_2022_dashboard(
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
        )

    summary_metrics = dict(backtest["summary_metrics"])
    match_predictions = pd.DataFrame(backtest["match_predictions"]).copy()
    team_backtest_table = pd.DataFrame(backtest["team_backtest_table"]).copy()
    group_backtest_table = pd.DataFrame(backtest["group_backtest_table"]).copy()
    bracket_summary = dict(backtest["bracket_summary"])

    team_name_lookup = (
        team_backtest_table.loc[:, ["team_id", "display_name"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .set_index("team_id")["display_name"]
        .astype(str)
        .to_dict()
    )
    predicted_champion_name = team_name_lookup.get(summary_metrics["predicted_champion_team_id"], str(summary_metrics["predicted_champion_team_id"]))
    actual_champion_name = team_name_lookup.get(summary_metrics["actual_champion_team_id"], str(summary_metrics["actual_champion_team_id"]))

    metric_cols = st.columns(6)
    metric_cols[0].metric("Log Loss", format_decimal(summary_metrics["multiclass_log_loss"], 4))
    metric_cols[1].metric("Brier Score", format_decimal(summary_metrics["multiclass_brier_score"], 4))
    metric_cols[2].metric("Top-1 Accuracy", format_percent(summary_metrics["top1_match_accuracy"]))
    metric_cols[3].metric("Champion Hit", "Yes" if summary_metrics["exact_champion_hit"] else "No")
    metric_cols[4].metric("Semi-final Hits", f"{int(summary_metrics['semifinal_hit_count'])}/4")
    metric_cols[5].metric("R16 Hits", f"{int(summary_metrics['round_of_16_hit_count'])}/16")
    st.caption(
        f"Draw calibration: predicted {format_percent(summary_metrics['draw_rate_predicted'])} | actual {format_percent(summary_metrics['draw_rate_actual'])}"
    )

    st.markdown(
        f"**Champion**  Predicted: `{predicted_champion_name}` | Actual: `{actual_champion_name}`"
    )

    bracket_cols = st.columns(2)
    with bracket_cols[0]:
        predicted_finalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["predicted_finalist_team_ids"])
        predicted_semifinalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["predicted_semifinalist_team_ids"])
        st.markdown("**Predicted Bracket Summary**")
        st.write(f"Champion: {team_name_lookup.get(bracket_summary['predicted_champion_team_id'], str(bracket_summary['predicted_champion_team_id']))}")
        st.write(f"Finalists: {predicted_finalists}")
        st.write(f"Semi-finalists: {predicted_semifinalists}")
    with bracket_cols[1]:
        actual_finalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["actual_finalist_team_ids"])
        actual_semifinalists = ", ".join(team_name_lookup.get(team_id, str(team_id)) for team_id in bracket_summary["actual_semifinalist_team_ids"])
        st.markdown("**Actual 2022 Outcome**")
        st.write(f"Champion: {team_name_lookup.get(bracket_summary['actual_champion_team_id'], str(bracket_summary['actual_champion_team_id']))}")
        st.write(f"Finalists: {actual_finalists}")
        st.write(f"Semi-finalists: {actual_semifinalists}")

    st.markdown("**Match Predictions**")
    st.dataframe(
        match_predictions.loc[
            :,
            [
                "match_number",
                "stage",
                "group_code",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "lambda_home",
                "lambda_away",
                "home_win_prob",
                "draw_prob",
                "away_win_prob",
                "predicted_outcome",
                "actual_outcome",
                "top1_correct",
            ],
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Group Finish Backtest**")
    st.dataframe(
        group_backtest_table,
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Team Advancement Backtest**")
    st.dataframe(
        team_backtest_table.loc[
            :,
            [
                "group_code",
                "display_name",
                "actual_stage",
                "actual_group_rank",
                "modal_group_rank",
                "r16_prob",
                "qf_prob",
                "sf_prob",
                "final_prob",
                "champion_prob",
            ],
        ],
        width="stretch",
        hide_index=True,
    )


def render_v4_2022_backtest_dashboard() -> None:
    """Render the 2022 holdout backtest page for the V4 model."""
    inject_styles()

    _, fixtures_df, _, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V4_BACKTEST_2022_STATE_KEY not in st.session_state:
        st.session_state[V4_BACKTEST_2022_STATE_KEY] = default_simulation_settings()
    migrate_default_training_scope(
        V4_BACKTEST_2022_STATE_KEY,
        default_scope=DEFAULT_V4_TRAINING_SCOPE,
        widget_key="v4_backtest_2022_training_scope",
    )
    current_settings = dict(st.session_state[V4_BACKTEST_2022_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
    initial_simulation_label = str(
        st.session_state.get("v4_backtest_2022_simulation_label", current_settings["simulation_label"])
    )
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_OPTIONS[initial_simulation_label],
        title="World Cup 2022 V4 Backtest",
        model_version=V4_MODEL_VERSION,
        model_label=V4_MODEL_LABEL,
    )
    with render_filter_bar("Model Filters"):
        simulation_label = st.radio(
            "Simulation runs",
            simulation_labels,
            index=simulation_labels.index(current_settings["simulation_label"]),
            horizontal=True,
            key="v4_backtest_2022_simulation_label",
        )
        form_match_window = int(
            st.slider(
                "Last k matches",
                min_value=FORM_WINDOW_MIN,
                max_value=FORM_WINDOW_MAX,
                value=max(FORM_WINDOW_MIN, min(FORM_WINDOW_MAX, int(current_settings.get("form_match_window", DEFAULT_RECENT_MATCH_WINDOW)))),
                key="v4_backtest_2022_form_match_window",
            )
        )
        current_training_scope = str(current_settings.get("training_scope", DEFAULT_V4_TRAINING_SCOPE))
        training_scope_label = st.radio(
            "Training data",
            tuple(TRAINING_SCOPE_LABELS.keys()),
            index=tuple(TRAINING_SCOPE_LABELS.keys()).index(
                TRAINING_SCOPE_LABEL_BY_VALUE.get(current_training_scope, "World Cup only")
            ),
            horizontal=True,
            key="v4_backtest_2022_training_scope",
        )
        training_scope = TRAINING_SCOPE_LABELS[training_scope_label]
    st.session_state[V4_BACKTEST_2022_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "training_scope": training_scope,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    st.caption(
        f"Model {V4_MODEL_VERSION}: {V4_MODEL_SUMMARY}. "
        "This V4 holdout uses quadratic form, Dixon-Coles probabilities, stage effects, and time-decayed training weights."
    )
    render_drilldown_backtest_banner()
    with st.spinner(f"Running the 2022 V4 backtest with {simulation_count:,} simulations..."):
        backtest = run_v4_backtest_2022_dashboard(
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
        )

    summary_metrics = dict(backtest["summary_metrics"])
    match_predictions = pd.DataFrame(backtest["match_predictions"]).copy()
    team_backtest_table = pd.DataFrame(backtest["team_backtest_table"]).copy()
    group_backtest_table = pd.DataFrame(backtest["group_backtest_table"]).copy()
    training_metadata = dict(backtest.get("training_metadata", {}))

    metric_cols = st.columns(6)
    metric_cols[0].metric("Log Loss", format_decimal(summary_metrics["multiclass_log_loss"], 4))
    metric_cols[1].metric("Brier Score", format_decimal(summary_metrics["multiclass_brier_score"], 4))
    metric_cols[2].metric("Top-1 Accuracy", format_percent(summary_metrics["top1_match_accuracy"]))
    metric_cols[3].metric("Champion Hit", "Yes" if summary_metrics["exact_champion_hit"] else "No")
    metric_cols[4].metric("Semi-final Hits", f"{int(summary_metrics['semifinal_hit_count'])}/4")
    metric_cols[5].metric("R16 Hits", f"{int(summary_metrics['round_of_16_hit_count'])}/16")
    st.caption(
        f"Draw calibration: predicted {format_percent(summary_metrics['draw_rate_predicted'])} | actual {format_percent(summary_metrics['draw_rate_actual'])}"
    )
    with st.expander("V4 training metadata", expanded=False):
        st.json(training_metadata)

    st.markdown("**Match Predictions**")
    match_columns = [
        "match_number",
        "stage",
        "group_code",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "lambda_home_adj",
        "lambda_away_adj",
        "rho",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "predicted_outcome",
        "actual_outcome",
        "top1_correct",
    ]
    st.dataframe(match_predictions.loc[:, [column for column in match_columns if column in match_predictions.columns]], width="stretch", hide_index=True)
    st.markdown("**Group Finish Backtest**")
    st.dataframe(group_backtest_table, width="stretch", hide_index=True)
    st.markdown("**Team Advancement Backtest**")
    st.dataframe(team_backtest_table, width="stretch", hide_index=True)


def render_v4_rolling_backtest_dashboard() -> None:
    """Render artifact-backed multi-fold validation findings."""
    inject_styles()
    _, _, _, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        SIMULATION_COUNT,
        title="Multi-Fold Backtest Findings",
        model_version="Validation",
        model_label="2014, 2018, and 2022 holdout folds",
    )

    artifacts = load_validation_artifacts()
    if not artifacts["available"]:
        st.warning(artifacts["warning"])
        return

    metadata = dict(artifacts["metadata"])
    run_settings = dict(metadata.get("run_settings", {}))
    st.caption(
        "Committed validation artifacts are loaded from disk, so this page does not rerun the expensive 20,000-simulation "
        "multi-fold backtest. See [docs/model_card.md](docs/model_card.md) for full validation detail."
    )
    st.caption(
        f"Generated at {metadata.get('generated_at_utc', 'unknown time')} | "
        f"seed {run_settings.get('seed', 'unknown')} | simulations {int(run_settings.get('simulations', 0)):,}"
    )

    aggregate_display = artifacts["aggregate_display"]
    per_fold_display = artifacts["per_fold_display"]
    calibration_display = artifacts["calibration_display"]

    model_families = _validation_filter_options(aggregate_display, "model_family")
    scopes = _validation_filter_options(aggregate_display, "scope")
    fold_years = _validation_filter_options(per_fold_display, "fold_year")
    with render_filter_bar("Validation Filters"):
        selected_models = st.multiselect(
            "Model family",
            model_families,
            default=model_families,
            key="multi_fold_model_family",
        )
        selected_scopes = st.multiselect(
            "Training scope",
            scopes,
            default=scopes,
            key="multi_fold_training_scope",
        )
        selected_folds = st.multiselect(
            "Fold year",
            fold_years,
            default=fold_years,
            key="multi_fold_year",
        )

    render_validation_findings_cards(artifacts["headline_findings"])

    aggregate_filtered = _filter_validation_frame(
        aggregate_display,
        model_families=selected_models,
        scopes=selected_scopes,
    )
    aggregate_columns = [
        "model",
        "scope",
        "log_loss mean+/-std",
        "brier mean+/-std",
        "top1_acc mean+/-std",
        "champion_hits/3",
    ]
    st.markdown("**Aggregate Validation**")
    st.dataframe(
        aggregate_filtered.loc[:, [column for column in aggregate_columns if column in aggregate_filtered.columns]],
        width="stretch",
        hide_index=True,
    )

    per_fold_filtered = _filter_validation_frame(
        per_fold_display,
        model_families=selected_models,
        scopes=selected_scopes,
        fold_years=selected_folds,
    )
    per_fold_columns = [
        "fold_year",
        "model",
        "scope",
        "log_loss",
        "brier",
        "top1_acc",
        "draw_pred",
        "draw_actual",
        "r16_hits",
        "sf_hits",
        "champion_hit",
    ]
    st.markdown("**Per-Fold Results**")
    st.dataframe(
        per_fold_filtered.loc[:, [column for column in per_fold_columns if column in per_fold_filtered.columns]],
        width="stretch",
        hide_index=True,
    )

    calibration_filtered = _filter_validation_frame(
        calibration_display,
        model_families=selected_models,
    )
    st.markdown("**Calibration ECE**")
    st.dataframe(
        calibration_filtered.loc[
            :,
            [column for column in ["model", "home_win_ece", "draw_ece", "away_win_ece"] if column in calibration_filtered.columns],
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Anomaly Flags**")
    anomaly_notes = list(artifacts["anomaly_notes"])
    if anomaly_notes:
        st.dataframe(pd.DataFrame({"flag": anomaly_notes}), width="stretch", hide_index=True)
    else:
        st.caption("No anomaly flags were generated from the current validation thresholds.")


def render_home_page() -> None:
    """Render the landing page for the grouped dashboard pages."""
    inject_styles()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    base_df, fixtures_df, lead_in_df, metadata = load_data()
    render_dashboard_header(world_cup_logo_data_uri, metadata, SIMULATION_COUNT, title="World Cup 2026 Projections Dashboard")
    render_countdown_timer(fixtures_df)

    st.markdown(
        """
        <div class="wc-home-intro">
          <div>
            <div class="wc-home-intro-kicker">What this project is</div>
            <h2 class="wc-home-intro-title">A data-driven 2026 FIFA Men's World Cup research hub</h2>
            <p class="wc-home-intro-copy">
              This app brings together three views of the tournament: historical World Cup analysis, qualification
              and team context, and simulation-based 2026 projections. Use the analysis pages to understand past
              tournament patterns and qualification performance, the team report cards to inspect one country at a
              time, and the model pages to compare how different methods project the group stage and knockout path.
            </p>
          </div>
          <div class="wc-home-intro-panel">
            <div class="wc-home-intro-panel-title">How to use it</div>
            <div class="wc-home-intro-panel-value">Analyze, inspect, project</div>
            <div class="wc-home-intro-panel-copy">Start with the section that matches your question: history and qualifiers, one-team reports, or model projections.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    home_settings = default_simulation_settings()
    simulation_count = SIMULATION_OPTIONS[str(home_settings["simulation_label"])]
    form_match_window = int(home_settings["form_match_window"])
    training_scope = PRIMARY_MODEL.default_training_scope
    artifact_settings = ArtifactSettings(
        model_id=PRIMARY_MODEL.model_id,
        model_version=PRIMARY_MODEL.model_version,
        data_build_date=str(metadata.get("build_date", "")),
        simulations=simulation_count,
        match_window=form_match_window,
        training_scope=training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    load_result = load_or_run_probability_artifact(
        artifact_settings,
        lambda: build_v4_probability_artifact(
            base_df,
            fixtures_df,
            lead_in_df,
            simulations=simulation_count,
            match_window=form_match_window,
            training_scope=training_scope,
            seed=DEFAULT_SIMULATION_SEED,
            bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        ),
        spinner_label=f"Building V4 projection overview from {simulation_count:,} simulations...",
        force_refresh=False,
    )
    display_artifact_status(load_result, "V4 primary")
    if load_result.artifact is None:
        st.error("Primary V4 probability artifact could not be loaded or created.")
        return
    artifact_scope = str(load_result.artifact.metadata.get("training_scope", training_scope))
    artifact_scope_display = TRAINING_SCOPE_LABEL_BY_VALUE.get(artifact_scope, artifact_scope)
    dashboard_df = load_result.artifact.dashboard_df
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    dashboard_df["top2_prob"] = (
        pd.to_numeric(dashboard_df["prob_1"], errors="coerce").fillna(0.0)
        + pd.to_numeric(dashboard_df["prob_2"], errors="coerce").fillna(0.0)
    )

    champion_name, champion_detail = team_value_detail(first_team_by_metric(dashboard_df, "champion_prob"), "champion_prob")
    finalist_name, finalist_detail = team_value_detail(first_team_by_metric(dashboard_df, "final_prob"), "final_prob")
    group_name, group_detail = team_value_detail(first_team_by_metric(dashboard_df, "top2_prob"), "top2_prob")
    knockout_name, knockout_detail = team_value_detail(first_team_by_metric(dashboard_df, "ko_prob"), "ko_prob")
    first_kickoff = get_first_kickoff_details(fixtures_df)

    st.markdown(
        f"""
        <div class="wc-home-section">
          <div class="wc-home-section-head">
            <div>
              <h2 class="wc-home-section-title">Projection Snapshot</h2>
              <p class="wc-home-section-note">Start with V4 for the current pre-tournament projection. This snapshot trains on historical World Cup finals only; use the top navigation to move into team reports, model tables, and backtests.</p>
            </div>
            <span class="wc-home-badge">V4 Primary | {artifact_scope_display}</span>
          </div>
          <div class="wc-home-metric-grid">
            {build_home_metric_card("Favorite to Win", champion_name, champion_detail)}
            {build_home_metric_card("Most Likely Finalist", finalist_name, finalist_detail)}
            {build_home_metric_card("Best Group Outlook", group_name, group_detail)}
            {build_home_metric_card("Most Likely KO Team", knockout_name, knockout_detail)}
            {build_home_metric_card("Opening Match", first_kickoff["match_label"], f'{first_kickoff["kickoff_date_label"]} at {first_kickoff["kickoff_utc_time_label"]} UTC')}
            {build_home_metric_card("Projection Run", f'{simulation_count:,} simulations', f'Model {PRIMARY_MODEL.model_version} | last {form_match_window} matches | {artifact_scope_display} training')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="wc-home-section">
          <div class="wc-home-section-head">
            <div>
              <h2 class="wc-home-section-title">Where To Go Next</h2>
              <p class="wc-home-section-note">Use the grouped top navigation to choose between historical analysis, qualification context, team reports, projection models, and backtests.</p>
            </div>
          </div>
          <div class="wc-home-route-grid">
            {build_home_route_card("Team Report Card", "Reports > Team Report Card", "Deep dive into one country with history, qualification path, squad context, and projection outlook.")}
            {build_home_route_card("V4 Enhanced Poisson", "Models > V4 Enhanced Poisson", "Start here for primary 2026 probabilities, group tables, and all-country rankings.")}
            {build_home_route_card("V2 Probabilities", "Legacy > V2 Probabilities", "Compare the current projection against the alternate form-weighted multinomial model.")}
            {build_home_route_card("V2 Form", "Legacy > V2 Form", "Inspect recent form inputs, team strength components, and confederation-level form tables.")}
            {build_home_route_card("Analysis", "Reports > Analysis", "Explore historical World Cup patterns behind participation, scoring, hosts, winners, and qualifiers.")}
            {build_home_route_card("Backtests", "Backtests", "Review V4 multi-fold validation and 2022 drilldowns before trusting current projections.")}
          </div>
        </div>
        <div class="wc-home-section">
          <div class="wc-home-section-head">
            <div>
              <h2 class="wc-home-section-title">Model Guide</h2>
              <p class="wc-home-section-note">Each model page keeps its own settings and exports, so comparisons do not interfere with each other.</p>
            </div>
          </div>
          <div class="wc-home-model-grid">
            {build_home_model_card(MODEL_VERSION, "V1 Team Strength", MODEL_SUMMARY)}
            {build_home_model_card(V2_MODEL_VERSION, "V2 Form and Probabilities", V2_MODEL_SUMMARY)}
            {build_home_model_card(V3_MODEL_VERSION, "V3 Poisson Regression", V3_MODEL_SUMMARY)}
            {build_home_model_card(V4_MODEL_VERSION, "V4 Enhanced Poisson", V4_MODEL_SUMMARY, recommended=True)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Method notes"):
        st.markdown(
            f"""
            - **ELO Rating:** numerical measure of a team's relative strength, updated after each match. Higher ratings indicate stronger teams.
            - **ELO Change/Delta:** points gained or lost after a match based on result, expected result from rating difference, and margin of victory.
            - **Training scope:** `{artifact_scope}` uses historical World Cup finals matches for the home projection snapshot.
            - **Simulation count:** the home snapshot uses the default V4 setting of `{simulation_count:,}` tournament simulations.
            - **Interpretation:** projections are pre-tournament probabilities from model simulations, not guarantees.
            """
        )


def render_analysis_page() -> None:
    """Render the historical analysis page from the grouped navigation router."""
    from apps import historical_eda

    historical_eda.render_historical_eda_page()


def render_team_report_card_navigation_page() -> None:
    """Render the team report card page from the grouped navigation router."""
    from apps import team_report_card

    team_report_card.render_team_report_card_page()


def render_v1_navigation_page() -> None:
    """Render the V1 model page from the grouped navigation router."""
    render_v1_dashboard()


def render_v2_form_navigation_page() -> None:
    """Render the V2 form model page from the grouped navigation router."""
    render_v2_dashboard()


def render_v2_probabilities_navigation_page() -> None:
    """Render the V2 probability model page from the grouped navigation router."""
    render_v2_probabilities_dashboard()


def render_v3_probabilities_navigation_page() -> None:
    """Render the V3 probability model page from the grouped navigation router."""
    render_v3_probabilities_dashboard()


def render_v4_probabilities_navigation_page() -> None:
    """Render the V4 probability model page from the grouped navigation router."""
    render_v4_probabilities_dashboard()


def render_v2_backtest_navigation_page() -> None:
    """Render the V2 2022 backtest page from the grouped navigation router."""
    render_v2_2022_backtest_dashboard()


def render_v3_backtest_navigation_page() -> None:
    """Render the V3 2022 backtest page from the grouped navigation router."""
    render_v3_2022_backtest_dashboard()


def render_v4_backtest_navigation_page() -> None:
    """Render the V4 2022 backtest page from the grouped navigation router."""
    render_v4_2022_backtest_dashboard()


def render_v4_rolling_backtest_navigation_page() -> None:
    """Render the V4 rolling backtest page from the grouped navigation router."""
    render_v4_rolling_backtest_dashboard()
