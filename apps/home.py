import html
import base64
from datetime import datetime
import importlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import world_cup_simulation as simulation


simulation = importlib.reload(simulation)
MODEL_LABEL = simulation.MODEL_LABEL
MODEL_SUMMARY = simulation.MODEL_SUMMARY
MODEL_VERSION = simulation.MODEL_VERSION
V2_MODEL_LABEL = simulation.V2_MODEL_LABEL
V2_MODEL_SUMMARY = simulation.V2_MODEL_SUMMARY
V2_MODEL_VERSION = simulation.V2_MODEL_VERSION
build_deterministic_bracket = simulation.build_deterministic_bracket
build_deterministic_bracket_v2 = simulation.build_deterministic_bracket_v2
build_v2_team_strengths = simulation.build_v2_team_strengths
build_v2_match_feature_table = simulation.build_v2_match_feature_table
build_weighted_form_table = simulation.build_weighted_form_table
fit_v2_match_multinomial_model = simulation.fit_v2_match_multinomial_model
FORM_SCHEDULE_DIFFICULTY_NEUTRAL = simulation.FORM_SCHEDULE_DIFFICULTY_NEUTRAL
get_modal_group_rankings = simulation.get_modal_group_rankings
simulate_group_probabilities = simulation.simulate_group_probabilities
simulate_group_probabilities_v2 = simulation.simulate_group_probabilities_v2
WEIGHTED_FORM_COMPOSITE_WEIGHTS = simulation.WEIGHTED_FORM_COMPOSITE_WEIGHTS

DATA_DIR = ROOT / "INT-World Cup" / "world_cup" / "2026"
EXPORT_DIR = ROOT / "assets" / "charts" / "generated"
WORLD_CUP_LOGO_PATH = ROOT / "assets" / "logos" / "world-cup" / "fifa-world-cup-2026.football.cc.svg"
CHAMPION_TROPHY_PATH = ROOT / "assets" / "logos" / "world-cup" / "Coupe-du-monde.svg"
SIMULATION_COUNT = 20000
SIMULATION_OPTIONS = {
    "250": 250,
    "500": 500,
    "1k": 1000,
    "5k": 5000,
    "10k": 10000,
    "20k": 20000,
    "100k": 100000,
}
DEFAULT_RECENT_MATCH_WINDOW = 10
DEFAULT_SIMULATION_LABEL = "100k"
GROUP_ORDER = list("ABCDEFGHIJKL")
VIEW_OPTIONS = ("Single group", "All groups", "All Countries", "Form", "Bracket")
SCREENSHOT_CHANNELS = ("chrome", "msedge")
CURRENT_HOLDER_TEAM_ID = "ARG"
BRACKET_HEAD_TO_HEAD_SIMULATIONS = 1000
BRACKET_EXPORT_VIEWPORT_SIZE = "1800,1200"
EXPORT_VIEWPORT_HEIGHT = 1400
EXPORT_MIN_VIEWPORT_WIDTH = 1400
EXPORT_MAX_VIEWPORT_WIDTH = 3200
FORM_WINDOW_MIN = 3
FORM_WINDOW_MAX = 20
FORM_CONFEDERATION_ORDER = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")
V1_VIEW_OPTIONS = ("Single group", "All groups", "All Countries", "Bracket")
V2_VIEW_OPTIONS = ("All Countries", "Single confederation", "All confederations")
V2_PROB_VIEW_OPTIONS = ("Single group", "All groups", "All Countries", "Bracket")
V1_STATE_KEY = "simulation_settings_v1"
V2_STATE_KEY = "simulation_settings_v2"
V2_PROB_STATE_KEY = "simulation_settings_v2_prob"
PROBABILITY_PALETTES = {
    "prob_1": ((220, 252, 231), (22, 163, 74)),
    "prob_2": ((219, 234, 254), (37, 99, 235)),
    "prob_3": ((254, 243, 199), (217, 119, 6)),
    "prob_4": ((254, 226, 226), (220, 38, 38)),
    "top8_third_prob": ((250, 245, 200), (202, 138, 4)),
    "ko_prob": ((224, 242, 254), (8, 145, 178)),
    "r16_prob": ((224, 231, 255), (79, 70, 229)),
    "qf_prob": ((233, 213, 255), (147, 51, 234)),
    "sf_prob": ((255, 228, 230), (225, 29, 72)),
    "final_prob": ((255, 237, 213), (234, 88, 12)),
    "champion_prob": ((254, 240, 138), (202, 138, 4)),
}
FORM_RED_TEXT = "#791F1F"
FORM_AMBER_TEXT = "#633806"
FORM_GREEN_TEXT = "#173404"
FORM_RED_GRADIENT = ("#FCEBEB", "#F7C1C1", "#F09595", "#E24B4A", "#A32D2D")
FORM_AMBER_GRADIENT = ("#FAEEDA", "#FAC775", "#EF9F27", "#BA7517", "#854F0B")
FORM_GREEN_GRADIENT = ("#EAF3DE", "#C0DD97", "#97C459", "#639922", "#3B6D11")
ALL_COUNTRIES_KNOCKOUT_COLUMNS = (
    ("top8_third_prob", "Top 8 3rd %"),
    ("ko_prob", "KO %"),
    ("r16_prob", "R16 %"),
    ("qf_prob", "QF %"),
    ("sf_prob", "SF %"),
    ("final_prob", "Final %"),
    ("champion_prob", "Champion %"),
)


def fix_mojibake(value: str) -> str:
    """Repair common UTF-8 decoding artifacts in source text fields."""
    if not isinstance(value, str):
        return value
    if all(marker not in value for marker in ("\u00c3", "\u00c2")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


@st.cache_data(show_spinner=False)
def load_svg_data_uri(svg_path: str) -> str:
    """Load a local SVG file as a data URI for inline display and export."""
    svg_bytes = Path(svg_path).read_bytes()
    encoded = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_world_cup_logo_data_uri() -> str:
    """Load the dashboard World Cup logo as a data URI."""
    return load_svg_data_uri(str(WORLD_CUP_LOGO_PATH))


@st.cache_data(show_spinner=False)
def load_champion_trophy_data_uri() -> str:
    """Load the champion trophy SVG as a data URI."""
    return load_svg_data_uri(str(CHAMPION_TROPHY_PATH))


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load the dashboard inputs: teams, ratings, fixtures, lead-in form, and metadata."""
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    groups = pd.read_csv(DATA_DIR / "groups.csv")
    fifa = pd.read_csv(DATA_DIR / "fifa_rank_snapshots.csv")
    elo = pd.read_csv(DATA_DIR / "elo_snapshots.csv")
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    lead_in = pd.read_csv(DATA_DIR / "team_results_lead_in.csv")
    manifest = pd.read_json(DATA_DIR / "manifest.json", typ="series").to_dict()

    text_columns = ["canonical_name", "tournament_name"]
    for frame in (teams, groups, fifa, elo, fixtures, lead_in):
        for column in text_columns:
            if column in frame.columns:
                frame[column] = frame[column].map(fix_mojibake)
    groups["team_name"] = groups["team_name"].map(fix_mojibake)
    if "qualified_team_name" in lead_in.columns:
        lead_in["qualified_team_name"] = lead_in["qualified_team_name"].map(fix_mojibake)
    if "opponent_name" in lead_in.columns:
        lead_in["opponent_name"] = lead_in["opponent_name"].map(fix_mojibake)

    latest_fifa = (
        fifa.sort_values(["snapshot_date", "source_as_of"])
        .drop_duplicates(subset=["team_id"], keep="last")
        .loc[:, ["team_id", "rank", "points", "snapshot_date"]]
        .rename(columns={"rank": "world_rank", "points": "fifa_points", "snapshot_date": "fifa_snapshot_date"})
    )
    latest_elo = (
        elo.sort_values(["snapshot_date", "source_as_of"])
        .drop_duplicates(subset=["team_id"], keep="last")
        .loc[:, ["team_id", "elo_rank", "elo_rating", "snapshot_date"]]
        .rename(columns={"snapshot_date": "elo_snapshot_date"})
    )

    team_columns = [
        "team_id",
        "tournament_name",
        "canonical_name",
        "flag_icon_code",
        "group_code",
        "confederation",
        "world_cup_participations",
        "weighted_world_cup_participations",
        "weighted_world_cup_placement_score",
    ]
    available_team_columns = [column_name for column_name in team_columns if column_name in teams.columns]

    merged = (
        groups.merge(
            teams.loc[:, available_team_columns],
            on=["team_id", "group_code"],
            how="left",
        )
        .merge(latest_fifa, on="team_id", how="left")
        .merge(latest_elo, on="team_id", how="left")
    )

    merged["display_name"] = merged["tournament_name"].fillna(merged["team_name"]).map(fix_mojibake)
    merged["world_rank"] = pd.to_numeric(merged["world_rank"], errors="coerce")
    merged["fifa_points"] = pd.to_numeric(merged["fifa_points"], errors="coerce")
    merged["elo_rating"] = pd.to_numeric(merged["elo_rating"], errors="coerce")
    merged["elo_rank"] = pd.to_numeric(merged["elo_rank"], errors="coerce")
    if "world_cup_participations" in merged.columns:
        merged["world_cup_participations"] = pd.to_numeric(merged["world_cup_participations"], errors="coerce")
    if "weighted_world_cup_participations" in merged.columns:
        merged["weighted_world_cup_participations"] = pd.to_numeric(
            merged["weighted_world_cup_participations"],
            errors="coerce",
        )
    if "weighted_world_cup_placement_score" in merged.columns:
        merged["weighted_world_cup_placement_score"] = pd.to_numeric(
            merged["weighted_world_cup_placement_score"],
            errors="coerce",
        )

    metadata = {
        "build_date": manifest.get("build_date", ""),
        "fifa_snapshot_date": latest_fifa["fifa_snapshot_date"].dropna().max(),
        "elo_snapshot_date": latest_elo["elo_snapshot_date"].dropna().max(),
    }
    return merged, fixtures, lead_in, metadata


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
def load_v2_match_model(form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW) -> dict[str, object]:
    """Fit and cache the v2 multinomial model artifacts for the active form window."""
    return fit_v2_match_multinomial_model(match_window=form_match_window)


@st.cache_data(show_spinner=False)
def simulate_probabilities_v2_dashboard(
    base_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    lead_in_df: pd.DataFrame,
    simulations: int = SIMULATION_COUNT,
    match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> pd.DataFrame:
    """Estimate tournament probabilities from the v2 multinomial simulator."""
    return simulate_group_probabilities_v2(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
    )


def ensure_dashboard_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee probability columns expected by the dashboard exist on the dataframe."""
    normalized = df.copy()
    for column_name in ("prob_1", "prob_2", "prob_3", "prob_4"):
        if column_name not in normalized.columns:
            normalized[column_name] = 0.0
    for column_name in ("top8_third_prob", "r16_prob", "qf_prob", "sf_prob", "final_prob", "champion_prob"):
        if column_name not in normalized.columns:
            normalized[column_name] = 0.0
    if "ko_prob" not in normalized.columns:
        normalized["ko_prob"] = (
            normalized["prob_1"].fillna(0.0)
            + normalized["prob_2"].fillna(0.0)
            + normalized["top8_third_prob"].fillna(0.0)
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


def chart_subtitle(base_label: str, simulation_count: int | None = None) -> str:
    """Return a chart subtitle with an optional simulation-count suffix."""
    if simulation_count is None:
        return base_label
    return f"{base_label} | {simulation_count:,} simulations"


def configure_page(page_title: str) -> None:
    """Configure the Streamlit page once per entrypoint."""
    st.set_page_config(page_title=page_title, layout="wide")


def render_dashboard_header(
    world_cup_logo_data_uri: str,
    metadata: dict[str, str],
    simulation_count: int,
    title: str = "World Cup 2026 Group Dashboard",
    model_version: str = MODEL_VERSION,
    model_label: str = MODEL_LABEL,
) -> None:
    """Render the shared dashboard header."""
    st.markdown(
        f"""
        <div class="wc-header">
          <div class="wc-header-bar">
            <img class="wc-title-logo" src="{world_cup_logo_data_uri}" alt="FIFA World Cup 2026 logo" />
            <div>
              <div class="wc-kicker">Pre-Tournament Predictions</div>
              <h1 style="margin:0;">{html.escape(title)}</h1>
              <div class="wc-meta">
                Model: {html.escape(model_version)} ({html.escape(model_label)}) |
                Build date: {html.escape(str(metadata["build_date"]))} |
                FIFA snapshot: {html.escape(str(metadata["fifa_snapshot_date"]))} |
                Elo snapshot: {html.escape(str(metadata["elo_snapshot_date"]))} |
                Simulations per group: {simulation_count:,}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def shared_css() -> str:
    """Return the shared CSS used by both Streamlit rendering and exported HTML files."""
    return """
    @import url('https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/css/flag-icons.min.css');

    body {
        margin: 0;
        background: #eef3f8;
        color: #0f172a;
        font-family: "Segoe UI", Tahoma, sans-serif;
    }
    html.wc-export-mode,
    body.wc-export-mode {
        width: max-content;
        min-width: 100%;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .wc-export-page {
        padding: 24px;
    }
    .wc-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(280px, 1fr));
        gap: 16px;
        align-items: start;
    }
    .wc-grid-single {
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
    }
    .wc-export-mode .wc-grid-single {
        width: max-content;
    }
    .wc-export-mode .wc-grid-single .wc-card,
    .wc-export-mode .wc-grid-single .wc-table-wrap {
        width: max-content;
        max-width: none;
    }
    .wc-export-mode .wc-grid-single .wc-table-wrap {
        overflow: visible;
    }
    .wc-export-mode .wc-grid-single table.wc-table {
        width: max-content;
        min-width: 0;
    }
    .wc-card {
        border: 1px solid #dfe5ec;
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
        overflow: hidden;
        margin: 0.55rem 0 0.85rem;
        padding-bottom: 0.6rem;
    }
    .wc-export-mode .wc-card {
        overflow: visible;
    }
    .wc-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 14px 16px 10px;
    }
    .wc-card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
    }
    .wc-card-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .wc-group-pill {
        min-width: 2.1rem;
        height: 2.1rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #0f172a;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .wc-table-wrap {
        width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
        -webkit-overflow-scrolling: touch;
        padding-bottom: 0.15rem;
    }
    table.wc-table {
        width: 100%;
        min-width: 720px;
        border-collapse: collapse;
        table-layout: auto;
    }
    .wc-table thead th {
        background: #0f172a;
        color: #ffffff;
        font-size: clamp(0.68rem, 0.64rem + 0.18vw, 0.76rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.78rem 0.65rem;
        text-align: left;
        white-space: nowrap;
    }
    .wc-table thead th.wc-num,
    .wc-table tbody td.wc-num {
        text-align: right;
        white-space: nowrap;
    }
    .wc-table thead th.wc-group-col,
    .wc-table tbody td.wc-group-col {
        text-align: center;
        width: 56px;
    }
    .wc-table tbody td {
        border-bottom: 1px solid #e8eef5;
        padding: 0.72rem 0.65rem;
        color: #0f172a;
        font-size: clamp(0.82rem, 0.78rem + 0.2vw, 0.93rem);
        vertical-align: middle;
        background-color: rgba(255, 255, 255, 0.82);
        overflow-wrap: anywhere;
    }
    .wc-table tbody tr:last-child td {
        border-bottom: none;
    }
    .wc-name-cell {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.62rem;
        font-weight: 600;
        min-width: 0;
        width: 100%;
    }
    .wc-name-main {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        min-width: 0;
        flex: 1 1 auto;
    }
    .wc-name-cell .fi {
        font-size: 1.18rem;
        border-radius: 999px;
        box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
        flex: 0 0 auto;
    }
    .wc-name-text {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: normal;
        line-height: 1.22;
    }
    .wc-qual-marker {
        position: relative;
        flex: 0 0 auto;
        width: 0.28rem;
        height: 1.75rem;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(148, 163, 184, 0.16);
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.12);
    }
    .wc-qual-segment {
        position: absolute;
        left: 0;
        right: 0;
    }
    .wc-qual-segment-top2 {
        bottom: 0;
        background: linear-gradient(180deg, #22c55e 0%, #15803d 100%);
    }
    .wc-qual-segment-third {
        background: linear-gradient(180deg, #fb923c 0%, #ea580c 100%);
    }
    .wc-holder-cell {
        background: linear-gradient(180deg, #fef3c7 0%, #fde68a 100%);
        color: #7c4a03;
    }
    .wc-holder-cell .wc-name-text {
        color: #7c4a03;
        font-weight: 800;
    }
    .wc-holder-cell .fi {
        box-shadow: inset 0 0 0 1px rgba(124, 74, 3, 0.18), 0 0 0 2px rgba(251, 191, 36, 0.22);
    }
    .wc-prob {
        font-variant-numeric: tabular-nums;
        font-weight: 700;
    }
    .wc-kicker {
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 0.35rem;
    }
    .wc-grid .wc-table-wrap {
        overflow-x: visible;
    }
    .wc-grid table.wc-table {
        min-width: 0;
        table-layout: fixed;
    }
    .wc-grid .wc-table thead th {
        font-size: 0.62rem;
        padding: 0.56rem 0.34rem;
    }
    .wc-grid .wc-table tbody td {
        font-size: 0.78rem;
        padding: 0.56rem 0.34rem;
    }
    .wc-grid .wc-name-cell {
        justify-content: center;
        gap: 0;
    }
    .wc-grid .wc-name-main {
        justify-content: center;
        gap: 0;
        flex: 0 0 auto;
    }
    .wc-grid .wc-name-cell .fi {
        font-size: 1.05rem;
    }
    .wc-grid .wc-name-text {
        display: none;
    }
    .wc-grid .wc-qual-marker {
        margin-left: 0.24rem;
        height: 1.45rem;
    }
    .wc-header {
        margin-bottom: 1.2rem;
    }
    .wc-header-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .wc-title-logo {
        width: 72px;
        height: 72px;
        object-fit: contain;
        flex: 0 0 auto;
    }
    .wc-meta {
        color: #475569;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }
    .wc-header-icon-label {
        display: inline-flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.3rem;
    }
    .wc-header-icon {
        width: 0.92rem;
        height: 0.92rem;
        object-fit: contain;
        vertical-align: middle;
    }
    .wc-bracket-board {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(190px, 230px) minmax(0, 1fr);
        gap: 14px;
        align-items: center;
        margin-top: 0.65rem;
    }
    .wc-bracket-side {
        display: grid;
        grid-template-columns: repeat(4, minmax(128px, 168px));
        justify-content: space-between;
        gap: 10px;
        align-items: center;
    }
    .wc-bracket-round {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .wc-bracket-round-left-r32,
    .wc-bracket-round-right-r32 {
        padding-top: 0;
    }
    .wc-bracket-round-left-r16,
    .wc-bracket-round-right-r16 {
        padding-top: 2.35rem;
    }
    .wc-bracket-round-left-qf,
    .wc-bracket-round-right-qf {
        padding-top: 4.7rem;
    }
    .wc-bracket-round-left-sf,
    .wc-bracket-round-right-sf {
        padding-top: 7.05rem;
    }
    .wc-bracket-round-title {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0f172a;
        padding: 0.2rem 0.1rem;
    }
    .wc-bracket-side-right .wc-bracket-round-title {
        text-align: right;
    }
    .wc-bracket-final-column {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.85rem;
        min-height: 100%;
    }
    .wc-bracket-final-title {
        font-size: 0.9rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0f172a;
    }
    .wc-bracket-match {
        border: 1px solid #dfe5ec;
        border-radius: 16px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.05);
        padding: 0.65rem 0.7rem;
    }
    .wc-bracket-final-column .wc-bracket-match {
        width: 100%;
        max-width: 220px;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
    }
    .wc-bracket-match-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        margin-bottom: 0.45rem;
    }
    .wc-bracket-side-right .wc-bracket-match-head {
        flex-direction: row-reverse;
    }
    .wc-bracket-match-number {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .wc-bracket-match-prob {
        font-size: 0.76rem;
        font-weight: 800;
        color: #0f766e;
        background: rgba(13, 148, 136, 0.1);
        border-radius: 999px;
        padding: 0.18rem 0.45rem;
        white-space: nowrap;
    }
    .wc-bracket-teams {
        display: flex;
        flex-direction: column;
        gap: 0.42rem;
    }
    .wc-bracket-team {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        border-radius: 12px;
        padding: 0.38rem 0.48rem;
        color: #0f172a;
        background: rgba(248, 250, 252, 0.9);
    }
    .wc-bracket-side-right .wc-bracket-team {
        flex-direction: row-reverse;
    }
    .wc-bracket-team-win {
        background: linear-gradient(180deg, #dcfce7 0%, #bbf7d0 100%);
        box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.18);
        font-weight: 700;
    }
    .wc-bracket-team .fi {
        font-size: 1rem;
        border-radius: 999px;
        flex: 0 0 auto;
    }
    .wc-bracket-team-name {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .wc-bracket-side-right .wc-bracket-team-name {
        text-align: right;
    }
    .wc-bracket-note {
        color: #475569;
        font-size: 0.9rem;
        margin: 0.35rem 0 0.2rem;
    }
    @media (max-width: 1380px) {
        .wc-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .wc-bracket-board {
            grid-template-columns: 1fr;
        }
        .wc-bracket-side {
            grid-template-columns: repeat(2, minmax(180px, 1fr));
        }
        .wc-bracket-round-left-r16,
        .wc-bracket-round-right-r16,
        .wc-bracket-round-left-qf,
        .wc-bracket-round-right-qf,
        .wc-bracket-round-left-sf,
        .wc-bracket-round-right-sf {
            padding-top: 0;
        }
    }
    @media (max-width: 860px) {
        .wc-grid {
            grid-template-columns: 1fr;
        }
        .wc-bracket-side {
            grid-template-columns: 1fr;
        }
        .wc-bracket-side-right .wc-bracket-round-title {
            text-align: left;
        }
        .wc-bracket-side-right .wc-bracket-match-head,
        .wc-bracket-side-right .wc-bracket-team {
            flex-direction: row;
        }
        .wc-bracket-side-right .wc-bracket-team-name {
            text-align: left;
        }
    }
    @media (max-width: 760px) {
        .wc-export-page {
            padding: 12px;
        }
        .wc-header-bar {
            align-items: flex-start;
        }
        .wc-title-logo {
            width: 56px;
            height: 56px;
        }
        .wc-table thead th,
        .wc-table tbody td {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        table.wc-table {
            min-width: 560px;
        }
        .wc-name-cell {
            justify-content: center;
            gap: 0;
        }
        .wc-name-main {
            justify-content: center;
            gap: 0;
        }
        .wc-name-cell .fi {
            font-size: 1.28rem;
        }
        .wc-name-text {
            display: none;
        }
        .wc-qual-marker {
            margin-left: 0.24rem;
            height: 1.5rem;
        }
        .wc-bracket-side {
            grid-template-columns: 1fr;
        }
    }
    """


def inject_styles() -> None:
    """Inject the dashboard CSS and flag-icons stylesheet into the Streamlit page."""
    st.markdown(f"<style>{shared_css()}</style>", unsafe_allow_html=True)


def get_first_kickoff_details(fixtures_df: pd.DataFrame) -> dict[str, str]:
    """Return the earliest scheduled group-stage fixture and its formatted kickoff strings."""
    fixtures = fixtures_df.copy()
    fixtures["match_number"] = pd.to_numeric(fixtures["match_number"], errors="coerce")
    fixtures["kickoff_datetime_utc"] = pd.to_datetime(fixtures["kickoff_datetime_utc"], errors="coerce", utc=True)
    first_fixture = (
        fixtures[(fixtures["round_code"] == "GS") & fixtures["kickoff_datetime_utc"].notna()]
        .sort_values(["kickoff_datetime_utc", "match_number"], kind="stable")
        .iloc[0]
    )
    kickoff_utc = first_fixture["kickoff_datetime_utc"]
    kickoff_local_raw = str(first_fixture.get("kickoff_datetime_local", "")).strip()
    local_time_label = kickoff_local_raw[11:16] if len(kickoff_local_raw) >= 16 else kickoff_utc.strftime("%H:%M")
    return {
        "kickoff_iso_utc": kickoff_utc.isoformat().replace("+00:00", "Z"),
        "kickoff_date_label": kickoff_utc.strftime("%B-%d-%Y"),
        "kickoff_utc_time_label": kickoff_utc.strftime("%H:%M"),
        "kickoff_local_time_label": local_time_label,
        "match_label": f'{first_fixture["home_tournament_name"]} vs {first_fixture["away_tournament_name"]}',
    }


def build_countdown_html(kickoff_details: dict[str, str]) -> str:
    """Build the live countdown widget markup for the first World Cup kickoff."""
    kickoff_iso_utc = html.escape(kickoff_details["kickoff_iso_utc"])
    kickoff_date_label = html.escape(kickoff_details["kickoff_date_label"])
    kickoff_utc_time_label = html.escape(kickoff_details["kickoff_utc_time_label"])
    kickoff_local_time_label = html.escape(kickoff_details["kickoff_local_time_label"])
    match_label = html.escape(kickoff_details["match_label"])
    return f"""
    <div style="margin:0 0 0.9rem;">
      <div style="border:1px solid rgba(191,219,254,0.35);border-radius:22px;padding:22px 24px;background:
      radial-gradient(circle at top, rgba(96,165,250,0.18), transparent 45%),
      linear-gradient(135deg,#0f172a 0%,#172554 52%,#1e293b 100%);color:#f8fafc;box-shadow:0 16px 40px rgba(15,23,42,0.24);">
        <div style="text-align:center;">
          <div style="font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;color:#bfdbfe;margin-bottom:0.55rem;font-weight:700;">Countdown To Opening Kickoff</div>
          <div style="font-size:1.45rem;font-weight:800;line-height:1.2;margin-bottom:0.5rem;">{match_label}</div>
          <div id="wc-countdown-value" style="font-size:3.1rem;font-weight:900;line-height:1.02;letter-spacing:-0.04em;color:#ffffff;text-shadow:0 6px 24px rgba(147,197,253,0.22);margin:0.15rem 0 0.85rem;">Loading countdown...</div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:18px;margin-top:0.55rem;padding-top:0.9rem;border-top:1px solid rgba(191,219,254,0.18);">
          <div style="text-align:left;">
            <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;margin-bottom:0.2rem;">Date</div>
            <div style="font-size:1rem;font-weight:700;color:#eff6ff;">{kickoff_date_label}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;margin-bottom:0.2rem;">Time [local | UTC]</div>
            <div style="font-size:1rem;font-weight:700;color:#eff6ff;">{kickoff_local_time_label} | {kickoff_utc_time_label}</div>
          </div>
        </div>
      </div>
    </div>
    <script>
      const countdownNode = document.getElementById("wc-countdown-value");
      const kickoffTime = new Date("{kickoff_iso_utc}").getTime();

      function updateCountdown() {{
        const deltaMs = kickoffTime - Date.now();
        if (deltaMs <= 0) {{
          countdownNode.textContent = "Kickoff is live";
          return;
        }}

        const totalSeconds = Math.floor(deltaMs / 1000);
        const days = Math.floor(totalSeconds / 86400);
        const hours = Math.floor((totalSeconds % 86400) / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        countdownNode.textContent = `${{days}}d ${{hours}}h ${{minutes}}m ${{seconds}}s`;
      }}

      updateCountdown();
      window.setInterval(updateCountdown, 1000);
    </script>
    """


def render_countdown_timer(fixtures_df: pd.DataFrame) -> None:
    """Render a live countdown to the first scheduled group-stage kickoff."""
    kickoff_details = get_first_kickoff_details(fixtures_df)
    components.html(build_countdown_html(kickoff_details), height=225)


def format_percent(value: float) -> str:
    """Render a probability value as a one-decimal-place percentage string."""
    return f"{value:.1f}%"


def format_decimal(value: float, decimals: int = 1) -> str:
    """Render a numeric value with a fixed number of decimal places."""
    return f"{float(value):.{decimals}f}"


def probability_cell_style(column_name: str, value: float, column_min: float, column_max: float) -> str:
    """Build a column-relative heatmap fill for one probability cell."""
    light_rgb, dark_rgb = PROBABILITY_PALETTES[column_name]
    if column_max > column_min:
        intensity = (float(value) - column_min) / (column_max - column_min)
    else:
        intensity = 0.5

    red = round(light_rgb[0] + (dark_rgb[0] - light_rgb[0]) * intensity)
    green = round(light_rgb[1] + (dark_rgb[1] - light_rgb[1]) * intensity)
    blue = round(light_rgb[2] + (dark_rgb[2] - light_rgb[2]) * intensity)
    return f"background-color: rgb({red}, {green}, {blue});"


def form_cell_style(fill_color: str, text_color: str) -> str:
    """Build a consistent fill/text style for form-table cells."""
    return f"background-color: {fill_color}; color: {text_color};"


def interpolate_hex_color(start_hex: str, end_hex: str, weight: float) -> str:
    """Interpolate between two hex colors and return the blended hex value."""
    clamped_weight = max(0.0, min(1.0, float(weight)))
    start_rgb = tuple(int(start_hex[index:index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end_hex[index:index + 2], 16) for index in (1, 3, 5))
    blended = tuple(
        round(start_component + (end_component - start_component) * clamped_weight)
        for start_component, end_component in zip(start_rgb, end_rgb)
    )
    return "#{:02X}{:02X}{:02X}".format(*blended)


def gradient_fill_color(stops: tuple[str, ...], position: float) -> str:
    """Return an interpolated fill color along a multi-stop hex gradient."""
    if not stops:
        raise ValueError("stops must contain at least one color")
    if len(stops) == 1:
        return stops[0]

    clamped_position = max(0.0, min(1.0, float(position)))
    scaled_position = clamped_position * (len(stops) - 1)
    lower_index = min(len(stops) - 2, int(scaled_position))
    upper_index = lower_index + 1
    local_weight = scaled_position - lower_index
    return interpolate_hex_color(stops[lower_index], stops[upper_index], local_weight)


def sequential_form_cell_style(value: float, column_min: float, column_max: float) -> str:
    """Build a low-mid-high style for form columns without a neutral anchor."""
    if pd.isna(value):
        return ""
    if pd.isna(column_min) or pd.isna(column_max) or column_max <= column_min:
        return form_cell_style(gradient_fill_color(FORM_AMBER_GRADIENT, 0.5), FORM_AMBER_TEXT)

    normalized = max(0.0, min(1.0, (float(value) - column_min) / (column_max - column_min)))
    if normalized <= (1.0 / 3.0):
        tier_position = 1.0 - (normalized / (1.0 / 3.0))
        return form_cell_style(gradient_fill_color(FORM_RED_GRADIENT, tier_position), FORM_RED_TEXT)
    if normalized <= (2.0 / 3.0):
        tier_position = (normalized - (1.0 / 3.0)) / (1.0 / 3.0)
        return form_cell_style(gradient_fill_color(FORM_AMBER_GRADIENT, tier_position), FORM_AMBER_TEXT)
    tier_position = (normalized - (2.0 / 3.0)) / (1.0 / 3.0)
    return form_cell_style(gradient_fill_color(FORM_GREEN_GRADIENT, tier_position), FORM_GREEN_TEXT)


def diverging_form_cell_style(
    value: float,
    anchor: float,
    negative_span: float,
    positive_span: float,
    reverse: bool = False,
) -> str:
    """Build a red-amber-green diverging style centered on a meaningful anchor."""
    if pd.isna(value):
        return ""
    difference = float(value) - float(anchor)
    negative_text = FORM_GREEN_TEXT if reverse else FORM_RED_TEXT
    positive_text = FORM_RED_TEXT if reverse else FORM_GREEN_TEXT
    negative_gradient = FORM_GREEN_GRADIENT if reverse else FORM_RED_GRADIENT
    positive_gradient = FORM_RED_GRADIENT if reverse else FORM_GREEN_GRADIENT

    if abs(difference) < 1e-12:
        return form_cell_style(gradient_fill_color(FORM_AMBER_GRADIENT, 0.0), FORM_AMBER_TEXT)

    if difference < 0:
        if negative_span <= 0:
            return form_cell_style(gradient_fill_color(FORM_AMBER_GRADIENT, 0.0), FORM_AMBER_TEXT)
        normalized = min(1.0, abs(difference) / negative_span)
    else:
        if positive_span <= 0:
            return form_cell_style(gradient_fill_color(FORM_AMBER_GRADIENT, 0.0), FORM_AMBER_TEXT)
        normalized = min(1.0, difference / positive_span)

    if normalized < 0.5:
        tier_position = normalized / 0.5
        return form_cell_style(gradient_fill_color(FORM_AMBER_GRADIENT, tier_position), FORM_AMBER_TEXT)
    if difference < 0:
        tier_position = (normalized - 0.5) / 0.5
        return form_cell_style(gradient_fill_color(negative_gradient, tier_position), negative_text)
    tier_position = (normalized - 0.5) / 0.5
    return form_cell_style(gradient_fill_color(positive_gradient, tier_position), positive_text)


def current_holder_cell_class(team_id: str) -> str:
    """Return the cell class for the current World Cup holder."""
    return " wc-holder-cell" if team_id == CURRENT_HOLDER_TEAM_ID else ""


def render_group_qualification_marker(top2_prob: float, third_prob: float) -> str:
    """Render a compact vertical rail for top-two and best-third qualification chances."""
    top2_height = max(0.0, min(100.0, float(top2_prob)))
    third_height = max(0.0, min(100.0 - top2_height, float(third_prob)))
    segments = []
    if top2_height > 0:
        segments.append(
            f'<span class="wc-qual-segment wc-qual-segment-top2" style="height:{top2_height:.1f}%;"></span>'
        )
    if third_height > 0:
        segments.append(
            f'<span class="wc-qual-segment wc-qual-segment-third" style="bottom:{top2_height:.1f}%;height:{third_height:.1f}%;"></span>'
        )
    return f'<span class="wc-qual-marker" aria-hidden="true">{"".join(segments)}</span>'


def render_name_cell(
    flag_icon_code: str,
    display_name: str,
    show_group_qualification_marker: bool = False,
    top2_prob: float = 0.0,
    third_prob: float = 0.0,
) -> str:
    """Render the team name cell with a flag-icons badge when a code is available."""
    safe_name = html.escape(display_name)
    marker = render_group_qualification_marker(top2_prob, third_prob) if show_group_qualification_marker else ""
    if isinstance(flag_icon_code, str) and flag_icon_code:
        return (
            '<div class="wc-name-cell">'
            '<span class="wc-name-main">'
            f'<span class="fi fi-{html.escape(flag_icon_code)}"></span>'
            f'<span class="wc-name-text">{safe_name}</span>'
            "</span>"
            f"{marker}"
            "</div>"
        )
    return (
        '<div class="wc-name-cell">'
        f'<span class="wc-name-main"><span class="wc-name-text">{safe_name}</span></span>'
        f"{marker}"
        "</div>"
    )


def champion_column_header() -> str:
    """Render the Champion column header with the local trophy icon."""
    trophy_data_uri = load_champion_trophy_data_uri()
    return (
        '<span class="wc-header-icon-label">'
        f'<img class="wc-header-icon" src="{trophy_data_uri}" alt="Champion trophy" />'
        "<span>Champion %</span>"
        "</span>"
    )


def build_table_card_html(
    headers: list[str],
    body_rows: list[str],
    title: str,
    card_subtitle: str,
    group_pill_label: str | None = None,
) -> str:
    """Render a standard card wrapper around a table body."""
    group_pill = ""
    if group_pill_label is None and title.startswith("Group "):
        title_parts = title.split()
        if len(title_parts) >= 2:
            group_pill_label = title_parts[1]
    if group_pill_label:
        group_pill = f'<span class="wc-group-pill">{html.escape(group_pill_label)}</span>'
    card_title = html.escape(title)
    safe_card_subtitle = html.escape(card_subtitle)
    return textwrap.dedent(
        f"""
        <div class="wc-card">
          <div class="wc-card-header">
            <div>
              <div class="wc-card-subtitle">{safe_card_subtitle}</div>
              <div class="wc-card-title">{card_title}</div>
            </div>
            {group_pill}
          </div>
          <div class="wc-table-wrap">
            <table class="wc-table">
              <thead><tr>{''.join(headers)}</tr></thead>
              <tbody>{''.join(body_rows)}</tbody>
            </table>
          </div>
        </div>
        """
    ).strip()


def build_probability_table_html(
    df: pd.DataFrame,
    title: str,
    include_group_column: bool = False,
    include_ko_column: bool = False,
    card_subtitle: str = "Pre-Tournament Probability Table",
    group_pill_label: str | None = None,
) -> str:
    """Render one probability table as a styled HTML card."""
    df = ensure_dashboard_probability_columns(df)
    include_rank_column = include_group_column
    show_group_qualification_marker = not include_group_column and not include_ko_column
    probability_columns = ["prob_1", "prob_2", "prob_3", "prob_4"]
    if include_ko_column:
        probability_columns.extend(column_name for column_name, _ in ALL_COUNTRIES_KNOCKOUT_COLUMNS)
    column_ranges = {
        column_name: (float(df[column_name].min()), float(df[column_name].max()))
        for column_name in probability_columns
    }
    headers = []
    if include_group_column:
        headers.append('<th class="wc-group-col">Group</th>')
    headers.extend(
        [
            "<th>Rank</th>" if include_rank_column else "",
            "<th>Country</th>",
            '<th class="wc-num">World Rank</th>',
            '<th class="wc-num">Elo</th>',
            '<th class="wc-num">1st %</th>',
            '<th class="wc-num">2nd %</th>',
            '<th class="wc-num">3rd %</th>',
            '<th class="wc-num">4th %</th>',
        ]
    )
    headers = [header for header in headers if header]
    if include_ko_column:
        for column_name, label in ALL_COUNTRIES_KNOCKOUT_COLUMNS:
            if column_name == "champion_prob":
                headers.append(f'<th class="wc-num">{champion_column_header()}</th>')
            else:
                headers.append(f'<th class="wc-num">{html.escape(label)}</th>')

    body_rows = []
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        cells = []
        if include_group_column:
            cells.append(f'<td class="wc-group-col"><span class="wc-group-pill">{html.escape(str(row.group_code))}</span></td>')
        if include_rank_column:
            cells.append(f'<td class="wc-num">{rank}</td>')
        cells.extend(
            [
                (
                    f'<td class="{current_holder_cell_class(row.team_id).strip()}">'
                    f'{render_name_cell(row.flag_icon_code, row.display_name, show_group_qualification_marker=show_group_qualification_marker, top2_prob=row.prob_1 + row.prob_2, third_prob=row.top8_third_prob)}'
                    "</td>"
                ),
                f'<td class="wc-num">{int(row.world_rank)}</td>',
                f'<td class="wc-num">{int(row.elo_rating)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_1", row.prob_1, *column_ranges["prob_1"])}">{format_percent(row.prob_1)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_2", row.prob_2, *column_ranges["prob_2"])}">{format_percent(row.prob_2)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_3", row.prob_3, *column_ranges["prob_3"])}">{format_percent(row.prob_3)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_4", row.prob_4, *column_ranges["prob_4"])}">{format_percent(row.prob_4)}</td>',
            ]
        )
        if include_ko_column:
            for column_name, _ in ALL_COUNTRIES_KNOCKOUT_COLUMNS:
                column_value = getattr(row, column_name)
                cells.append(
                    f'<td class="wc-num wc-prob" style="{probability_cell_style(column_name, column_value, *column_ranges[column_name])}">{format_percent(column_value)}</td>'
                )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return build_table_card_html(headers, body_rows, title, card_subtitle, group_pill_label=group_pill_label)


def build_form_table_html(
    df: pd.DataFrame,
    title: str,
    card_subtitle: str = "Weighted Recent Form Table",
    group_pill_label: str | None = None,
) -> str:
    """Render the recent-form table as a styled HTML card."""
    has_history_columns = all(
        column_name in df.columns
        for column_name in (
            "weighted_world_cup_participations",
            "weighted_world_cup_placement_score",
            "history_score",
            "v2_strength",
        )
    )
    sequential_columns = ["results_form", "expected_score", "form"]
    if has_history_columns:
        sequential_columns.extend(
            [
                "weighted_world_cup_participations",
                "weighted_world_cup_placement_score",
                "history_score",
                "v2_strength",
            ]
        )
    sequential_ranges = {
        column_name: (
            float(numeric_values.min()) if not numeric_values.empty else float("nan"),
            float(numeric_values.max()) if not numeric_values.empty else float("nan"),
        )
        for column_name in sequential_columns
        for numeric_values in [pd.to_numeric(df[column_name], errors="coerce").dropna()]
    }
    gd_form_values = pd.to_numeric(df["gd_form"], errors="coerce").dropna()
    perf_vs_exp_values = pd.to_numeric(df["perf_vs_exp"], errors="coerce").dropna()
    sched_diff_values = pd.to_numeric(df["schedule_difficulty"], errors="coerce").dropna()
    gd_form_negative_span = float(abs(gd_form_values.min())) if not gd_form_values.empty and gd_form_values.min() < 0 else 0.0
    gd_form_positive_span = float(gd_form_values.max()) if not gd_form_values.empty and gd_form_values.max() > 0 else 0.0
    perf_negative_span = float(abs(perf_vs_exp_values.min())) if not perf_vs_exp_values.empty and perf_vs_exp_values.min() < 0 else 0.0
    perf_positive_span = float(perf_vs_exp_values.max()) if not perf_vs_exp_values.empty and perf_vs_exp_values.max() > 0 else 0.0
    sched_easy_span = (
        float(FORM_SCHEDULE_DIFFICULTY_NEUTRAL - sched_diff_values.min())
        if not sched_diff_values.empty and sched_diff_values.min() < FORM_SCHEDULE_DIFFICULTY_NEUTRAL
        else 0.0
    )
    sched_hard_span = (
        float(sched_diff_values.max() - FORM_SCHEDULE_DIFFICULTY_NEUTRAL)
        if not sched_diff_values.empty and sched_diff_values.max() > FORM_SCHEDULE_DIFFICULTY_NEUTRAL
        else 0.0
    )
    headers = [
        '<th class="wc-num">Rank</th>',
        "<th>Country</th>",
        "<th>Confederation</th>",
        '<th class="wc-num">W</th>',
        '<th class="wc-num">D</th>',
        '<th class="wc-num">L</th>',
        '<th class="wc-num">GS</th>',
        '<th class="wc-num">GA</th>',
        '<th class="wc-num">ELO</th>',
        '<th class="wc-num">OPP</th>',
        '<th class="wc-num">Avg Gap</th>',
        '<th class="wc-num">Sched Diff</th>',
        '<th class="wc-num">Results Form</th>',
        '<th class="wc-num">GD Form</th>',
        '<th class="wc-num">Exp</th>',
        '<th class="wc-num">Perf vs Exp</th>',
        '<th class="wc-num">Elo Delta Form</th>',
        '<th class="wc-num">Form</th>',
    ]
    if has_history_columns:
        headers.extend(
            [
                '<th class="wc-num">Wtd WC Apps</th>',
                '<th class="wc-num">Wtd WC Place</th>',
                '<th class="wc-num">History</th>',
                '<th class="wc-num">V2 Strength</th>',
            ]
        )

    body_rows = []
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        cells = [
            f'<td class="wc-num">{rank}</td>',
            (
                f'<td class="{current_holder_cell_class(row.team_id).strip()}">'
                f'{render_name_cell(row.flag_icon_code, row.display_name)}'
                "</td>"
            ),
            f'<td>{html.escape(str(row.confederation))}</td>',
            f'<td class="wc-num">{int(row.wins)}</td>',
            f'<td class="wc-num">{int(row.draws)}</td>',
            f'<td class="wc-num">{int(row.losses)}</td>',
            f'<td class="wc-num">{int(row.goals_for)}</td>',
            f'<td class="wc-num">{int(row.goals_against)}</td>',
            f'<td class="wc-num">{int(round(float(row.elo_rating)))}</td>',
            f'<td class="wc-num">{format_decimal(row.avg_opp_elo)}</td>',
            f'<td class="wc-num">{format_decimal(row.avg_elo_gap)}</td>',
            f'<td class="wc-num" style="{diverging_form_cell_style(row.schedule_difficulty, FORM_SCHEDULE_DIFFICULTY_NEUTRAL, sched_easy_span, sched_hard_span, reverse=True)}">{format_decimal(row.schedule_difficulty)}</td>',
            f'<td class="wc-num" style="{sequential_form_cell_style(row.results_form, *sequential_ranges["results_form"])}">{format_decimal(row.results_form, decimals=3)}</td>',
            f'<td class="wc-num" style="{diverging_form_cell_style(row.gd_form, 0.0, gd_form_negative_span, gd_form_positive_span)}">{format_decimal(row.gd_form, decimals=3)}</td>',
            f'<td class="wc-num" style="{sequential_form_cell_style(row.expected_score, *sequential_ranges["expected_score"])}">{format_decimal(row.expected_score, decimals=3)}</td>',
            f'<td class="wc-num" style="{diverging_form_cell_style(row.perf_vs_exp, 0.0, perf_negative_span, perf_positive_span)}">{format_decimal(row.perf_vs_exp, decimals=3)}</td>',
            f'<td class="wc-num">{format_decimal(row.elo_delta_form, decimals=3)}</td>',
            f'<td class="wc-num" style="{sequential_form_cell_style(row.form, *sequential_ranges["form"])}">{format_decimal(row.form)}</td>',
        ]
        if has_history_columns:
            cells.extend(
                [
                    f'<td class="wc-num" style="{sequential_form_cell_style(row.weighted_world_cup_participations, *sequential_ranges["weighted_world_cup_participations"])}">{format_decimal(row.weighted_world_cup_participations)}</td>',
                    f'<td class="wc-num" style="{sequential_form_cell_style(row.weighted_world_cup_placement_score, *sequential_ranges["weighted_world_cup_placement_score"])}">{format_decimal(row.weighted_world_cup_placement_score, decimals=4)}</td>',
                    f'<td class="wc-num" style="{sequential_form_cell_style(row.history_score, *sequential_ranges["history_score"])}">{format_decimal(row.history_score, decimals=4)}</td>',
                    f'<td class="wc-num" style="{sequential_form_cell_style(row.v2_strength, *sequential_ranges["v2_strength"])}">{format_decimal(row.v2_strength)}</td>',
                ]
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return build_table_card_html(headers, body_rows, title, card_subtitle, group_pill_label=group_pill_label)


def build_table_html(
    df: pd.DataFrame,
    title: str,
    include_group_column: bool = False,
    include_ko_column: bool = False,
    card_subtitle: str = "Pre-Tournament Probability Table",
    group_pill_label: str | None = None,
    table_kind: str = "probability",
) -> str:
    """Render one dashboard table card."""
    if table_kind == "form":
        return build_form_table_html(
            df,
            title,
            card_subtitle=card_subtitle,
            group_pill_label=group_pill_label,
        )
    return build_probability_table_html(
        df,
        title,
        include_group_column=include_group_column,
        include_ko_column=include_ko_column,
        card_subtitle=card_subtitle,
        group_pill_label=group_pill_label,
    )


def group_table_frame(df: pd.DataFrame, group_code: str) -> pd.DataFrame:
    """Return one group in the standard probability-table display order."""
    group_df = df[df["group_code"] == group_code].copy()
    return group_df.sort_values(["prob_1", "elo_rating", "world_rank"], ascending=[False, False, True])


def projected_group_table_frame(df: pd.DataFrame, group_code: str) -> pd.DataFrame:
    """Return one group ordered by the same modal ranking the deterministic bracket uses."""
    group_df = df[df["group_code"] == group_code].copy()
    if group_df.empty:
        return group_df

    try:
        modal_group_rankings = get_modal_group_rankings(df)
    except ValueError:
        return group_table_frame(df, group_code)

    projected_order = modal_group_rankings.get(group_code)
    if not projected_order:
        return group_table_frame(df, group_code)

    projected_rank_lookup = {team_id: rank for rank, team_id in enumerate(projected_order, start=1)}
    group_df["projected_rank"] = (
        group_df["team_id"].map(projected_rank_lookup).fillna(len(projected_order) + 1).astype(int)
    )
    return group_df.sort_values(
        ["projected_rank", "prob_1", "elo_rating", "world_rank"],
        ascending=[True, False, False, True],
        kind="stable",
    ).drop(columns=["projected_rank"])


def all_teams_table_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full team table sorted globally by projected chance of finishing 1st."""
    sort_columns = []
    ascending = []
    for column_name in (
        "champion_prob",
        "final_prob",
        "sf_prob",
        "qf_prob",
        "r16_prob",
        "ko_prob",
        "top8_third_prob",
        "prob_1",
    ):
        if column_name in df.columns:
            sort_columns.append(column_name)
            ascending.append(False)
    sort_columns.extend(["elo_rating", "world_rank"])
    ascending.extend([False, True])
    return df.sort_values(sort_columns, ascending=ascending)


def form_table_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full recent-form table sorted by weighted form descending."""
    if "v2_strength" in df.columns:
        return df.sort_values(["v2_strength", "form", "elo_rating", "world_rank"], ascending=[False, False, False, True], kind="stable")
    return df.sort_values(["form", "elo_rating", "world_rank"], ascending=[False, False, True], kind="stable")


def confederation_form_table_frame(df: pd.DataFrame, confederation: str) -> pd.DataFrame:
    """Return one confederation-specific form table sorted by weighted form descending."""
    confed_df = df[df["confederation"] == confederation].copy()
    return form_table_frame(confed_df)


def ordered_confederations(df: pd.DataFrame) -> list[str]:
    """Return confederations in a stable dashboard order, then any extras alphabetically."""
    present = {str(value) for value in df["confederation"].dropna().unique()}
    ordered = [confederation for confederation in FORM_CONFEDERATION_ORDER if confederation in present]
    extras = sorted(present.difference(FORM_CONFEDERATION_ORDER))
    return ordered + extras


def build_form_view_tables(
    form_df: pd.DataFrame,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> list[dict[str, object]]:
    """Build the overall form table plus one table per confederation."""
    subtitle = (
        f"V2 Team Strength | Rating 40 / Form 40 / History 20 | Last {form_match_window} Pre-tournament Matches"
        if "v2_strength" in form_df.columns
        else f"Weighted Recent Form | Last {form_match_window} Pre-tournament Matches | data: eloratings.net | @cartierkut1"
    )
    tables: list[dict[str, object]] = [
        {
            "title": "All Countries",
            "stem": "form_all_countries",
            "frame": form_table_frame(form_df),
            "include_group_column": False,
            "include_ko_column": False,
            "card_subtitle": subtitle,
            "group_pill_label": None,
            "table_kind": "form",
        }
    ]
    for confederation in ordered_confederations(form_df):
        confed_df = confederation_form_table_frame(form_df, confederation)
        if confed_df.empty:
            continue
        tables.append(
            {
                "title": confederation,
                "stem": f"form_{confederation.lower()}",
                "frame": confed_df,
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": subtitle,
                "group_pill_label": None,
                "table_kind": "form",
            }
        )
    return tables


def build_confederation_form_tables(
    form_df: pd.DataFrame,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> list[dict[str, object]]:
    """Build one form table per confederation."""
    subtitle = (
        f"V2 Team Strength | Rating 40 / Form 40 / History 20 | Last {form_match_window} Pre-tournament Matches"
        if "v2_strength" in form_df.columns
        else f"Weighted Recent Form | Last {form_match_window} Pre-tournament Matches | data: eloratings.net | @cartierkut1"
    )
    tables: list[dict[str, object]] = []
    for confederation in ordered_confederations(form_df):
        confed_df = confederation_form_table_frame(form_df, confederation)
        if confed_df.empty:
            continue
        tables.append(
            {
                "title": confederation,
                "stem": f"form_{confederation.lower()}",
                "frame": confed_df,
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": subtitle,
                "group_pill_label": None,
                "table_kind": "form",
            }
        )
    return tables


def current_form_view_tables(
    form_df: pd.DataFrame,
    view_mode: str,
    selected_confederation: str,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> list[dict[str, object]]:
    """Describe the tables needed for the active V2 form view."""
    subtitle = (
        f"V2 Team Strength | Rating 40 / Form 40 / History 20 | Last {form_match_window} Pre-tournament Matches"
        if "v2_strength" in form_df.columns
        else f"Weighted Recent Form | Last {form_match_window} Pre-tournament Matches | data: eloratings.net | @cartierkut1"
    )
    if view_mode == "All Countries":
        return [
            {
                "title": "All Countries",
                "stem": "form_all_countries",
                "frame": form_table_frame(form_df),
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": subtitle,
                "group_pill_label": None,
                "table_kind": "form",
            }
        ]
    if view_mode == "Single confederation":
        return [
            {
                "title": selected_confederation,
                "frame": confederation_form_table_frame(form_df, selected_confederation),
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": subtitle,
                "group_pill_label": None,
                "table_kind": "form",
            }
        ]
    return build_confederation_form_tables(form_df, form_match_window=form_match_window)


def team_metadata_lookup(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Build a lookup of display labels and flag codes for bracket rendering."""
    unique_rows = df.drop_duplicates(subset=["team_id"], keep="first")
    return {
        str(row.team_id): {
            "display_name": str(row.display_name),
            "flag_icon_code": str(row.flag_icon_code) if pd.notna(row.flag_icon_code) else "",
        }
        for row in unique_rows.itertuples(index=False)
    }


def render_bracket_team(team_id: str, metadata_lookup: dict[str, dict[str, str]], is_winner: bool) -> str:
    """Render one team row inside a bracket match card."""
    metadata = metadata_lookup.get(team_id, {"display_name": team_id, "flag_icon_code": ""})
    classes = "wc-bracket-team wc-bracket-team-win" if is_winner else "wc-bracket-team"
    team_name = html.escape(metadata["display_name"])
    flag_icon_code = metadata["flag_icon_code"]
    if flag_icon_code:
        label = (
            f'<span class="fi fi-{html.escape(flag_icon_code)}"></span>'
            f'<span class="wc-bracket-team-name">{team_name}</span>'
        )
    else:
        label = f'<span class="wc-bracket-team-name">{team_name}</span>'
    return f'<div class="{classes}">{label}</div>'


def render_bracket_match(match: dict[str, object], metadata_lookup: dict[str, dict[str, str]]) -> str:
    """Render one predicted knockout match card."""
    winner_team_id = str(match["winner_team_id"])
    home_team_id = str(match["home_team_id"])
    away_team_id = str(match["away_team_id"])
    probability_label = format_percent(float(match["winner_win_prob"]))
    return textwrap.dedent(
        f"""
        <div class="wc-bracket-match">
          <div class="wc-bracket-match-head">
            <div class="wc-bracket-match-number">Match {int(match["match_number"])}</div>
            <div class="wc-bracket-match-prob">{probability_label}</div>
          </div>
          <div class="wc-bracket-teams">
            {render_bracket_team(home_team_id, metadata_lookup, home_team_id == winner_team_id)}
            {render_bracket_team(away_team_id, metadata_lookup, away_team_id == winner_team_id)}
          </div>
        </div>
        """
    ).strip()


def build_bracket_round_column(round_data: dict[str, object], side: str) -> str:
    """Render one bracket round column for the left or right half of the tree."""
    round_code = str(round_data["round_code"]).lower()
    classes = f"wc-bracket-round wc-bracket-round-{side}-{round_code}"
    matches_html = "".join(round_data["matches"])
    return textwrap.dedent(
        f"""
        <div class="{classes}">
          <div class="wc-bracket-round-title">{html.escape(str(round_data["round_label"]))}</div>
          {matches_html}
        </div>
        """
    ).strip()


def build_bracket_html(
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
    card_subtitle: str = "Predicted Knockout Bracket",
) -> str:
    """Render the deterministic knockout bracket as a left-right tournament tree."""
    round_lookup = {
        str(round_data["round_code"]): {
            **round_data,
            "matches": [render_bracket_match(match, metadata_lookup) for match in round_data["matches"]],
        }
        for round_data in bracket_data["rounds"]
    }
    left_order = ["R32", "R16", "QF", "SF"]
    right_order = ["SF", "QF", "R16", "R32"]
    left_columns = []
    right_columns = []
    for round_code in left_order:
        round_data = round_lookup[round_code]
        midpoint = len(round_data["matches"]) // 2
        left_columns.append(
            build_bracket_round_column(
                {
                    "round_code": round_data["round_code"],
                    "round_label": round_data["round_label"],
                    "matches": round_data["matches"][:midpoint],
                },
                side="left",
            )
        )
    for round_code in right_order:
        round_data = round_lookup[round_code]
        midpoint = len(round_data["matches"]) // 2
        right_columns.append(
            build_bracket_round_column(
                {
                    "round_code": round_data["round_code"],
                    "round_label": round_data["round_label"],
                    "matches": round_data["matches"][midpoint:],
                },
                side="right",
            )
        )
    final_round = round_lookup["F"]
    final_match_html = final_round["matches"][0] if final_round["matches"] else ""
    qualifying_groups = html.escape(str(bracket_data["qualifying_third_place_groups"]))
    return textwrap.dedent(
        f"""
        <div class="wc-card">
          <div class="wc-card-header">
            <div>
              <div class="wc-card-subtitle">{html.escape(card_subtitle)}</div>
              <div class="wc-card-title">Bracket</div>
            </div>
          </div>
          <div class="wc-table-wrap">
            <div class="wc-bracket-note">Best third-place groups in this predicted bracket: {qualifying_groups}</div>
            <div class="wc-bracket-board">
              <div class="wc-bracket-side wc-bracket-side-left">{''.join(left_columns)}</div>
              <div class="wc-bracket-final-column">
                <div class="wc-bracket-final-title">{html.escape(str(final_round["round_label"]))}</div>
                {final_match_html}
              </div>
              <div class="wc-bracket-side wc-bracket-side-right">{''.join(right_columns)}</div>
            </div>
          </div>
        </div>
        """
    ).strip()


def current_view_tables(
    df: pd.DataFrame | None,
    view_mode: str,
    selected_group: str,
    simulation_count: int | None = None,
    form_df: pd.DataFrame | None = None,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> list[dict[str, object]]:
    """Describe the tables needed for the active dashboard view."""
    if view_mode == "Form":
        if form_df is None:
            raise ValueError("Form view requires form_df")
        return build_form_view_tables(form_df, form_match_window=form_match_window)
    if df is None:
        raise ValueError("Probability table views require a dataframe")
    if view_mode == "Single group":
        return [
            {
                "title": f"Group {selected_group}",
                "stem": f"group_{selected_group.lower()}",
                "frame": projected_group_table_frame(df, selected_group),
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": chart_subtitle("Bracket-Aligned Projected Order", simulation_count),
                "group_pill_label": selected_group,
                "table_kind": "probability",
            },
        ]
    if view_mode == "All groups":
        tables = []
        for group_code in GROUP_ORDER:
            group_df = projected_group_table_frame(df, group_code)
            if group_df.empty:
                continue
            tables.append(
                {
                    "title": f"Group {group_code}",
                    "stem": f"group_{group_code.lower()}",
                    "frame": group_df,
                    "include_group_column": False,
                    "include_ko_column": False,
                    "card_subtitle": chart_subtitle("Bracket-Aligned Projected Order", simulation_count),
                    "group_pill_label": group_code,
                    "table_kind": "probability",
                }
            )
        return tables
    combined = all_teams_table_frame(df)
    return [
        {
            "title": "All Countries",
            "stem": "all_Countries",
            "frame": combined,
            "include_group_column": True,
            "include_ko_column": True,
            "card_subtitle": chart_subtitle("Pre-Tournament Probability Table", simulation_count),
            "group_pill_label": None,
            "table_kind": "probability",
        }
    ]


def render_tables(
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
) -> None:
    """Render one or many HTML tables into the Streamlit dashboard."""
    if separate_sections and not multi_column:
        section_html = "".join(
            (
                '<div class="wc-grid-single">'
                + build_table_html(
                    table["frame"],
                    table["title"],
                    include_group_column=table["include_group_column"],
                    include_ko_column=table["include_ko_column"],
                    card_subtitle=str(table.get("card_subtitle", "Pre-Tournament Probability Table")),
                    group_pill_label=table.get("group_pill_label"),
                    table_kind=str(table.get("table_kind", "probability")),
                )
                + "</div>"
            )
            for table in tables
        )
        st.markdown(section_html, unsafe_allow_html=True)
        return

    container_class = "wc-grid" if multi_column else "wc-grid-single"
    grid_html = "".join(
        build_table_html(
            table["frame"],
            table["title"],
            include_group_column=table["include_group_column"],
            include_ko_column=table["include_ko_column"],
            card_subtitle=str(table.get("card_subtitle", "Pre-Tournament Probability Table")),
            group_pill_label=table.get("group_pill_label"),
            table_kind=str(table.get("table_kind", "probability")),
        )
        for table in tables
    )
    st.markdown(f'<div class="{container_class}">{grid_html}</div>', unsafe_allow_html=True)


def render_bracket(
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
    simulation_count: int | None = None,
) -> None:
    """Render the deterministic knockout bracket view."""
    st.markdown(
        build_bracket_html(
            bracket_data,
            metadata_lookup,
            card_subtitle=chart_subtitle("Predicted Knockout Bracket", simulation_count),
        ),
        unsafe_allow_html=True,
    )


def render_export_document(
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
) -> str:
    """Render a complete standalone HTML document for export."""
    if separate_sections and not multi_column:
        tables_html = "".join(
            (
                '<div class="wc-grid-single">'
                + build_table_html(
                    table["frame"],
                    table["title"],
                    include_group_column=table["include_group_column"],
                    include_ko_column=table["include_ko_column"],
                    card_subtitle=str(table.get("card_subtitle", "Pre-Tournament Probability Table")),
                    group_pill_label=table.get("group_pill_label"),
                    table_kind=str(table.get("table_kind", "probability")),
                )
                + "</div>"
            )
            for table in tables
        )
    else:
        container_class = "wc-grid" if multi_column else "wc-grid-single"
        cards_html = "".join(
            build_table_html(
                table["frame"],
                table["title"],
                include_group_column=table["include_group_column"],
                include_ko_column=table["include_ko_column"],
                card_subtitle=str(table.get("card_subtitle", "Pre-Tournament Probability Table")),
                group_pill_label=table.get("group_pill_label"),
                table_kind=str(table.get("table_kind", "probability")),
            )
            for table in tables
        )
        tables_html = f'<div class="{container_class}">{cards_html}</div>'
    document = f"""<!DOCTYPE html>
<html class="wc-export-mode" lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>{shared_css()}</style>
</head>
<body class="wc-export-mode">
  <div class="wc-export-page">
    {tables_html}
  </div>
</body>
</html>
"""
    return document


def render_bracket_document(
    page_title: str,
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
    simulation_count: int | None = None,
) -> str:
    """Render a standalone HTML document for the bracket view."""
    bracket_html = build_bracket_html(
        bracket_data,
        metadata_lookup,
        card_subtitle=chart_subtitle("Predicted Knockout Bracket", simulation_count),
    )
    return f"""<!DOCTYPE html>
<html class="wc-export-mode" lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>{shared_css()}</style>
</head>
<body class="wc-export-mode">
  <div class="wc-export-page">
    {bracket_html}
  </div>
</body>
</html>
"""


def build_export_stem(filename_stem: str, export_suffix: str | None = None) -> str:
    """Build the export filename stem, adding a unique suffix when requested."""
    return filename_stem if not export_suffix else f"{filename_stem}_{export_suffix}"


def generate_export_suffix() -> str:
    """Generate a timestamp suffix so each export writes a fresh artifact."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def estimate_export_column_count(table: dict[str, object]) -> int:
    """Estimate the visible column count for one exported table."""
    table_kind = str(table.get("table_kind", "probability"))
    if table_kind == "form":
        frame = table.get("frame")
        if isinstance(frame, pd.DataFrame) and "v2_strength" in frame.columns:
            return 22
        return 18

    column_count = 7
    if bool(table.get("include_group_column")):
        column_count += 2
    if bool(table.get("include_ko_column")):
        column_count += len(ALL_COUNTRIES_KNOCKOUT_COLUMNS)
    return column_count


def estimate_export_viewport_size(
    tables: list[dict[str, object]],
    multi_column: bool,
) -> str:
    """Estimate a screenshot viewport wide enough for the exported content."""
    if not tables:
        return f"{EXPORT_MIN_VIEWPORT_WIDTH},{EXPORT_VIEWPORT_HEIGHT}"

    if multi_column:
        visible_columns = min(3, max(1, len(tables)))
        width = 700 * visible_columns + 120
    else:
        column_count = max(estimate_export_column_count(table) for table in tables)
        width = 560 + max(0, column_count - 1) * 104

    width = max(EXPORT_MIN_VIEWPORT_WIDTH, min(EXPORT_MAX_VIEWPORT_WIDTH, width))
    return f"{width},{EXPORT_VIEWPORT_HEIGHT}"


def export_document_png(
    filename_stem: str,
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
    export_suffix: str | None = None,
) -> Path:
    """Export a complete standalone HTML view as a PNG screenshot."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = render_export_document(
        page_title,
        tables,
        multi_column,
        separate_sections=separate_sections,
    )
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    output_path = EXPORT_DIR / f"{output_stem}.png"

    with tempfile.TemporaryDirectory(prefix="wc_export_", dir=str(EXPORT_DIR)) as temp_dir:
        temp_html_path = Path(temp_dir) / f"{output_stem}.html"
        temp_html_path.write_text(document, encoding="utf-8")
        page_url = temp_html_path.resolve().as_uri()
        viewport_size = estimate_export_viewport_size(tables, multi_column=multi_column)

        last_error = ""
        for channel in SCREENSHOT_CHANNELS:
            command = build_screenshot_command(
                page_url,
                output_path,
                channel,
                viewport_size=viewport_size,
            )
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return output_path
            except FileNotFoundError:
                last_error = "playwright.exe was not found on PATH."
                break
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or exc.stdout or str(exc)).strip()

        raise RuntimeError(f"PNG export failed: {last_error}")


def build_screenshot_command(
    page_url: str,
    output_path: Path,
    channel: str,
    viewport_size: str | None = None,
) -> list[str]:
    """Build the Playwright screenshot command, optionally forcing a viewport size."""
    command = [
        "playwright.exe",
        "screenshot",
        "--full-page",
        "--wait-for-timeout",
        "1500",
        "--channel",
        channel,
    ]
    if viewport_size:
        command.extend(["--viewport-size", viewport_size])
    command.extend([page_url, str(output_path)])
    return command


def export_bracket_png(
    filename_stem: str,
    page_title: str,
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
    simulation_count: int | None = None,
    export_suffix: str | None = None,
) -> Path:
    """Export the deterministic bracket view as a PNG screenshot."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = render_bracket_document(
        page_title,
        bracket_data,
        metadata_lookup,
        simulation_count=simulation_count,
    )
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    output_path = EXPORT_DIR / f"{output_stem}.png"

    with tempfile.TemporaryDirectory(prefix="wc_export_", dir=str(EXPORT_DIR)) as temp_dir:
        temp_html_path = Path(temp_dir) / f"{output_stem}.html"
        temp_html_path.write_text(document, encoding="utf-8")
        page_url = temp_html_path.resolve().as_uri()

        last_error = ""
        for channel in SCREENSHOT_CHANNELS:
            command = build_screenshot_command(
                page_url,
                output_path,
                channel,
                viewport_size=BRACKET_EXPORT_VIEWPORT_SIZE,
            )
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return output_path
            except FileNotFoundError:
                last_error = "playwright.exe was not found on PATH."
                break
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or exc.stdout or str(exc)).strip()

        raise RuntimeError(f"PNG export failed: {last_error}")


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


def render_v1_dashboard() -> None:
    """Render the version 1 probability and bracket dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V1_STATE_KEY not in st.session_state:
        st.session_state[V1_STATE_KEY] = default_simulation_settings()
    current_settings = dict(st.session_state[V1_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
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
    render_dashboard_header(world_cup_logo_data_uri, metadata, simulation_count, title="World Cup 2026 V1")
    render_countdown_timer(fixtures_df)

    st.caption(
        f"Model {MODEL_VERSION}: {MODEL_SUMMARY}. "
        "Probabilities come from a fixture-by-fixture group simulation using the real 2026 schedule, "
        "an Elo-only baseline (100% / 0%), "
        f"recent form from the last {DEFAULT_RECENT_MATCH_WINDOW} matches, "
        "built from points vs goal difference (70% / 30%), "
        "and a ratings-vs-form blend (50% / 50%). "
        "Top 8 3rd% is the share of runs where a team finishes third and still advances. "
        "KO% means reaching the Round of 32; R16%, QF%, SF%, Final%, and Champion% track deeper knockout progression. "
        "This page is isolated to the original probability and bracket model."
    )
    with st.spinner(f"Running {simulation_count:,} simulations..."):
        dashboard_df = simulate_probabilities(
            base_df=base_df,
            fixtures_df=fixtures_df,
            lead_in_df=lead_in_df,
            simulations=simulation_count,
        )
        bracket_data = build_deterministic_bracket(
            dashboard_df,
            fixtures_df,
            head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        )
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    metadata_lookup = team_metadata_lookup(dashboard_df)
    tables = [] if view_mode == "Bracket" else current_view_tables(
        dashboard_df,
        view_mode,
        selected_group,
        simulation_count=simulation_count,
    )
    multi_column = view_mode == "All groups"

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Export This V1 View", use_container_width=True, key="v1_export_current"):
            try:
                export_path = export_current_view(
                    view_mode,
                    selected_group,
                    tables,
                    bracket_data=bracket_data,
                    metadata_lookup=metadata_lookup,
                    simulation_count=simulation_count,
                )
                st.success(f"Exported current view to {export_path}")
            except RuntimeError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))
    with action_cols[1]:
        if st.button("Export All V1 Tables", use_container_width=True, key="v1_export_all"):
            try:
                exported_paths = export_all_tables(
                    probability_df=dashboard_df,
                    simulation_count=simulation_count,
                )
                st.success(f"Exported {len(exported_paths)} PNG tables to {EXPORT_DIR}")
            except RuntimeError as exc:
                st.error(str(exc))

    if view_mode == "Bracket":
        if bracket_data is None or metadata_lookup is None:
            st.error("Bracket view is unavailable because no bracket data was generated.")
            return
        render_bracket(bracket_data, metadata_lookup, simulation_count=simulation_count)
    else:
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
    form_composite_weights = (
        results_weight,
        gd_weight,
        perf_weight,
        elo_delta_weight,
    )
    view_mode = st.radio("View", V2_VIEW_OPTIONS, horizontal=True, key="v2_view_mode")
    st.session_state[V2_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
        "v2_results_weight": results_weight,
        "v2_gd_weight": gd_weight,
        "v2_perf_weight": perf_weight,
        "v2_elo_delta_weight": elo_delta_weight,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    render_dashboard_header(world_cup_logo_data_uri, metadata, simulation_count, title="World Cup 2026 V2")
    render_countdown_timer(fixtures_df)
    st.caption(
        f"V2 isolates the history-aware model from V1. This page ranks all 48 teams using rating (40%), weighted lead-in form (40%), "
        f"and World Cup history (20%). Form covers the last {form_match_window} Elo-rated matches with component weights: "
        f"Results {results_weight}, GD {gd_weight}, PoE {perf_weight}, Elo Delta {elo_delta_weight}. "
        "History blends weighted World Cup placement (70%) with weighted appearance count (30%), with DNQ editions scored as zero."
    )
    with st.spinner(f"Computing V2 history-aware strength for the last {form_match_window} matches..."):
        form_df = build_v2_team_strengths(
            base_df,
            lead_in_df,
            match_window=form_match_window,
            form_composite_weights=form_composite_weights,
        )
    available_confederations = ordered_confederations(form_df)
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

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Export This V2 Page", use_container_width=True, key="v2_export_current"):
            try:
                export_stem = "form_all_countries" if view_mode == "All Countries" else (
                    f"form_{selected_confederation.lower()}" if view_mode == "Single confederation" and selected_confederation else "form_all_confederations"
                )
                export_title = "All Countries" if view_mode == "All Countries" else (
                    selected_confederation if view_mode == "Single confederation" and selected_confederation else "All Confederations"
                )
                export_path = export_document_png(
                    export_stem,
                    export_title,
                    tables,
                    multi_column=False,
                    export_suffix=generate_export_suffix(),
                )
                st.success(f"Exported current view to {export_path}")
            except RuntimeError as exc:
                st.error(str(exc))
    with action_cols[1]:
        if st.button("Export All V2 Tables", use_container_width=True, key="v2_export_all"):
            try:
                exported_paths = export_all_tables(
                    form_df=form_df,
                    simulation_count=simulation_count,
                    form_match_window=form_match_window,
                )
                st.success(f"Exported {len(exported_paths)} PNG tables to {EXPORT_DIR}")
            except RuntimeError as exc:
                st.error(str(exc))
    render_tables(tables, multi_column=False)


def render_v2_probabilities_dashboard() -> None:
    """Render the version 2 multinomial probability and bracket dashboard."""
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    if V2_PROB_STATE_KEY not in st.session_state:
        st.session_state[V2_PROB_STATE_KEY] = default_simulation_settings()
    current_settings = dict(st.session_state[V2_PROB_STATE_KEY])

    simulation_labels = tuple(SIMULATION_OPTIONS.keys())
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
    view_mode = st.radio("View", V2_PROB_VIEW_OPTIONS, horizontal=True, key="v2_prob_view_mode")
    selected_group = (
        st.selectbox("Group", GROUP_ORDER, index=0, key="v2_prob_selected_group")
        if view_mode == "Single group"
        else GROUP_ORDER[0]
    )

    st.session_state[V2_PROB_STATE_KEY] = {
        "simulation_label": simulation_label,
        "form_match_window": form_match_window,
    }

    simulation_count = SIMULATION_OPTIONS[simulation_label]
    render_dashboard_header(
        world_cup_logo_data_uri,
        metadata,
        simulation_count,
        title="World Cup 2026 V2 Probabilities",
        model_version=V2_MODEL_VERSION,
        model_label=V2_MODEL_LABEL,
    )
    render_countdown_timer(fixtures_df)
    st.caption(
        f"Model {V2_MODEL_VERSION}: {V2_MODEL_SUMMARY}. "
        f"The v2 page trains a three-class multinomial regression on all World Cup matches from 1950 through 2022, "
        f"then simulates the real 2026 bracket using pre-tournament Elo, weighted form from the last {form_match_window} Elo-rated matches, "
        "and prior World Cup history features. Knockout draws are interpreted using the local historical file semantics: "
        "level before penalties, then resolved by the model's non-draw split."
    )
    with st.spinner(f"Training v2 model and running {simulation_count:,} simulations..."):
        model_bundle = load_v2_match_model(form_match_window)
        dashboard_df = simulate_probabilities_v2_dashboard(
            base_df=base_df,
            fixtures_df=fixtures_df,
            lead_in_df=lead_in_df,
            simulations=simulation_count,
            match_window=form_match_window,
        )
        bracket_data = build_deterministic_bracket_v2(
            dashboard_df,
            fixtures_df,
            dashboard_df,
            model_bundle,
            head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        )
    dashboard_df = ensure_dashboard_probability_columns(dashboard_df)
    metadata_lookup = team_metadata_lookup(dashboard_df)
    tables = [] if view_mode == "Bracket" else current_view_tables(
        dashboard_df,
        view_mode,
        selected_group,
        simulation_count=simulation_count,
    )
    multi_column = view_mode == "All groups"

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Export This V2 Probability View", use_container_width=True, key="v2_prob_export_current"):
            try:
                export_path = export_current_view(
                    view_mode,
                    selected_group,
                    tables,
                    bracket_data=bracket_data,
                    metadata_lookup=metadata_lookup,
                    simulation_count=simulation_count,
                )
                st.success(f"Exported current view to {export_path}")
            except RuntimeError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))
    with action_cols[1]:
        if st.button("Export All V2 Probability Tables", use_container_width=True, key="v2_prob_export_all"):
            try:
                exported_paths = export_all_tables(
                    probability_df=dashboard_df,
                    simulation_count=simulation_count,
                )
                st.success(f"Exported {len(exported_paths)} PNG tables to {EXPORT_DIR}")
            except RuntimeError as exc:
                st.error(str(exc))

    if view_mode == "Bracket":
        render_bracket(bracket_data, metadata_lookup, simulation_count=simulation_count)
    else:
        render_tables(tables, multi_column=multi_column)


def main() -> None:
    """Render the landing page for the versioned dashboard pages."""
    configure_page("World Cup 2026 Dashboard")
    inject_styles()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    _, fixtures_df, _, metadata = load_data()
    render_dashboard_header(world_cup_logo_data_uri, metadata, SIMULATION_COUNT, title="World Cup 2026 Dashboard")
    render_countdown_timer(fixtures_df)
    st.markdown(
        """
        ### Versions
        Use the sidebar pages to keep model versions isolated.

        - `V1 Probabilities` contains the original group-probability and bracket workflow.
        - `V2 Form` contains the weighted-form tables and confederation splits.
        - `V2 Probabilities` contains the multinomial match model and full-tournament Monte Carlo outputs.

        Settings and exports are separated per page so changes in one version do not interfere with the other by accident.
        """
    )


if __name__ == "__main__":
    main()
