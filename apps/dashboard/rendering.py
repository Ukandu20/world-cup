from __future__ import annotations

import html
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import (
    ALL_COUNTRIES_KNOCKOUT_COLUMNS,
    BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    CURRENT_HOLDER_TEAM_ID,
    DEFAULT_RECENT_MATCH_WINDOW,
    FORM_AMBER_GRADIENT,
    FORM_AMBER_TEXT,
    FORM_CONFEDERATION_ORDER,
    FORM_GREEN_GRADIENT,
    FORM_GREEN_TEXT,
    FORM_RED_GRADIENT,
    FORM_RED_TEXT,
    FORM_SCHEDULE_DIFFICULTY_NEUTRAL,
    GROUP_ORDER,
    MODEL_LABEL,
    MODEL_VERSION,
    PROBABILITY_PALETTES,
    SIMULATION_COUNT,
    V1_VIEW_OPTIONS,
    V2_MODEL_LABEL,
    V2_MODEL_SUMMARY,
    V2_MODEL_VERSION,
    V2_PROB_VIEW_OPTIONS,
    V3_MODEL_LABEL,
    V3_MODEL_SUMMARY,
    V3_MODEL_VERSION,
    VIEW_OPTIONS,
    build_deterministic_bracket,
    build_deterministic_bracket_v2,
    build_deterministic_bracket_v3,
    build_v2_team_strengths,
    get_modal_group_rankings,
)
from .data import load_champion_trophy_data_uri
from .modeling import ensure_dashboard_probability_columns


def chart_subtitle(base_label: str, simulation_count: int | None = None) -> str:
    """Return a chart subtitle with an optional simulation-count suffix."""
    if simulation_count is None:
        return base_label
    return f"@cartierkut1"


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
    logo_markup = (
        f'<img class="wc-title-logo" src="{world_cup_logo_data_uri}" alt="FIFA World Cup 2026 logo" />'
        if world_cup_logo_data_uri
        else ""
    )
    st.markdown(
        f"""
        <div class="wc-header">
          <div class="wc-header-bar">
            {logo_markup}
            <div>
              <div class="wc-kicker">Pre-Tournament Predictions</div>
              <h1 style="margin:0;">{html.escape(title)}</h1>
              <div class="wc-meta">
                Author: Okechi Ukandu |
                Build date: {html.escape(str(metadata["build_date"]))} |                
                Simulations count: {simulation_count:,}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_filter_bar(title: str = "Filters", expanded: bool = True) -> st.delta_generator.DeltaGenerator:
    """Render a compact expandable in-page filter container."""
    return st.expander(title, expanded=expanded)


def shared_css() -> str:
    """Return the shared CSS used by both Streamlit rendering and exported HTML files."""
    return """
    @import url('https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/css/flag-icons.min.css');

    :root {
        --wc-bg: #EFE3CF;
        --wc-surface: #F6EBD8;
        --wc-surface-strong: #E8D5B8;
        --wc-text: #3A2A1A;
        --wc-muted: #5A4632;
        --wc-line: #D8C8AF;
        --wc-positive: #2F6F3E;
        --wc-positive-soft: rgba(47, 111, 62, 0.12);
        --wc-gold: #C99700;
        --wc-gold-soft: rgba(201, 151, 0, 0.16);
        --wc-danger: #B23A30;
        --wc-shadow: rgba(58, 42, 26, 0.07);
        --wc-font: Gill Sans, Inter, sans-serif;
    }
    body {
        margin: 0;
        background: var(--wc-bg);
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    .stApp {
        background: var(--wc-bg);
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    h1, h2, h3, h4, h5, h6,
    p, label, span,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"] {
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    span[class*="material-symbols"],
    i[class*="material-symbols"],
    [data-testid="stIconMaterial"],
    [class*="material-symbols"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 1.15em !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: "liga" !important;
    }
    [data-testid="stCaptionContainer"],
    .stMarkdown small {
        color: var(--wc-muted);
    }
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: var(--wc-bg);
        color: var(--wc-text);
    }
    [data-testid="stHeader"]::before,
    header[data-testid="stHeader"]::before {
        background: var(--wc-bg);
    }
    [data-testid="stAppViewContainer"] > header {
        background: var(--wc-bg);
        color: var(--wc-text);
        box-shadow: inset 0 -1px 0 var(--wc-line);
    }
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] a,
    [data-testid="stToolbar"] svg,
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] a,
    [data-testid="stHeader"] svg {
        color: var(--wc-muted);
        fill: var(--wc-muted);
    }
    [data-testid="stToolbar"] button:hover,
    [data-testid="stHeader"] button:hover {
        color: var(--wc-text);
        background: var(--wc-surface-strong);
    }
    [data-testid="stNav"],
    nav,
    [role="navigation"] {
        background: var(--wc-bg);
        color: var(--wc-text);
        border-bottom: 1px solid var(--wc-line);
        font-family: var(--wc-font);
    }
    [data-testid="stNav"] *,
    nav *,
    [role="navigation"] * {
        font-family: var(--wc-font);
    }
    [data-testid="stNav"] button,
    [data-testid="stNav"] a,
    nav button,
    nav a,
    [role="navigation"] button,
    [role="navigation"] a {
        color: var(--wc-muted);
        background: transparent;
        border-radius: 8px;
        font-weight: 800;
    }
    [data-testid="stNav"] button:hover,
    [data-testid="stNav"] a:hover,
    nav button:hover,
    nav a:hover,
    [role="navigation"] button:hover,
    [role="navigation"] a:hover {
        color: var(--wc-text);
        background: var(--wc-surface-strong);
    }
    [data-testid="stNav"] button[aria-current="page"],
    [data-testid="stNav"] a[aria-current="page"],
    nav button[aria-current="page"],
    nav a[aria-current="page"],
    [role="navigation"] button[aria-current="page"],
    [role="navigation"] a[aria-current="page"],
    [data-baseweb="menu"] [aria-selected="true"] {
        color: var(--wc-bg);
        background: var(--wc-muted);
    }
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background: var(--wc-surface);
        color: var(--wc-text);
        border: 1px solid var(--wc-line);
        box-shadow: 0 12px 26px rgba(58, 42, 26, 0.12);
        font-family: var(--wc-font);
    }
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"],
    [data-baseweb="menu"] [role="menuitem"] {
        color: var(--wc-text);
        background: var(--wc-surface);
        font-family: var(--wc-font);
        font-weight: 700;
    }
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [role="menuitem"]:hover {
        color: var(--wc-text);
        background: var(--wc-surface-strong);
    }
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] a,
    [data-baseweb="popover"] button,
    [data-baseweb="popover"] [role="menu"],
    [data-baseweb="popover"] [role="menuitem"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
        background-color: var(--wc-surface) !important;
        color: var(--wc-text) !important;
        border-color: var(--wc-line) !important;
        font-family: var(--wc-font) !important;
    }
    [data-baseweb="popover"] svg,
    [data-baseweb="popover"] span,
    [data-baseweb="popover"] p {
        color: var(--wc-text) !important;
        fill: var(--wc-text) !important;
    }
    [data-baseweb="popover"] span:not([class*="material-symbols"]):not([data-testid="stIconMaterial"]),
    [data-baseweb="popover"] p {
        font-family: var(--wc-font) !important;
    }
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] a:hover,
    [data-baseweb="popover"] button:hover,
    [data-baseweb="popover"] [role="menuitem"]:hover,
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"]:hover {
        background-color: var(--wc-surface-strong) !important;
        color: var(--wc-text) !important;
    }
    [data-baseweb="popover"] [aria-selected="true"],
    [data-baseweb="popover"] [aria-current="page"],
    [data-baseweb="popover"] a[aria-current="page"],
    [data-baseweb="popover"] button[aria-current="page"] {
        background-color: var(--wc-surface-strong) !important;
        color: var(--wc-text) !important;
        box-shadow: none !important;
    }
    [data-baseweb="popover"] [aria-selected="true"] *,
    [data-baseweb="popover"] [aria-current="page"] *,
    [data-baseweb="popover"] a[aria-current="page"] *,
    [data-baseweb="popover"] button[aria-current="page"] * {
        color: var(--wc-text) !important;
        fill: var(--wc-text) !important;
    }
    [data-baseweb="popover"] a:focus,
    [data-baseweb="popover"] button:focus,
    [data-baseweb="popover"] [role="menuitem"]:focus,
    [data-baseweb="popover"] [role="option"]:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stBaseButton-header"],
    [data-testid="stBaseButton-headerNoPadding"] {
        color: var(--wc-muted);
    }
    [data-testid="stButton"] button,
    [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]),
    button[kind="primary"],
    button[kind="secondary"] {
        border: 1px solid var(--wc-muted);
        border-radius: 8px;
        background: var(--wc-surface-strong);
        color: var(--wc-text);
        box-shadow: 0 6px 14px rgba(58, 42, 26, 0.06);
        font-family: var(--wc-font);
        font-weight: 800;
    }
    [data-testid="stButton"] button *,
    [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]) * {
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    [data-testid="stButton"] button:hover,
    [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        border-color: var(--wc-positive);
        background: var(--wc-positive);
        color: var(--wc-bg);
        box-shadow: 0 8px 18px rgba(47, 111, 62, 0.16);
    }
    [data-testid="stButton"] button:hover *,
    [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):hover * {
        color: var(--wc-bg);
    }
    [data-testid="stButton"] button:focus,
    [data-testid="stButton"] button:focus-visible,
    [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):focus,
    [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):focus-visible {
        outline: none;
        border-color: var(--wc-positive);
        box-shadow: 0 0 0 3px rgba(47, 111, 62, 0.20);
    }
    [data-testid="stButton"] button:disabled,
    [data-testid^="stBaseButton-"]:disabled,
    button:disabled {
        border-color: var(--wc-line);
        background: rgba(232, 213, 184, 0.58);
        color: rgba(90, 70, 50, 0.62);
        box-shadow: none;
        cursor: not-allowed;
    }
    [data-testid="stButton"] button:disabled *,
    [data-testid^="stBaseButton-"]:disabled * {
        color: rgba(90, 70, 50, 0.62);
    }
    [data-testid="stRadio"] label,
    [data-baseweb="radio"] label {
        color: var(--wc-text);
        font-family: var(--wc-font);
        font-weight: 700;
    }
    [data-baseweb="radio"] div[role="radio"] {
        border-color: var(--wc-muted);
        background-color: var(--wc-bg);
    }
    [data-baseweb="radio"] div[role="radio"][aria-checked="true"] {
        border-color: var(--wc-positive);
        background-color: var(--wc-positive);
        box-shadow: inset 0 0 0 4px var(--wc-bg);
    }
    [data-baseweb="radio"] div[role="radio"]:focus,
    [data-baseweb="radio"] div[role="radio"]:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(47, 111, 62, 0.20);
    }
    [data-baseweb="slider"] [role="slider"] {
        background-color: var(--wc-positive);
        border-color: var(--wc-positive);
        box-shadow: 0 0 0 3px rgba(47, 111, 62, 0.14);
    }
    [data-baseweb="slider"] [role="slider"]:focus,
    [data-baseweb="slider"] [role="slider"]:focus-visible {
        outline: none;
        box-shadow: 0 0 0 4px rgba(47, 111, 62, 0.22);
    }
    [data-baseweb="slider"] div {
        color: var(--wc-text);
    }
    [data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
    [data-baseweb="slider"] div[style*="background: rgb(255, 75, 75)"] {
        background-color: var(--wc-positive) !important;
    }
    [data-baseweb="slider"] div[style*="background-color: rgb(240, 242, 246)"],
    [data-baseweb="slider"] div[style*="background: rgb(240, 242, 246)"] {
        background-color: var(--wc-line) !important;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        border-color: var(--wc-line);
        background: var(--wc-surface);
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    [data-baseweb="select"] > div:hover,
    [data-baseweb="input"] > div:hover,
    [data-baseweb="textarea"] > div:hover {
        border-color: var(--wc-muted);
    }
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="textarea"] > div:focus-within {
        border-color: var(--wc-positive);
        box-shadow: 0 0 0 3px rgba(47, 111, 62, 0.16);
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    [data-baseweb="tag"] {
        border: 1px solid rgba(47, 111, 62, 0.30);
        background: var(--wc-positive-soft);
        color: var(--wc-text);
        font-family: var(--wc-font);
        font-weight: 800;
    }
    [data-baseweb="checkbox"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stToggle"] label {
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
    [data-testid="stToggle"] [data-testid="stWidgetLabel"] {
        color: var(--wc-text);
    }
    [data-testid="stAlert"] {
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        color: var(--wc-text);
        box-shadow: 0 6px 14px rgba(58, 42, 26, 0.05);
    }
    [data-testid="stAlert"] *,
    [data-testid="stException"] * {
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        color: var(--wc-text);
        box-shadow: 0 6px 14px rgba(58, 42, 26, 0.05);
        overflow: hidden;
    }
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {
        font-family: var(--wc-font);
    }
    [data-testid="stJson"],
    [data-testid="stCodeBlock"],
    pre,
    code {
        border-radius: 6px;
        background: rgba(90, 70, 50, 0.12);
        color: var(--wc-positive);
        font-family: Consolas, "Courier New", monospace;
        font-weight: 700;
    }
    [data-testid="stJson"] {
        border: 1px solid var(--wc-line);
        padding: 0.7rem;
    }
    hr {
        border-color: var(--wc-line);
    }
    ::selection {
        background: var(--wc-gold-soft);
        color: var(--wc-text);
    }
    html.wc-export-mode,
    body.wc-export-mode {
        width: max-content;
        min-width: 100%;
    }
    .block-container {
        background: var(--wc-bg);
        padding-top: 4.25rem;
        padding-bottom: 2rem;
    }
    .wc-export-page {
        background: var(--wc-bg);
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
        border: 1px solid var(--wc-line);
        border-radius: 10px;
        background: var(--wc-surface);
        box-shadow: 0 10px 22px var(--wc-shadow);
        overflow: hidden;
        height: 100%;
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
        font-weight: 800;
        color: var(--wc-text);
    }
    .wc-card-subtitle {
        font-size: 0.8rem;
        color: var(--wc-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 800;
    }
    .wc-group-pill {
        min-width: 2.1rem;
        height: 2.1rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--wc-muted);
        color: var(--wc-bg);
        font-weight: 800;
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
        background: var(--wc-muted);
        color: var(--wc-bg);
        font-size: clamp(0.68rem, 0.64rem + 0.18vw, 0.76rem);
        font-weight: 800;
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
        border-bottom: 1px solid var(--wc-line);
        padding: 0.72rem 0.65rem;
        color: var(--wc-text);
        font-size: clamp(0.82rem, 0.78rem + 0.2vw, 0.93rem);
        vertical-align: middle;
        background-color: rgba(246, 235, 216, 0.86);
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
        box-shadow: inset 0 0 0 1px rgba(90, 70, 50, 0.22);
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
        background: rgba(90, 70, 50, 0.14);
        box-shadow: inset 0 0 0 1px rgba(90, 70, 50, 0.12);
    }
    .wc-qual-segment {
        position: absolute;
        left: 0;
        right: 0;
    }
    .wc-qual-segment-top2 {
        bottom: 0;
        background: linear-gradient(180deg, #4F8A5B 0%, var(--wc-positive) 100%);
    }
    .wc-qual-segment-third {
        background: linear-gradient(180deg, #E1B94F 0%, var(--wc-gold) 100%);
    }
    .wc-holder-cell {
        background: linear-gradient(180deg, #F3E4B4 0%, #E8D5A1 100%);
        color: var(--wc-text);
    }
    .wc-holder-cell .wc-name-text {
        color: var(--wc-text);
        font-weight: 800;
    }
    .wc-holder-cell .fi {
        box-shadow: inset 0 0 0 1px rgba(90, 70, 50, 0.24), 0 0 0 2px var(--wc-gold-soft);
    }
    .wc-prob {
        font-variant-numeric: tabular-nums;
        font-weight: 700;
    }
    .wc-kicker {
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: var(--wc-muted);
        margin-bottom: 0.35rem;
        font-weight: 800;
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
        margin-bottom: 0.85rem;
        color: var(--wc-text);
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
        color: var(--wc-muted);
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
    [data-testid="stExpander"] {
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.05);
        margin: 0.25rem 0 1rem;
    }
    [data-testid="stExpander"] details {
        border: none;
    }
    [data-testid="stExpander"] summary {
        background: var(--wc-surface-strong);
        color: var(--wc-text);
        border-radius: 8px 8px 0 0;
        font-family: var(--wc-font);
        font-weight: 800;
        min-height: 2.65rem;
        padding: 0.7rem 1rem;
    }
    [data-testid="stExpander"] summary svg {
        color: var(--wc-muted);
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] p {
        color: var(--wc-text);
        font-family: var(--wc-font);
    }
    [data-testid="stExpander"] [role="radiogroup"] label {
        color: var(--wc-muted);
        font-weight: 700;
    }
    [data-testid="stExpander"] [role="radiogroup"] label:has(input:checked) {
        color: var(--wc-text);
    }
    [data-testid="stExpander"] [data-baseweb="select"] > div {
        background: var(--wc-surface-strong);
        border-color: var(--wc-muted);
        color: var(--wc-text);
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--wc-line);
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--wc-muted);
        font-family: var(--wc-font);
        font-weight: 800;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--wc-text);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--wc-muted);
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
        color: var(--wc-text);
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
        color: var(--wc-text);
    }
    .wc-bracket-match {
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.06);
        padding: 0.65rem 0.7rem;
    }
    .wc-bracket-final-column .wc-bracket-match {
        width: 100%;
        max-width: 220px;
        box-shadow: 0 12px 26px rgba(58, 42, 26, 0.12);
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
        color: var(--wc-muted);
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .wc-bracket-match-prob {
        font-size: 0.76rem;
        font-weight: 800;
        color: var(--wc-bg);
        background: var(--wc-positive);
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
        color: var(--wc-text);
        background: rgba(239, 227, 207, 0.72);
    }
    .wc-bracket-side-right .wc-bracket-team {
        flex-direction: row-reverse;
    }
    .wc-bracket-team-win {
        background: var(--wc-positive-soft);
        box-shadow: inset 0 0 0 1px rgba(47, 111, 62, 0.22);
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
        color: var(--wc-muted);
        font-size: 0.9rem;
        margin: 0.35rem 0 0.2rem;
    }
    .wc-home-intro {
        display: grid;
        grid-template-columns: minmax(0, 1.8fr) minmax(260px, 0.8fr);
        gap: 1rem;
        align-items: stretch;
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.05);
        padding: 1.1rem 1.15rem;
        margin: 0.5rem 0 1.1rem;
    }
    .wc-home-intro-kicker {
        color: var(--wc-positive);
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .wc-home-intro-title {
        color: var(--wc-text);
        font-size: 1.45rem;
        line-height: 1.2;
        font-weight: 850;
        margin: 0 0 0.55rem;
    }
    .wc-home-intro-copy {
        color: var(--wc-text);
        font-size: 0.98rem;
        line-height: 1.5;
        margin: 0;
        max-width: 74rem;
    }
    .wc-home-intro-panel {
        border-left: 1px solid var(--wc-line);
        padding-left: 1rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 0;
    }
    .wc-home-intro-panel-title {
        color: var(--wc-muted);
        font-size: 0.75rem;
        font-weight: 850;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .wc-home-intro-panel-value {
        color: var(--wc-text);
        font-size: 1.12rem;
        line-height: 1.2;
        font-weight: 850;
        margin-bottom: 0.4rem;
    }
    .wc-home-intro-panel-copy {
        color: var(--wc-muted);
        font-size: 0.88rem;
        line-height: 1.4;
        margin: 0;
    }
    .wc-home-section {
        margin: 1rem 0 1.35rem;
    }
    .wc-home-section-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin: 0 0 0.75rem;
    }
    .wc-home-section-title {
        margin: 0;
        font-size: 1.28rem;
        line-height: 1.2;
        font-weight: 800;
        color: var(--wc-text);
    }
    .wc-home-section-note {
        margin: 0.2rem 0 0;
        color: var(--wc-muted);
        font-size: 0.94rem;
        line-height: 1.45;
    }
    .wc-home-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.24rem 0.58rem;
        background: var(--wc-positive);
        color: var(--wc-bg);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .wc-home-metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.8rem 0 1.05rem;
    }
    .wc-home-metric {
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        padding: 0.95rem 1rem;
        min-height: 116px;
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.05);
    }
    .wc-home-metric-label {
        color: var(--wc-muted);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.42rem;
    }
    .wc-home-metric-value {
        color: var(--wc-text);
        font-size: 1.35rem;
        line-height: 1.15;
        font-weight: 850;
        overflow-wrap: anywhere;
    }
    .wc-home-metric-detail {
        color: var(--wc-muted);
        font-size: 0.88rem;
        line-height: 1.35;
        margin-top: 0.42rem;
    }
    .wc-home-route-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
    }
    .wc-home-route-card,
    .wc-home-model-card {
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        background: var(--wc-surface);
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 18px rgba(58, 42, 26, 0.05);
    }
    .wc-home-route-card {
        min-height: 128px;
    }
    .wc-home-route-title,
    .wc-home-model-title {
        color: var(--wc-text);
        font-size: 1rem;
        line-height: 1.2;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .wc-home-route-destination,
    .wc-home-model-version {
        color: var(--wc-positive);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.42rem;
    }
    .wc-home-route-copy,
    .wc-home-model-copy {
        color: var(--wc-muted);
        font-size: 0.9rem;
        line-height: 1.42;
        margin: 0;
    }
    .wc-home-model-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
    }
    .wc-home-model-card-recommended {
        border-color: rgba(47, 111, 62, 0.45);
        box-shadow: inset 0 0 0 1px rgba(47, 111, 62, 0.12), 0 8px 18px rgba(58, 42, 26, 0.05);
    }
    @media (max-width: 1380px) {
        .wc-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .wc-home-metric-grid,
        .wc-home-route-grid {
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
        .wc-home-intro {
            grid-template-columns: 1fr;
        }
        .wc-home-intro-panel {
            border-left: none;
            border-top: 1px solid var(--wc-line);
            padding-left: 0;
            padding-top: 0.85rem;
        }
        .wc-home-metric-grid,
        .wc-home-route-grid,
        .wc-home-model-grid {
            grid-template-columns: 1fr;
        }
        .wc-home-section-head {
            align-items: flex-start;
            flex-direction: column;
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
    <style>
      * {{
        box-sizing: border-box;
      }}
      html,
      body {{
        margin: 0;
        max-width: 100%;
        overflow: hidden;
        background: #EFE3CF;
        color: #3A2A1A;
        font-family: Gill Sans, Inter, sans-serif;
      }}
      .wc-countdown-wrap {{
        margin: 0 0 0.85rem;
        max-width: 100%;
        min-width: 0;
        overflow: hidden;
      }}
      .wc-countdown-card {{
        width: 100%;
        max-width: 100%;
        min-width: 0;
        overflow: hidden;
        border: 1px solid #D8C8AF;
        border-radius: 10px;
        padding: 20px 22px;
        background:
          radial-gradient(circle at top, rgba(201, 151, 0, 0.16), transparent 42%),
          linear-gradient(135deg, #F6EBD8 0%, #E8D5B8 100%);
        color: #3A2A1A;
        box-shadow: 0 12px 26px rgba(58, 42, 26, 0.10);
      }}
      .wc-countdown-main {{
        min-width: 0;
        text-align: center;
      }}
      .wc-countdown-kicker,
      .wc-countdown-meta-label {{
        text-transform: uppercase;
        color: #5A4632;
        font-weight: 850;
        overflow-wrap: anywhere;
      }}
      .wc-countdown-kicker {{
        margin-bottom: 0.5rem;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
      }}
      .wc-countdown-match {{
        margin-bottom: 0.45rem;
        color: #3A2A1A;
        font-size: clamp(1.05rem, 4vw, 1.42rem);
        font-weight: 820;
        line-height: 1.22;
        overflow-wrap: anywhere;
      }}
      #wc-countdown-value {{
        margin: 0.15rem 0 0.8rem;
        color: #2F6F3E;
        font-size: clamp(2rem, 8vw, 3rem);
        font-weight: 900;
        line-height: 1.08;
        text-shadow: 0 5px 18px rgba(47, 111, 62, 0.14);
        overflow-wrap: anywhere;
      }}
      .wc-countdown-meta {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        min-width: 0;
        margin-top: 0.55rem;
        padding-top: 0.85rem;
        border-top: 1px solid #D8C8AF;
      }}
      .wc-countdown-meta-item {{
        min-width: 0;
      }}
      .wc-countdown-meta-item:last-child {{
        text-align: right;
      }}
      .wc-countdown-meta-label {{
        margin-bottom: 0.2rem;
        color: #5A4632;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
      }}
      .wc-countdown-meta-value {{
        color: #3A2A1A;
        font-size: 0.98rem;
        font-weight: 750;
        line-height: 1.25;
        overflow-wrap: anywhere;
      }}
      @media (max-width: 520px) {{
        .wc-countdown-card {{
          border-radius: 14px;
          padding: 16px;
        }}
        .wc-countdown-meta {{
          display: grid;
          grid-template-columns: 1fr;
          gap: 0.65rem;
          align-items: start;
        }}
        .wc-countdown-meta-item,
        .wc-countdown-meta-item:last-child {{
          text-align: center;
        }}
      }}
    </style>
    <div class="wc-countdown-wrap">
      <div class="wc-countdown-card">
        <div class="wc-countdown-main">
          <div class="wc-countdown-kicker">Countdown To Opening Kickoff</div>
          <div class="wc-countdown-match">{match_label}</div>
          <div id="wc-countdown-value">Loading countdown...</div>
        </div>
        <div class="wc-countdown-meta">
          <div class="wc-countdown-meta-item">
            <div class="wc-countdown-meta-label">Date</div>
            <div class="wc-countdown-meta-value">{kickoff_date_label}</div>
          </div>
          <div class="wc-countdown-meta-item">
            <div class="wc-countdown-meta-label">Time [local | UTC]</div>
            <div class="wc-countdown-meta-value">{kickoff_local_time_label} | {kickoff_utc_time_label}</div>
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
    st.iframe(build_countdown_html(kickoff_details), height=235)


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
    if not trophy_data_uri:
        return "Champion %"
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
    card_subtitle: str = "Pre-Tournament Probability Odds",
    group_pill_label: str | None = None,
) -> str:
    """Render one probability table as a styled HTML card."""
    df = ensure_dashboard_probability_columns(df)
    is_all_countries_view = include_group_column and include_ko_column
    include_rank_column = include_group_column
    show_group_qualification_marker = not include_group_column and not include_ko_column
    probability_columns = [] if is_all_countries_view else ["prob_1", "prob_2", "prob_3", "prob_4"]
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
            "<th>Confederation</th>" if is_all_countries_view else "",
            '<th class="wc-num">World Rank</th>',
            '<th class="wc-num">Elo</th>',
            '<th class="wc-num">1st %</th>' if not is_all_countries_view else "",
            '<th class="wc-num">2nd %</th>' if not is_all_countries_view else "",
            '<th class="wc-num">3rd %</th>' if not is_all_countries_view else "",
            '<th class="wc-num">4th %</th>' if not is_all_countries_view else "",
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
                f'<td>{html.escape(str(getattr(row, "confederation", "")))}</td>' if is_all_countries_view else "",
                f'<td class="wc-num">{int(row.world_rank)}</td>',
                f'<td class="wc-num">{int(row.elo_rating)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_1", row.prob_1, *column_ranges["prob_1"])}">{format_percent(row.prob_1)}</td>' if not is_all_countries_view else "",
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_2", row.prob_2, *column_ranges["prob_2"])}">{format_percent(row.prob_2)}</td>' if not is_all_countries_view else "",
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_3", row.prob_3, *column_ranges["prob_3"])}">{format_percent(row.prob_3)}</td>' if not is_all_countries_view else "",
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_4", row.prob_4, *column_ranges["prob_4"])}">{format_percent(row.prob_4)}</td>' if not is_all_countries_view else "",
            ]
        )
        cells = [cell for cell in cells if cell]
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
