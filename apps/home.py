import html
import base64
from datetime import datetime
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

from world_cup_simulation import simulate_group_probabilities

DATA_DIR = ROOT / "INT-World Cup" / "world_cup" / "2026"
EXPORT_DIR = ROOT / "assets" / "charts" / "generated"
WORLD_CUP_LOGO_PATH = ROOT / "assets" / "logos" / "world-cup" / "fifa-world-cup-2026.football.cc.svg"
SIMULATION_COUNT = 20000
SIMULATION_OPTIONS = {
    "10k": 10000,
    "20k": 20000,
    "100k": 100000,
}
GROUP_ORDER = list("ABCDEFGHIJKL")
VIEW_OPTIONS = ("Single group", "All groups", "All Countries")
SCREENSHOT_CHANNELS = ("chrome", "msedge")
CURRENT_HOLDER_TEAM_ID = "ARG"
PROBABILITY_PALETTES = {
    "prob_1": ((220, 252, 231), (22, 163, 74)),
    "prob_2": ((219, 234, 254), (37, 99, 235)),
    "prob_3": ((254, 243, 199), (217, 119, 6)),
    "prob_4": ((254, 226, 226), (220, 38, 38)),
}


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
def load_world_cup_logo_data_uri() -> str:
    """Load the local World Cup logo as a data URI for inline display and export."""
    svg_bytes = WORLD_CUP_LOGO_PATH.read_bytes()
    encoded = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


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

    merged = (
        groups.merge(
            teams.loc[:, ["team_id", "tournament_name", "canonical_name", "flag_icon_code", "group_code"]],
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
) -> pd.DataFrame:
    """Estimate group finishing probabilities from the fixture-based Monte Carlo model."""
    return simulate_group_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        group_order=GROUP_ORDER,
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
    .wc-card {
        border: 1px solid #dfe5ec;
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
        overflow: hidden;
        margin: 0.55rem 0 0.85rem;
        padding-bottom: 0.6rem;
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
        gap: 0.62rem;
        font-weight: 600;
        min-width: 0;
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
    .wc-grid .wc-name-cell .fi {
        font-size: 1.05rem;
    }
    .wc-grid .wc-name-text {
        display: none;
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
    @media (max-width: 1380px) {
        .wc-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 860px) {
        .wc-grid {
            grid-template-columns: 1fr;
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
        .wc-name-cell .fi {
            font-size: 1.28rem;
        }
        .wc-name-text {
            display: none;
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


def current_holder_cell_class(team_id: str) -> str:
    """Return the cell class for the current World Cup holder."""
    return " wc-holder-cell" if team_id == CURRENT_HOLDER_TEAM_ID else ""


def render_name_cell(flag_icon_code: str, display_name: str) -> str:
    """Render the team name cell with a flag-icons badge when a code is available."""
    safe_name = html.escape(display_name)
    if isinstance(flag_icon_code, str) and flag_icon_code:
        return (
            '<div class="wc-name-cell">'
            f'<span class="fi fi-{html.escape(flag_icon_code)}"></span>'
            f'<span class="wc-name-text">{safe_name}</span>'
            "</div>"
        )
    return f'<div class="wc-name-cell"><span class="wc-name-text">{safe_name}</span></div>'


def build_table_html(df: pd.DataFrame, title: str, include_group_column: bool = False) -> str:
    """Render one standings table as a styled HTML card."""
    column_ranges = {
        column_name: (float(df[column_name].min()), float(df[column_name].max()))
        for column_name in ("prob_1", "prob_2", "prob_3", "prob_4")
    }
    headers = []
    if include_group_column:
        headers.append('<th class="wc-group-col">Group</th>')
    headers.extend(
        [
            "<th>Country</th>",
            '<th class="wc-num">World Rank</th>',
            '<th class="wc-num">Elo</th>',
            '<th class="wc-num">1st %</th>',
            '<th class="wc-num">2nd %</th>',
            '<th class="wc-num">3rd %</th>',
            '<th class="wc-num">4th %</th>',
        ]
    )

    body_rows = []
    for row in df.itertuples(index=False):
        cells = []
        if include_group_column:
            cells.append(f'<td class="wc-group-col"><span class="wc-group-pill">{html.escape(str(row.group_code))}</span></td>')
        cells.extend(
            [
                f'<td class="{current_holder_cell_class(row.team_id).strip()}">{render_name_cell(row.flag_icon_code, row.display_name)}</td>',
                f'<td class="wc-num">{int(row.world_rank)}</td>',
                f'<td class="wc-num">{int(row.elo_rating)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_1", row.prob_1, *column_ranges["prob_1"])}">{format_percent(row.prob_1)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_2", row.prob_2, *column_ranges["prob_2"])}">{format_percent(row.prob_2)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_3", row.prob_3, *column_ranges["prob_3"])}">{format_percent(row.prob_3)}</td>',
                f'<td class="wc-num wc-prob" style="{probability_cell_style("prob_4", row.prob_4, *column_ranges["prob_4"])}">{format_percent(row.prob_4)}</td>',
            ]
        )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    group_pill = ""
    if title.startswith("Group "):
        group_pill = f'<span class="wc-group-pill">{html.escape(title.split()[-1])}</span>'
    card_title = html.escape(title)
    return textwrap.dedent(
        f"""
        <div class="wc-card">
          <div class="wc-card-header">
            <div>
              <div class="wc-card-subtitle">Pre-Tournament Probability Table</div>
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


def group_table_frame(df: pd.DataFrame, group_code: str) -> pd.DataFrame:
    """Return one group sorted by projected finish strength."""
    group_df = df[df["group_code"] == group_code].copy()
    return group_df.sort_values(["prob_1", "elo_rating", "world_rank"], ascending=[False, False, True])


def all_teams_table_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full team table sorted globally by projected chance of finishing 1st."""
    return df.sort_values(["prob_1", "elo_rating", "world_rank"], ascending=[False, False, True])


def current_view_tables(df: pd.DataFrame, view_mode: str, selected_group: str) -> list[dict[str, object]]:
    """Describe the tables needed for the active dashboard view."""
    if view_mode == "Single group":
        return [
            {
                "title": f"Group {selected_group}",
                "stem": f"group_{selected_group.lower()}",
                "frame": group_table_frame(df, selected_group),
                "include_group_column": False,
            }
        ]
    if view_mode == "All groups":
        tables = []
        for group_code in GROUP_ORDER:
            group_df = group_table_frame(df, group_code)
            if group_df.empty:
                continue
            tables.append(
                {
                    "title": f"Group {group_code}",
                    "stem": f"group_{group_code.lower()}",
                    "frame": group_df,
                    "include_group_column": False,
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
        }
    ]


def render_tables(tables: list[dict[str, object]], multi_column: bool) -> None:
    """Render one or many HTML tables into the Streamlit dashboard."""
    if multi_column:
        grid_html = "".join(
            build_table_html(
                table["frame"],
                table["title"],
                include_group_column=table["include_group_column"],
            )
            for table in tables
        )
        st.markdown(f'<div class="wc-grid">{grid_html}</div>', unsafe_allow_html=True)
        return

    for table in tables:
        st.markdown(
            build_table_html(
                table["frame"],
                table["title"],
                include_group_column=table["include_group_column"],
            ),
            unsafe_allow_html=True,
        )


def render_export_document(page_title: str, tables: list[dict[str, object]], multi_column: bool) -> str:
    """Render a complete standalone HTML document for export."""
    container_class = "wc-grid" if multi_column else "wc-grid-single"
    tables_html = "".join(
        build_table_html(
            table["frame"],
            table["title"],
            include_group_column=table["include_group_column"],
        )
        for table in tables
    )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>{shared_css()}</style>
</head>
<body>
  <div class="wc-export-page">
    <div class="{container_class}">
      {tables_html}
    </div>
  </div>
</body>
</html>
"""
    return document


def build_export_stem(filename_stem: str, export_suffix: str | None = None) -> str:
    """Build the export filename stem, adding a unique suffix when requested."""
    return filename_stem if not export_suffix else f"{filename_stem}_{export_suffix}"


def generate_export_suffix() -> str:
    """Generate a timestamp suffix so each export writes a fresh artifact."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def export_document_png(
    filename_stem: str,
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    export_suffix: str | None = None,
) -> Path:
    """Export a complete standalone HTML view as a PNG screenshot."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = render_export_document(page_title, tables, multi_column)
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    output_path = EXPORT_DIR / f"{output_stem}.png"

    with tempfile.TemporaryDirectory(prefix="wc_export_", dir=str(EXPORT_DIR)) as temp_dir:
        temp_html_path = Path(temp_dir) / f"{output_stem}.html"
        temp_html_path.write_text(document, encoding="utf-8")
        page_url = temp_html_path.resolve().as_uri()

        last_error = ""
        for channel in SCREENSHOT_CHANNELS:
            command = [
                "playwright.exe",
                "screenshot",
                "--full-page",
                "--wait-for-timeout",
                "1500",
                "--channel",
                channel,
                page_url,
                str(output_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return output_path
            except FileNotFoundError:
                last_error = "playwright.exe was not found on PATH."
                break
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or exc.stdout or str(exc)).strip()

        raise RuntimeError(f"PNG export failed: {last_error}")


def export_current_view(view_mode: str, selected_group: str, tables: list[dict[str, object]]) -> Path:
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
    return export_document_png(
        "all_Countries_view",
        "All Countries View",
        tables,
        multi_column=False,
        export_suffix=export_suffix,
    )


def export_all_tables(df: pd.DataFrame) -> list[Path]:
    """Export every individual group table plus the combined all-Countries table as PNG files."""
    exported_paths: list[Path] = []
    export_suffix = generate_export_suffix()
    for group_code in GROUP_ORDER:
        group_df = group_table_frame(df, group_code)
        if group_df.empty:
            continue
        exported_paths.append(
            export_document_png(
                f"group_{group_code.lower()}",
                f"Group {group_code}",
                [{"title": f"Group {group_code}", "frame": group_df, "include_group_column": False}],
                multi_column=False,
                export_suffix=export_suffix,
            )
        )

    combined = all_teams_table_frame(df)
    exported_paths.append(
        export_document_png(
            "all_Countries",
            "All Countries",
            [{"title": "All Countries", "frame": combined, "include_group_column": True}],
            multi_column=False,
            export_suffix=export_suffix,
        )
    )
    return exported_paths


def main() -> None:
    """Configure the page, prepare the data, and render the exportable HTML dashboard."""
    st.set_page_config(page_title="World Cup 2026 Group Dashboard", layout="wide")
    inject_styles()

    base_df, fixtures_df, lead_in_df, metadata = load_data()
    world_cup_logo_data_uri = load_world_cup_logo_data_uri()
    simulation_label = st.radio(
        "Simulation runs",
        tuple(SIMULATION_OPTIONS.keys()),
        index=1,
        horizontal=True,
    )
    simulation_count = SIMULATION_OPTIONS[simulation_label]
    dashboard_df = simulate_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulation_count,
    )

    st.markdown(
        f"""
        <div class="wc-header">
          <div class="wc-header-bar">
            <img class="wc-title-logo" src="{world_cup_logo_data_uri}" alt="FIFA World Cup 2026 logo" />
            <div>
              <div class="wc-kicker">Pre-Tournament Predictions</div>
              <h1 style="margin:0;">World Cup 2026 Group Dashboard</h1>
              <div class="wc-meta">
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
    render_countdown_timer(fixtures_df)

    st.caption(
        "Probabilities come from a fixture-by-fixture group simulation using the real 2026 schedule, "
        "a blended Elo/FIFA baseline, and each country's last eight pre-tournament results. "
        "The percentage cells use color gradients so stronger finish likelihoods read more quickly."
    )

    view_mode = st.radio("View", VIEW_OPTIONS, horizontal=True)
    selected_group = st.selectbox("Group", GROUP_ORDER, index=0) if view_mode == "Single group" else GROUP_ORDER[0]
    tables = current_view_tables(dashboard_df, view_mode, selected_group)
    multi_column = view_mode == "All groups"

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Export This View", use_container_width=True):
            try:
                export_path = export_current_view(view_mode, selected_group, tables)
                st.success(f"Exported current view to {export_path}")
            except RuntimeError as exc:
                st.error(str(exc))
    with action_cols[1]:
        if st.button("Export All Tables", use_container_width=True):
            try:
                exported_paths = export_all_tables(dashboard_df)
                st.success(f"Exported {len(exported_paths)} PNG tables to {EXPORT_DIR}")
            except RuntimeError as exc:
                st.error(str(exc))

    render_tables(tables, multi_column=multi_column)


if __name__ == "__main__":
    main()
