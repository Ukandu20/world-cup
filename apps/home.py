import html
from pathlib import Path
import subprocess
import tempfile
import textwrap

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "INT-World Cup" / "world_cup" / "2026"
EXPORT_DIR = ROOT / "assets" / "charts" / "generated"
SIMULATION_COUNT = 20000
GROUP_ORDER = list("ABCDEFGHIJKL")
VIEW_OPTIONS = ("Single group", "All groups", "All teams")
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
def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and combine the group, team, FIFA ranking, and Elo datasets."""
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    groups = pd.read_csv(DATA_DIR / "groups.csv")
    fifa = pd.read_csv(DATA_DIR / "fifa_rank_snapshots.csv")
    elo = pd.read_csv(DATA_DIR / "elo_snapshots.csv")
    manifest = pd.read_json(DATA_DIR / "manifest.json", typ="series").to_dict()

    text_columns = ["canonical_name", "tournament_name"]
    for frame in (teams, groups, fifa, elo):
        for column in text_columns:
            if column in frame.columns:
                frame[column] = frame[column].map(fix_mojibake)
    groups["team_name"] = groups["team_name"].map(fix_mojibake)

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
    return merged, metadata


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a numeric series while handling empty or constant inputs safely."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


@st.cache_data(show_spinner=False)
def simulate_probabilities(base_df: pd.DataFrame, simulations: int = SIMULATION_COUNT) -> pd.DataFrame:
    """Estimate group finishing probabilities from a rating-weighted Monte Carlo model."""
    df = base_df.copy()
    df["strength_score"] = 0.65 * zscore(df["elo_rating"]) + 0.35 * zscore(df["fifa_points"])

    results: list[pd.DataFrame] = []
    for group_code in GROUP_ORDER:
        group = df[df["group_code"] == group_code].copy()
        if group.empty:
            continue

        scores = group["strength_score"].to_numpy(dtype=float)
        noise = np.random.default_rng(20260403 + ord(group_code)).gumbel(size=(simulations, len(group)))
        finish_order = np.argsort(-(scores + noise), axis=1)

        probabilities = {}
        for place in range(len(group)):
            counts = np.bincount(finish_order[:, place], minlength=len(group))
            probabilities[f"prob_{place + 1}"] = counts / simulations * 100

        probability_frame = pd.DataFrame(probabilities)
        probability_frame["team_id"] = group["team_id"].to_numpy()
        probability_frame["group_code"] = group_code
        results.append(probability_frame)

    probabilities_df = pd.concat(results, ignore_index=True)
    return df.merge(probabilities_df, on=["team_id", "group_code"], how="left")


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
            "<th>Name</th>",
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
            "title": "All Teams",
            "stem": "all_teams",
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


def export_document_png(filename_stem: str, page_title: str, tables: list[dict[str, object]], multi_column: bool) -> Path:
    """Export a complete standalone HTML view as a PNG screenshot."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = render_export_document(page_title, tables, multi_column)
    output_path = EXPORT_DIR / f"{filename_stem}.png"

    with tempfile.TemporaryDirectory(prefix="wc_export_", dir=str(EXPORT_DIR)) as temp_dir:
        temp_html_path = Path(temp_dir) / f"{filename_stem}.html"
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
    if view_mode == "Single group":
        return export_document_png(f"group_{selected_group.lower()}_view", f"Group {selected_group} View", tables, multi_column=False)
    if view_mode == "All groups":
        return export_document_png("all_groups_view", "All Groups View", tables, multi_column=True)
    return export_document_png("all_teams_view", "All Teams View", tables, multi_column=False)


def export_all_tables(df: pd.DataFrame) -> list[Path]:
    """Export every individual group table plus the combined all-teams table as PNG files."""
    exported_paths: list[Path] = []
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
            )
        )

    combined = all_teams_table_frame(df)
    exported_paths.append(
        export_document_png(
            "all_teams",
            "All Teams",
            [{"title": "All Teams", "frame": combined, "include_group_column": True}],
            multi_column=False,
        )
    )
    return exported_paths


def main() -> None:
    """Configure the page, prepare the data, and render the exportable HTML dashboard."""
    st.set_page_config(page_title="World Cup 2026 Group Dashboard", layout="wide")
    inject_styles()

    base_df, metadata = load_data()
    dashboard_df = simulate_probabilities(base_df)

    st.markdown(
        f"""
        <div class="wc-header">
          <div class="wc-kicker">Pre-Tournament Predictions</div>
          <h1 style="margin:0;">World Cup 2026 Group Dashboard</h1>
          <div class="wc-meta">
            Build date: {html.escape(str(metadata["build_date"]))} |
            FIFA snapshot: {html.escape(str(metadata["fifa_snapshot_date"]))} |
            Elo snapshot: {html.escape(str(metadata["elo_snapshot_date"]))} |
            Simulations per group: {SIMULATION_COUNT:,}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Probabilities are Pre-Tournament estimates from a rating-weighted simulation using Elo rating and FIFA points. "
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
