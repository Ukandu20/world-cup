
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "INT-World Cup" / "world_cup" / "2026"
SIMULATION_COUNT = 20000
GROUP_ORDER = list("ABCDEFGHIJKL")
VIEW_OPTIONS = ("Single group", "All groups", "All teams")


def fix_mojibake(value: str) -> str:
    if not isinstance(value, str):
        return value
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    if all(marker not in value for marker in ("Ã", "Â")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


fix_mojibake = normalize_text


def decode_display_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    if all(marker not in value for marker in ("\u00c3", "\u00c2")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


fix_mojibake = decode_display_text


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
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
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


@st.cache_data(show_spinner=False)
def simulate_probabilities(base_df: pd.DataFrame, simulations: int = SIMULATION_COUNT) -> pd.DataFrame:
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


def inject_styles() -> None:
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/css/flag-icons.min.css">
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .wc-kicker {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #5b6474;
            margin-bottom: 0.35rem;
        }
        .wc-header {
            margin-bottom: 1.2rem;
        }
        .wc-meta {
            color: #5b6474;
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }
        .wc-card {
            border: 1px solid #dfe5ec;
            border-radius: 16px;
            padding: 1rem 1rem 0.8rem;
            background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
            box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
            margin-bottom: 1rem;
        }
        .wc-group-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        table.wc-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.93rem;
        }
        .wc-table th {
            text-align: left;
            color: #4b5563;
            font-weight: 600;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #dfe5ec;
            padding: 0.55rem 0.45rem;
        }
        .wc-table td {
            border-bottom: 1px solid #eef2f7;
            padding: 0.7rem 0.45rem;
            vertical-align: middle;
        }
        .wc-table tr:last-child td {
            border-bottom: none;
        }
        .wc-name {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 600;
        }
        .wc-name .fi {
            border-radius: 999px;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
        }
        .wc-num {
            text-align: right;
            white-space: nowrap;
        }
        .wc-group-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: #0f172a;
            color: white;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def render_name_cell(flag_icon_code: str, display_name: str) -> str:
    safe_name = html.escape(display_name)
    if isinstance(flag_icon_code, str) and flag_icon_code:
        return f'<div class="wc-name"><span class="fi fi-{html.escape(flag_icon_code)}"></span><span>{safe_name}</span></div>'
    return f'<div class="wc-name"><span>{safe_name}</span></div>'


def render_table(df: pd.DataFrame, title: str | None = None, include_group_column: bool = False) -> str:
    headers = []
    if include_group_column:
        headers.append("<th>Group</th>")
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

    rows = []
    for row in df.itertuples(index=False):
        cells = []
        if include_group_column:
            cells.append(f'<td><span class="wc-group-chip">{html.escape(str(row.group_code))}</span></td>')
        cells.extend(
            [
                f"<td>{render_name_cell(row.flag_icon_code, row.display_name)}</td>",
                f'<td class="wc-num">{int(row.world_rank)}</td>',
                f'<td class="wc-num">{int(row.elo_rating)}</td>',
                f'<td class="wc-num">{format_percent(row.prob_1)}</td>',
                f'<td class="wc-num">{format_percent(row.prob_2)}</td>',
                f'<td class="wc-num">{format_percent(row.prob_3)}</td>',
                f'<td class="wc-num">{format_percent(row.prob_4)}</td>',
            ]
        )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    title_html = f'<div class="wc-group-title">{html.escape(title)}</div>' if title else ""
    return f"""
    <div class="wc-card">
      {title_html}
      <table class="wc-table">
        <thead><tr>{''.join(headers)}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def group_table_frame(df: pd.DataFrame, group_code: str) -> pd.DataFrame:
    group_df = df[df["group_code"] == group_code].copy()
    return group_df.sort_values(["prob_1", "elo_rating", "world_rank"], ascending=[False, False, True])


def render_single_group_view(df: pd.DataFrame) -> None:
    selected_group = st.selectbox("Group", GROUP_ORDER, index=0)
    group_df = group_table_frame(df, selected_group)
    st.markdown(render_table(group_df, title=f"Group {selected_group}"), unsafe_allow_html=True)


def render_all_groups_view(df: pd.DataFrame) -> None:
    columns = st.columns(3)
    for index, group_code in enumerate(GROUP_ORDER):
        group_df = group_table_frame(df, group_code)
        if group_df.empty:
            continue
        with columns[index % 3]:
            st.markdown(render_table(group_df, title=f"Group {group_code}"), unsafe_allow_html=True)


def render_all_teams_view(df: pd.DataFrame) -> None:
    combined = df.sort_values(["group_code", "prob_1", "elo_rating"], ascending=[True, False, False])
    st.markdown(render_table(combined, include_group_column=True), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="World Cup 2026 Preseason Dashboard", layout="wide")
    inject_styles()

    base_df, metadata = load_data()
    dashboard_df = simulate_probabilities(base_df)

    st.markdown(
        f"""
        <div class="wc-header">
          <div class="wc-kicker">Preseason Predictions</div>
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
        "Probabilities are preseason estimates from a rating-weighted simulation using Elo rating and FIFA points. "
        "Each group table shows the chance of finishing 1st through 4th."
    )

    view_mode = st.radio("View", VIEW_OPTIONS, horizontal=True)

    if view_mode == "Single group":
        render_single_group_view(dashboard_df)
    elif view_mode == "All groups":
        render_all_groups_view(dashboard_df)
    else:
        render_all_teams_view(dashboard_df)


if __name__ == "__main__":
    main()
