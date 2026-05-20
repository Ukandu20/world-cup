from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_cup_sim.analysis import (  # noqa: E402
    CONFEDERATION_ORDER,
    ERA_LABELS,
    build_2026_implication_tables,
    build_correlation_metrics,
    build_data_quality_summary,
    build_goal_metrics,
    build_host_effect_metrics,
    build_participation_metrics,
    build_winner_followup_metrics,
    load_historical_world_cup_data,
)


CONFEDERATION_COLORS = {
    "UEFA": "#1A56DB",
    "CAF": "#F5A623",
    "AFC": "#E02020",
    "CONMEBOL": "#27AE60",
    "CONCACAF": "#8E44AD",
    "OFC": "#17A8CD",
}

ERA_COLORS = {
    "Early Era": "#7A4E2D",
    "Golden Age": "#C99700",
    "Modern Era": "#2F6F73",
    "Contemporary": "#8A3FFC",
    "Recent": "#D1495B",
}

FIELD_SIZE_COLORS = {
    "<16-teams": "#C9DCF0",
    "16-teams": "#8AAFD4",
    "24-teams": "#4D7FB5",
    "32-teams": "#1E4D82",
    "48-teams": "#0A2040",
}

ALL_CONFEDERATION_FILTER = "All confederations"
QUALIFICATION_CYCLE_START = pd.Timestamp("2022-12-19")
QUALIFICATION_PLAYOFF_START = pd.Timestamp("2026-03-26")
QUALIFICATION_PLAYOFF_END = pd.Timestamp("2026-03-31")
QUALIFIER_SCORE_WEIGHTS = {
    "points_per_match": 0.35,
    "goal_difference_per_match": 0.25,
    "goals_for_per_match": 0.15,
    "defensive_score": 0.15,
    "elo_change_per_match": 0.10,
}
COUNTRY_NAME_ALIASES = {
    "China PR": "China",
    "Czechia": "Czechoslovakia",
    "DR Congo": "Zaire",
    "German DR": "Germany",
    "Serbia and Montenegro": "Serbia",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
}

PLACEMENT_SHORT_LABELS = {
    "Winner": "W",
    "Runner-up": "F",
    "Third Place": "3P",
    "Fourth Place": "4P",
    "Semi-final": "SF",
    "Quarter-final": "QF",
    "Round of 16": "R16",
    "Group Stage": "GS",
    "DNQ": "DNQ",
}
GOALS_STAGE_ORDER = ["Group Stage", "Round of 16", "Quarter-final", "Semi-final", "Third Place", "Final"]
GOALS_KNOCKOUT_STAGES = ["Round of 16", "Quarter-final", "Semi-final", "Third Place", "Final"]

CHART_BACKGROUND = "#EFE3CF"
CHART_TEXT_COLOR = "#3A2A1A"
CHART_AXIS_COLOR = "#5A4632"
CHART_POSITIVE_COLOR = "#2F6F3E"
CHART_NEGATIVE_COLOR = "#B23A30"
CHART_FONT_FAMILY = "Gill Sans, sans-serif"
PLOTLY_EXPORT_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 600,
        "width": 1000,
        "scale": 3,
    }
}
SOURCE_NOTE = "Data Source: Kaggle | @cartierkut1"
TITLE_PREFIX = "FIFA Men's World Cup"

PRE_TOURNAMENT_FEATURES = [
    "start_elo",
    "elo_rank",
    "is_host",
    "prior_world_cup_participations",
    "previous_finish_score",
    "previous_position",
    "prior_avg_finish_score",
    "prior_best_finish_score",
    "prior_avg_goals_per_match",
    "prior_avg_goal_diff_per_match",
    "form_l10_matches",
    "form_l10_win_pct",
    "form_l10_goals_for",
    "form_l10_goals_against",
    "form_l10_goal_difference",
    "form_l10_goals_per_match",
    "form_l10_goals_against_per_match",
    "form_l10_goal_difference_per_match",
    "form_l10_elo_change",
    "form_l10_elo_change_per_match",
    "weighted_form_l10_result_score",
    "weighted_form_l10_win_pct",
    "weighted_form_l10_goals_for_per_match",
    "weighted_form_l10_goals_against_per_match",
    "weighted_form_l10_goal_difference_per_match",
    "weighted_form_l10_elo_change_per_match",
]

IN_TOURNAMENT_FEATURES = [
    "matches_played",
    "gs",
    "ga",
    "goal_difference",
    "goals_per_match",
    "goals_against_per_match",
    "goal_difference_per_match",
    "finish_elo",
    "elo_change",
]

LAST_K_FEATURES = [
    "last_k_appearances",
    "last_k_appearance_rate",
    "last_k_avg_finish_score",
    "last_k_best_finish_score",
    "last_k_avg_position",
    "last_k_goals_for",
    "last_k_goals_against",
    "last_k_goal_difference",
    "last_k_goals_per_match",
    "last_k_goals_against_per_match",
    "last_k_goal_difference_per_match",
    "last_k_elo_change",
    "last_k_elo_change_per_appearance",
    "weighted_last_k_appearance_rate",
    "weighted_last_k_finish_score",
    "weighted_last_k_position",
    "weighted_last_k_goals_per_match",
    "weighted_last_k_goals_against_per_match",
    "weighted_last_k_goal_difference_per_match",
    "weighted_last_k_elo_change_per_appearance",
]

FEATURE_LABELS = {
    "start_elo": "Starting Elo",
    "elo_rank": "Elo Rank",
    "is_host": "Host",
    "prior_world_cup_participations": "Prior WC Participations",
    "previous_finish_score": "Previous Finish Score",
    "previous_position": "Previous Position",
    "prior_avg_finish_score": "Prior Avg Finish Score",
    "prior_best_finish_score": "Prior Best Finish Score",
    "prior_avg_goals_per_match": "Prior Avg Goals per Match",
    "prior_avg_goal_diff_per_match": "Prior Avg Goal Diff per Match",
    "form_l10_matches": "Last 10 Matches",
    "form_l10_win_pct": "Last 10 Win Pct",
    "form_l10_goals_for": "Last 10 Goals For",
    "form_l10_goals_against": "Last 10 Goals Against",
    "form_l10_goal_difference": "Last 10 Goal Difference",
    "form_l10_goals_per_match": "Last 10 Goals per Match",
    "form_l10_goals_against_per_match": "Last 10 Goals Against per Match",
    "form_l10_goal_difference_per_match": "Last 10 Goal Difference per Match",
    "form_l10_elo_change": "Last 10 Elo Change",
    "form_l10_elo_change_per_match": "Last 10 Elo Change per Match",
    "weighted_form_l10_result_score": "Weighted Last 10 Result Score",
    "weighted_form_l10_win_pct": "Weighted Last 10 Win Pct",
    "weighted_form_l10_goals_for_per_match": "Weighted Last 10 Goals per Match",
    "weighted_form_l10_goals_against_per_match": "Weighted Last 10 Goals Against per Match",
    "weighted_form_l10_goal_difference_per_match": "Weighted Last 10 Goal Difference per Match",
    "weighted_form_l10_elo_change_per_match": "Weighted Last 10 Elo Change per Match",
    "matches_played": "Matches Played",
    "gs": "Goals For",
    "ga": "Goals Against",
    "goal_difference": "Goal Difference",
    "goals_per_match": "Goals per Match",
    "goals_against_per_match": "Goals Against per Match",
    "goal_difference_per_match": "Goal Difference per Match",
    "finish_elo": "Finish Elo",
    "elo_change": "Elo Change",
    "prior_appearance_rate": "Last-k Appearance Rate",
    "last_k_appearances": "Last-k Appearances",
    "last_k_appearance_rate": "Last-k Appearance Rate",
    "last_k_avg_finish_score": "Last-k Avg Finish Score",
    "last_k_best_finish_score": "Last-k Best Finish Score",
    "last_k_avg_position": "Last-k Avg Position",
    "last_k_goals_for": "Last-k Goals For",
    "last_k_goals_against": "Last-k Goals Against",
    "last_k_goal_difference": "Last-k Goal Difference",
    "last_k_goals_per_match": "Last-k Goals per Match",
    "last_k_goals_against_per_match": "Last-k Goals Against per Match",
    "last_k_goal_difference_per_match": "Last-k Goal Difference per Match",
    "last_k_elo_change": "Last-k Elo Change",
    "last_k_elo_change_per_appearance": "Last-k Elo Change per Appearance",
    "weighted_last_k_appearance_rate": "Weighted Last-k Appearance Rate",
    "weighted_last_k_finish_score": "Weighted Last-k Finish Score",
    "weighted_last_k_position": "Weighted Last-k Position",
    "weighted_last_k_goals_per_match": "Weighted Last-k Goals per Match",
    "weighted_last_k_goals_against_per_match": "Weighted Last-k Goals Against per Match",
    "weighted_last_k_goal_difference_per_match": "Weighted Last-k Goal Difference per Match",
    "weighted_last_k_elo_change_per_appearance": "Weighted Last-k Elo Change per Appearance",
    "finish_score": "Finish Score",
    "current_finish_score": "Current Finish Score",
}


@st.cache_data(show_spinner=False)
def load_historical_eda_data() -> dict[str, pd.DataFrame]:
    return load_historical_world_cup_data()


@st.cache_data(show_spinner=False)
def compute_historical_eda_outputs(
    lookback: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    datasets = load_historical_eda_data()
    return (
        build_data_quality_summary(datasets),
        datasets["placement"].copy(),
        build_participation_metrics(datasets),
        build_goal_metrics(datasets),
        build_host_effect_metrics(datasets),
        build_winner_followup_metrics(datasets),
        build_correlation_metrics(datasets, lookback=lookback),
        build_2026_implication_tables(datasets),
    )


def is_world_cup_qualification_tournament(tournament: object) -> bool:
    label = str(tournament or "").strip().lower()
    if not label:
        return False
    if "world cup qualification" in label:
        return True
    return "world cup" in label and (
        "playoff" in label or "play-off" in label or "inter-confederation" in label
    )


def qualifier_stage_label(tournament: object, match_date: object | None = None) -> str:
    date_value = pd.to_datetime(match_date, errors="coerce")
    if pd.notna(date_value) and QUALIFICATION_PLAYOFF_START <= date_value <= QUALIFICATION_PLAYOFF_END:
        return "Qualifier playoffs"
    label = str(tournament or "").strip().lower()
    return "Qualifier playoffs" if "playoff" in label or "play-off" in label or "inter-confederation" in label else "Qualifiers"


def minmax_score(values: pd.Series, *, invert: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if numeric.empty:
        return numeric
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(0.0, index=values.index)
    if max_value == min_value:
        return pd.Series(50.0, index=values.index)
    if invert:
        return ((max_value - numeric) / (max_value - min_value) * 100).fillna(0.0)
    return ((numeric - min_value) / (max_value - min_value) * 100).fillna(0.0)


def normalize_boolean_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.upper().isin({"TRUE", "1", "YES"})


def build_qualifier_performance_tables(lead_in_df: pd.DataFrame, teams_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary_columns = [
        "team_id",
        "team",
        "confederation",
        "qualification_path",
        "matches",
        "wins",
        "draws",
        "losses",
        "points",
        "points_per_match",
        "goals_for",
        "goals_against",
        "goal_difference",
        "goals_for_per_match",
        "goals_against_per_match",
        "goal_difference_per_match",
        "elo_change",
        "elo_change_per_match",
        "performance_score",
    ]
    match_columns = [
        "date",
        "Date",
        "team_id",
        "Team",
        "Confederation",
        "Opponent",
        "Result",
        "Score",
        "Tournament",
        "Stage",
        "Venue",
        "Elo Change",
    ]
    if lead_in_df.empty or teams_df.empty:
        return {"summary": pd.DataFrame(columns=summary_columns), "matches": pd.DataFrame(columns=match_columns)}

    teams = teams_df.copy()
    teams["team_id"] = teams["team_id"].astype(str)
    team_name_column = "team" if "team" in teams.columns else "country"
    teams["team"] = teams[team_name_column].fillna(teams["team_id"]).astype(str)
    teams["confederation"] = teams.get("confederation", pd.Series("", index=teams.index)).fillna("").astype(str)
    teams["qualification_path"] = teams.get("qualification_path", pd.Series("", index=teams.index)).fillna("").astype(str)
    teams["is_host"] = normalize_boolean_series(teams.get("is_host", pd.Series(False, index=teams.index)))
    non_host_teams = teams.loc[~teams["is_host"], ["team_id", "team", "confederation", "qualification_path"]].drop_duplicates(
        "team_id"
    )
    non_host_ids = set(non_host_teams["team_id"])

    matches = lead_in_df.copy()
    matches["qualified_team_id"] = matches["qualified_team_id"].astype(str)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    tournament_mask = matches["tournament"].map(is_world_cup_qualification_tournament)
    matches = matches.loc[
        tournament_mask
        & matches["date"].ge(QUALIFICATION_CYCLE_START)
        & matches["qualified_team_id"].isin(non_host_ids)
    ].copy()
    if matches.empty:
        return {"summary": pd.DataFrame(columns=summary_columns), "matches": pd.DataFrame(columns=match_columns)}

    for column_name in ["team_score", "opponent_score", "team_elo_delta"]:
        matches[column_name] = pd.to_numeric(matches[column_name], errors="coerce").fillna(0.0)
    matches["result_normalized"] = matches["result"].fillna("").astype(str).str.lower()
    matches["points"] = matches["result_normalized"].map({"win": 3, "draw": 1, "loss": 0}).fillna(0).astype(int)
    matches["win"] = matches["result_normalized"].eq("win").astype(int)
    matches["draw"] = matches["result_normalized"].eq("draw").astype(int)
    matches["loss"] = matches["result_normalized"].eq("loss").astype(int)
    matches["goal_difference"] = matches["team_score"] - matches["opponent_score"]
    matches = matches.merge(
        non_host_teams,
        left_on="qualified_team_id",
        right_on="team_id",
        how="inner",
        validate="many_to_one",
    )

    summary = (
        matches.groupby(["team_id", "team", "confederation", "qualification_path"], as_index=False)
        .agg(
            matches=("lead_in_id", "count"),
            wins=("win", "sum"),
            draws=("draw", "sum"),
            losses=("loss", "sum"),
            points=("points", "sum"),
            goals_for=("team_score", "sum"),
            goals_against=("opponent_score", "sum"),
            goal_difference=("goal_difference", "sum"),
            elo_change=("team_elo_delta", "sum"),
        )
        .sort_values(["points", "goal_difference", "goals_for"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    for numerator, output_column in [
        ("points", "points_per_match"),
        ("goals_for", "goals_for_per_match"),
        ("goals_against", "goals_against_per_match"),
        ("goal_difference", "goal_difference_per_match"),
        ("elo_change", "elo_change_per_match"),
    ]:
        summary[output_column] = (summary[numerator] / summary["matches"].replace(0, pd.NA)).astype("Float64")

    component_scores = {
        "points_per_match": minmax_score(summary["points_per_match"]),
        "goal_difference_per_match": minmax_score(summary["goal_difference_per_match"]),
        "goals_for_per_match": minmax_score(summary["goals_for_per_match"]),
        "defensive_score": minmax_score(summary["goals_against_per_match"], invert=True),
        "elo_change_per_match": minmax_score(summary["elo_change_per_match"]),
    }
    summary["performance_score"] = sum(
        component_scores[column_name] * weight for column_name, weight in QUALIFIER_SCORE_WEIGHTS.items()
    ).round(1)
    numeric_summary_columns = [
        "points_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "goal_difference_per_match",
        "elo_change",
        "elo_change_per_match",
    ]
    for column_name in numeric_summary_columns:
        summary[column_name] = pd.to_numeric(summary[column_name], errors="coerce").round(3)
    for column_name in ["goals_for", "goals_against", "goal_difference"]:
        summary[column_name] = pd.to_numeric(summary[column_name], errors="coerce").round(0).astype(int)
    summary = summary.sort_values(
        ["performance_score", "points_per_match", "goal_difference_per_match"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    matches["Date"] = matches["date"].dt.strftime("%Y-%m-%d")
    matches["Team"] = matches["team"]
    matches["Confederation"] = matches["confederation"]
    matches["Opponent"] = matches.get("opponent_name", pd.Series("", index=matches.index)).fillna("").astype(str)
    matches["Result"] = matches["result_normalized"].map({"win": "W", "draw": "D", "loss": "L"}).fillna("")
    matches["Score"] = (
        matches["team_score"].astype(int).astype(str) + "-" + matches["opponent_score"].astype(int).astype(str)
    )
    matches["Tournament"] = matches["tournament"].fillna("").astype(str)
    matches["Stage"] = [
        qualifier_stage_label(tournament, match_date)
        for tournament, match_date in zip(matches["Tournament"], matches["date"], strict=False)
    ]
    matches["Venue"] = (
        matches.get("city", pd.Series("", index=matches.index)).fillna("").astype(str).str.strip()
        + matches.get("country", pd.Series("", index=matches.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .map(lambda value: f", {value}" if value else "")
    ).str.strip(", ")
    matches["Elo Change"] = pd.to_numeric(matches["team_elo_delta"], errors="coerce").round(1)
    matches = matches.sort_values(["date", "Team", "lead_in_id"], kind="stable").reset_index(drop=True)

    return {"summary": summary.loc[:, summary_columns], "matches": matches.loc[:, match_columns]}


@st.cache_data(show_spinner=False)
def load_qualifier_performance_tables() -> dict[str, pd.DataFrame]:
    data_root = ROOT / "data" / "processed" / "world_cup" / "2026"
    return build_qualifier_performance_tables(
        pd.read_csv(data_root / "team_results_lead_in.csv"),
        pd.read_csv(data_root / "teams.csv"),
    )


def render_metric_row(values: dict[str, object]) -> None:
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values.items()):
        column.metric(label, value)


def world_cup_chart_title(title: str) -> str:
    return title if TITLE_PREFIX in title else f"{TITLE_PREFIX} {title}"


def country_world_cup_chart_title(country: str, title: str) -> str:
    return f"{country}'s {TITLE_PREFIX} {title}"


def chart_caption_from_title(title: object) -> str | None:
    raw_title = str(title or "").strip()
    if not raw_title:
        return None
    clean_title = re.sub(r"<[^>]+>", "", raw_title)
    clean_title = " ".join(clean_title.split())
    base_title = clean_title.replace(TITLE_PREFIX, "").strip()

    caption_by_title = {
        "Tournament Size by Edition": "The above chart tracks how the FIFA Men's World Cup tournament size changed by over the decades; shaded eras and expansion markers show where format changes affect comparisons.",
        "Participation by Confederation": "Compares the number of countries represented in each confederation at every tournament, highlighting expansion effects across eras.",
        "Debutants by Edition": "Shows how many nations made their first World Cup appearance in each edition.",
        "Match Scoreline Distribution by Round": "Groups match scorelines by total goals on the y-axis; each point is still an individual match with exact scoreline details in hover.",
        "Host Nation Finishes": "Shows how host countries finished, with point size reflecting goals for and color identifying confederation.",
        "Champion Follow-up Performance": "Tracks how each champion performed at the next World Cup, including title defenses and failed qualifications.",
        "Pre-Tournament Feature Correlation with World Cup Finish Score": "Ranks leakage-safe pre-tournament indicators by Spearman correlation with normalized finish score.",
        "In-Tournament Stat Correlation with World Cup Finish Score": "Shows how tournament performance stats relate to final finish; these explain outcomes rather than predict them beforehand.",
        "Spearman Correlation Heatmap: Outcome and Predictors": "Displays pairwise Spearman correlations among finish score and pre-tournament predictors.",
        "Spearman Correlation Heatmap: Outcome and Tournament Stats": "Displays pairwise Spearman correlations among finish score and in-tournament performance stats.",
        "2026 Qualifier Performance Score": "Ranks qualified teams by lead-in qualifier performance using points, goal difference, and Elo movement.",
        "2026 Qualifier Attack vs Defense": "Places teams by qualifier scoring and concession rates; larger points indicate stronger points-per-match records.",
        "2026 Confederation Share": "Compares confederation representation in the expanded 2026 field.",
    }
    if base_title in caption_by_title:
        return caption_by_title[base_title]
    if "Placement by Edition" in base_title:
        return "Shows the selected country's placements over time."
    if "Goals Scored per Game" in base_title:
        return "Shows the selected country's scoring rate by edition, adjusted for how many matches it played."
    if "Goals Conceded per Game" in base_title:
        return "Shows the selected country's defensive record by edition, adjusted for how many matches it played."
    if "Goals Scored" in base_title:
        return "Shows the selected country's goals for by edition, with expansion markers for tournament format context."
    if "Goals Conceded" in base_title:
        return "Shows the selected country's goals against by edition, with expansion markers for tournament format context."
    if "Tournament Goals per Match" in base_title:
        return "Tracks scoring rate by tournament, which is more comparable across editions than raw goal totals."
    if "Tournament Total Goals" in base_title:
        return "Tracks total goals by tournament; tournament size and match count changes should be considered when comparing eras."
    if "Team Distribution" in base_title:
        return "Breaks down the selected edition's team field by confederation and country."
    if "Pre-Tournament Predictors + Last-" in base_title:
        return "Combines baseline pre-tournament predictors with recent World Cup history to compare their relationship with finish score."
    if "Last-" in base_title and "World Cup History Correlation" in base_title:
        return "Ranks recent World Cup history features by their Spearman correlation with current tournament finish score."
    if "Outcome and Last-" in base_title:
        return "Shows pairwise correlations among finish score and recent World Cup history features."
    if "Outcome, Baseline Predictors, and Last-" in base_title:
        return "Shows how baseline predictors and recent World Cup history features correlate with finish score and with each other."
    if " vs Finish Score" in base_title:
        return "Plots each team-edition observation to show the relationship between this feature and normalized finish score."
    if "2026 Qualifier Goals For" in base_title:
        return "Ranks qualified teams by attacking output during 2026 qualifying and playoff lead-in matches."
    if "2026 Qualifier Goals Against" in base_title:
        return "Ranks qualified teams by defensive record during 2026 qualifying and playoff lead-in matches."
    return None


def apply_original_chart_style(fig, title: str, height: int = 560):
    title = world_cup_chart_title(title)
    fig.update_layout(
        height=height,
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20, "color": CHART_TEXT_COLOR},
        },
        font={
            "family": CHART_FONT_FAMILY,
            "size": 11,
            "color": CHART_TEXT_COLOR,
        },
        margin={"l": 40, "r": 40, "t": 70, "b": 60},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        legend={"font": {"color": CHART_TEXT_COLOR}, "title": {"font": {"color": CHART_TEXT_COLOR}}},
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "linecolor": CHART_AXIS_COLOR,
            "tickcolor": CHART_AXIS_COLOR,
            "tickfont": {"size": 13, "color": CHART_AXIS_COLOR},
            "title": {"font": {"color": CHART_TEXT_COLOR}},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#D8C8AF",
            "zeroline": False,
            "linecolor": CHART_AXIS_COLOR,
            "tickcolor": CHART_AXIS_COLOR,
            "tickfont": {"size": 13, "color": CHART_AXIS_COLOR},
            "title": {"font": {"color": CHART_TEXT_COLOR}},
        },
    )
    fig.add_annotation(
        text=SOURCE_NOTE,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.14,
        showarrow=False,
        font={"size": 10, "color": CHART_AXIS_COLOR},
    )
    return fig


def render_plotly_chart(fig, key: str | None = None, caption: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG, key=key)
    resolved_caption = caption or chart_caption_from_title(fig.layout.title.text)
    if resolved_caption:
        st.caption(resolved_caption)


def render_column_plotly_chart(column, fig, caption: str | None = None) -> None:
    fig.update_layout(
        height=max(int(fig.layout.height or 560), 620),
        margin={"l": 28, "r": 18, "t": 82, "b": 72},
        title={"font": {"size": 16, "color": CHART_TEXT_COLOR}},
        font={"size": 10, "color": CHART_TEXT_COLOR},
        legend={"font": {"size": 9, "color": CHART_TEXT_COLOR}},
    )
    fig.update_xaxes(tickfont={"size": 10, "color": CHART_AXIS_COLOR}, title_standoff=8)
    fig.update_yaxes(tickfont={"size": 10, "color": CHART_AXIS_COLOR}, title_standoff=8)
    column.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG)
    resolved_caption = caption or chart_caption_from_title(fig.layout.title.text)
    if resolved_caption:
        column.caption(resolved_caption)


def edition_tick_values(frame: pd.DataFrame) -> list[int]:
    if "edition" not in frame.columns:
        return []
    return sorted(pd.to_numeric(frame["edition"], errors="coerce").dropna().astype(int).unique().tolist())


def set_edition_ticks(fig, frame: pd.DataFrame, tickangle: int = 30):
    tick_vals = edition_tick_values(frame)
    if tick_vals:
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[str(value) for value in tick_vals],
            tickangle=tickangle,
        )
    return fig


def canonical_country_name(country: object) -> str:
    country_name = str(country)
    return COUNTRY_NAME_ALIASES.get(country_name, country_name)


def canonical_placement_label(placement: object) -> str:
    placement_text = str(placement)
    return "" if placement_text == "nan" else placement_text


def prepare_country_goal_metrics(team_goals: pd.DataFrame) -> pd.DataFrame:
    goals = team_goals.copy()
    goals["country"] = goals["country"].map(canonical_country_name)
    goals["gf"] = pd.to_numeric(goals["gf"], errors="coerce").fillna(0)
    goals["ga"] = pd.to_numeric(goals["ga"], errors="coerce").fillna(0)
    goals["team_matches"] = pd.to_numeric(goals["team_matches"], errors="coerce").fillna(0)
    goals["position"] = pd.to_numeric(goals["position"], errors="coerce")
    goals = goals.sort_values(["country", "edition", "position"], na_position="last")
    grouped = (
        goals.groupby(["edition", "era", "tournament_id", "country"], dropna=False, as_index=False, observed=True)
        .agg(
            gf=("gf", "sum"),
            ga=("ga", "sum"),
            team_matches=("team_matches", "sum"),
            position=("position", "min"),
            placement=("placement", lambda series: canonical_placement_label(series.dropna().iloc[0]) if not series.dropna().empty else ""),
        )
        .sort_values(["country", "edition"])
        .reset_index(drop=True)
    )
    grouped["goal_difference"] = grouped["gf"] - grouped["ga"]
    grouped["goals_per_game"] = (grouped["gf"] / grouped["team_matches"].replace(0, pd.NA)).astype("Float64").round(3)
    grouped["goals_against_per_game"] = (
        grouped["ga"] / grouped["team_matches"].replace(0, pd.NA)
    ).astype("Float64").round(3)
    return grouped


def prepare_country_placement_metrics(placement: pd.DataFrame) -> pd.DataFrame:
    placement_history = placement.copy()
    placement_history["country"] = placement_history["country"].map(canonical_country_name)
    placement_history["position"] = pd.to_numeric(placement_history["position"], errors="coerce")
    placement_history = placement_history.sort_values(["country", "edition", "position"], na_position="last")
    return (
        placement_history.groupby(["edition", "era", "country"], as_index=False, observed=True)
        .agg(
            position=("position", "min"),
            placement=("placement", lambda series: canonical_placement_label(series.dropna().iloc[0]) if not series.dropna().empty else ""),
        )
        .sort_values(["country", "edition"])
        .reset_index(drop=True)
    )


def placement_axis_label(row) -> str:
    placement_name = canonical_placement_label(row.placement)
    return placement_name or f"Position {int(row.position)}"


def expansion_editions(participating: pd.DataFrame) -> pd.DataFrame:
    expansion = participating.sort_values("edition").copy()
    expansion["previous_team_counts"] = expansion["team_counts"].shift(1)
    return expansion.loc[
        expansion["previous_team_counts"].notna()
        & expansion["team_counts"].gt(expansion["previous_team_counts"])
    ].copy()


def add_expansion_markers(
    fig,
    expansion: pd.DataFrame,
    show_labels: bool = True,
):
    for row in expansion.itertuples(index=False):
        fig.add_vline(
            x=row.edition,
            line_dash="dot",
            line_color=CHART_AXIS_COLOR,
            line_width=1,
            opacity=0.65,
        )
        if not show_labels:
            continue
        fig.add_annotation(
            text=f"{int(row.team_counts)} teams",
            x=row.edition,
            y=0.985,
            xref="x",
            yref="paper",
            showarrow=False,
            textangle=0,
            xanchor="center",
            yanchor="top",
            font={"size": 10, "color": CHART_AXIS_COLOR},
            bgcolor="rgba(239,227,207,0.72)",
        )
    return fig


def add_era_backgrounds(fig, frame: pd.DataFrame):
    era_frame = frame.dropna(subset=["edition", "era"]).copy()
    if era_frame.empty:
        return fig
    era_ranges = (
        era_frame.groupby("era", observed=True)["edition"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values("min")
    )
    for row in era_ranges.itertuples(index=False):
        era_name = str(row.era)
        fill_color = ERA_COLORS.get(era_name, "#8B7355")
        fig.add_vrect(
            x0=float(row.min) - 1.8,
            x1=float(row.max) + 1.8,
            fillcolor=fill_color,
            opacity=0.12,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            text=era_name,
            x=(float(row.min) + float(row.max)) / 2,
            y=1.04,
            xref="x",
            yref="paper",
            showarrow=False,
            font={"size": 10, "color": CHART_AXIS_COLOR},
        )
    return fig


def add_country_best_finish_annotations(fig, country_goals: pd.DataFrame, y_column: str):
    if country_goals.empty or y_column not in country_goals.columns:
        return fig
    annotated = country_goals.copy()
    annotated["position"] = pd.to_numeric(annotated["position"], errors="coerce")
    annotated[y_column] = pd.to_numeric(annotated[y_column], errors="coerce")
    annotated = annotated.dropna(subset=["edition", y_column])
    winners = annotated.loc[annotated["placement"].eq("Winner")].copy()
    if not winners.empty:
        rows_to_annotate = winners.sort_values("edition")
        label_for_row = lambda row: "Winner"
    else:
        finish_rows = annotated.dropna(subset=["position"]).sort_values(["position", "edition"])
        if finish_rows.empty:
            return fig
        rows_to_annotate = finish_rows.head(1)
        label_for_row = lambda row: f"Best finish: {row.placement}"

    for row in rows_to_annotate.itertuples(index=False):
        fig.add_annotation(
            text=label_for_row(row),
            x=row.edition,
            y=getattr(row, y_column),
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=CHART_AXIS_COLOR,
            ax=0,
            ay=-45,
            bgcolor=CHART_BACKGROUND,
            bordercolor=CHART_AXIS_COLOR,
            borderwidth=1,
            font={"size": 11, "color": CHART_TEXT_COLOR},
        )
    return fig


def format_chart_value(value: object, value_format: str) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return ""
    return format(float(numeric_value), value_format)


def render_country_goal_line(
    country_goals: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    value_format: str,
    expansion: pd.DataFrame,
    trace_color: str,
    compact: bool = False,
):
    display_title = world_cup_chart_title(title)
    text_values = country_goals[y_column].map(lambda value: format_chart_value(value, value_format))
    fig = px.line(
        country_goals,
        x="edition",
        y=y_column,
        markers=True,
        text=text_values,
        hover_data={"era": True, "placement": True, "position": True, "team_matches": True, y_column: ":.3f"},
        labels={"edition": "Edition", y_column: y_label},
        title=display_title,
    )
    fig.update_traces(
        textposition="top center",
        line={"color": trace_color, "width": 1.8},
        marker={"color": trace_color, "size": 6},
    )
    apply_original_chart_style(fig, display_title)
    add_era_backgrounds(fig, country_goals)
    add_expansion_markers(fig, expansion, show_labels=not compact)
    add_country_best_finish_annotations(fig, country_goals, y_column)
    set_edition_ticks(fig, country_goals, tickangle=45 if compact else 30)
    return fig


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def build_correlation_table(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str = "finish_score",
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        if feature not in df.columns or target not in df.columns:
            continue
        analysis_df = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
        rows.append(
            {
                "feature": feature,
                "feature_label": feature_label(feature),
                "pearson_corr": analysis_df[feature].corr(analysis_df[target], method="pearson")
                if len(analysis_df) > 2
                else None,
                "spearman_corr": analysis_df[feature].corr(analysis_df[target], method="spearman")
                if len(analysis_df) > 2
                else None,
                "rows": len(analysis_df),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "feature_label", "pearson_corr", "spearman_corr", "rows"])
    corr_df = pd.DataFrame(rows)
    corr_df["abs_spearman_corr"] = corr_df["spearman_corr"].abs()
    return corr_df.sort_values("abs_spearman_corr", ascending=False).drop(columns="abs_spearman_corr")


def render_correlation_bar(corr_df: pd.DataFrame, title: str, key: str | None = None) -> None:
    if corr_df.empty:
        st.info("No correlation data is available for this chart.")
        return
    display_title = world_cup_chart_title(title)
    fig = px.bar(
        corr_df.sort_values("spearman_corr"),
        x="spearman_corr",
        y="feature_label",
        orientation="h",
        color="spearman_corr",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        text=corr_df.sort_values("spearman_corr")["spearman_corr"].map(lambda value: f"{value:.2f}"),
        hover_data={"pearson_corr": ":.3f", "rows": True, "feature": False, "feature_label": False},
        labels={"spearman_corr": "Spearman correlation", "feature_label": "Feature"},
        title=display_title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, display_title, height=max(500, 42 * len(corr_df) + 180))
    fig.update_layout(coloraxis_showscale=False)
    render_plotly_chart(fig, key=key)


def render_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    target: str = "finish_score",
    key: str | None = None,
) -> None:
    available = [column for column in [target, *columns] if column in df.columns]
    if len(available) < 2:
        st.info("No heatmap data is available for this chart.")
        return
    display_title = world_cup_chart_title(title)
    heatmap_corr = df[available].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    labels = [feature_label(column) for column in heatmap_corr.columns]
    heatmap_corr.index = labels
    heatmap_corr.columns = labels
    fig = px.imshow(
        heatmap_corr,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdYlGn",
        text_auto=".2f",
        aspect="auto",
        title=display_title,
    )
    apply_original_chart_style(fig, display_title, height=max(620, 48 * len(labels) + 220))
    fig.update_xaxes(tickangle=35)
    render_plotly_chart(fig, key=key)


def render_feature_scatter(
    df: pd.DataFrame,
    feature: str,
    target: str = "finish_score",
    key: str | None = None,
) -> None:
    if feature not in df.columns or target not in df.columns:
        return
    scatter_columns = [column for column in ["edition", "country", "placement", target, feature] if column in df.columns]
    scatter_df = df[scatter_columns].copy()
    scatter_df[feature] = pd.to_numeric(scatter_df[feature], errors="coerce")
    scatter_df[target] = pd.to_numeric(scatter_df[target], errors="coerce")
    scatter_df = scatter_df.dropna(subset=[feature, target])
    if scatter_df.empty:
        return
    title = world_cup_chart_title(f"{feature_label(feature)} vs {feature_label(target)}")
    fig = px.scatter(
        scatter_df,
        x=feature,
        y=target,
        color="placement" if "placement" in scatter_df.columns else None,
        hover_name="country",
        hover_data={column: True for column in ["edition"] if column in scatter_df.columns}
        | {feature: ":.3f", target: ":.3f"},
        labels={feature: feature_label(feature), target: feature_label(target)},
        title=title,
    )
    apply_original_chart_style(fig, title, height=550)
    render_plotly_chart(fig, key=key)


def render_participation_tab(outputs: dict[str, pd.DataFrame], placement: pd.DataFrame) -> None:
    participating = outputs["participating_teams"]
    confed = outputs["confederation_by_edition"]
    debutants = outputs["debutants_by_edition"]
    latest = outputs.get("latest_team_distribution")
    placement_history = prepare_country_placement_metrics(placement)

    min_year = int(participating["edition"].min())
    max_year = int(participating["edition"].max())
    year_range = st.slider(
        "Edition range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=4,
        key="historical_eda_participation_year_range",
    )
    confederation_options = [
        confederation
        for confederation in CONFEDERATION_ORDER
        if confederation in set(confed["confederation"].dropna())
    ]
    selected_confederations = st.multiselect(
        "Confederation filter",
        options=confederation_options,
        default=confederation_options,
        key="historical_eda_confederation_filter",
    )
    filtered_participating = participating[
        participating["edition"].between(year_range[0], year_range[1])
    ].copy()
    filtered_confed = confed[confed["edition"].between(year_range[0], year_range[1])].copy()
    if selected_confederations:
        filtered_confed = filtered_confed.loc[
            filtered_confed["confederation"].isin(selected_confederations)
        ].copy()
        if latest is not None and not latest.empty:
            latest = latest.loc[latest["confederation"].isin(selected_confederations)].copy()
    filtered_participating["team_count_label"] = filtered_participating["team_counts"].astype(int).astype(str)
    filtered_confed["participant_count_label"] = filtered_confed["participant_count"].astype(int).astype(str)
    debutants = debutants.copy()
    debutants["debutant_count_label"] = debutants["debutant_count"].astype(int).astype(str)
    expansion = expansion_editions(participating)
    filtered_expansion = expansion.loc[expansion["edition"].between(year_range[0], year_range[1])]

    render_metric_row(
        {
            "Editions": filtered_participating["edition"].nunique(),
            "Largest Tournament Size": int(filtered_participating["team_counts"].max()),
            "2026 Tournament Size": int(participating.loc[participating["edition"].eq(2026), "team_counts"].max()),
        }
    )

    fig = px.bar(
        filtered_participating,
        x="edition",
        y="team_counts",
        color="tournament_count",
        color_discrete_map=FIELD_SIZE_COLORS,
        text="team_count_label",
        labels={"edition": "Edition", "team_counts": "Teams", "tournament_count": "Tournament size"},
        title="FIFA Men's World Cup Tournament Size by Edition",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, "FIFA Men's World Cup Tournament Size by Edition")
    add_era_backgrounds(fig, filtered_participating)
    add_expansion_markers(fig, filtered_expansion)
    set_edition_ticks(fig, filtered_participating)
    render_plotly_chart(fig)

    fig = px.line(
        filtered_confed,
        x="edition",
        y="participant_count",
        color="confederation",
        text="participant_count_label",
        markers=True,
        category_orders={"confederation": CONFEDERATION_ORDER},
        color_discrete_map=CONFEDERATION_COLORS,
        labels={"edition": "Edition", "participant_count": "Teams"},
        title="FIFA Men's World Cup Participation by Confederation",
    )
    fig.update_traces(textposition="top center")
    apply_original_chart_style(fig, "FIFA Men's World Cup Participation by Confederation")
    add_era_backgrounds(fig, filtered_confed)
    add_expansion_markers(fig, filtered_expansion)
    set_edition_ticks(fig, filtered_confed)
    render_plotly_chart(fig)

    placement_countries = sorted(placement_history["country"].dropna().unique())
    selected_placement_country = st.selectbox(
        "Placement country",
        placement_countries,
        index=placement_countries.index("Brazil") if "Brazil" in placement_countries else 0,
        key="historical_eda_placement_country",
    )
    country_placement = placement_history.loc[
        placement_history["country"].eq(selected_placement_country)
        & placement_history["edition"].between(year_range[0], year_range[1])
    ].copy()
    country_placement["placement_label"] = country_placement.apply(placement_axis_label, axis=1)
    country_placement["placement_short_label"] = country_placement["placement"].map(
        PLACEMENT_SHORT_LABELS
    ).fillna(country_placement["placement_label"])
    placement_fig = px.line(
        country_placement,
        x="edition",
        y="position",
        markers=True,
        text="placement_short_label",
        hover_data={"era": True, "placement": True, "position": True},
        labels={"edition": "Edition", "position": "Placement"},
        title=country_world_cup_chart_title(selected_placement_country, "Placement by Edition"),
    )
    placement_fig.update_traces(
        textposition="top center",
        line={"color": CHART_TEXT_COLOR, "width": 1.5},
        marker={"color": CHART_TEXT_COLOR, "size": 5},
    )
    apply_original_chart_style(
        placement_fig,
        country_world_cup_chart_title(selected_placement_country, "Placement by Edition"),
    )
    add_era_backgrounds(placement_fig, country_placement)
    add_expansion_markers(placement_fig, filtered_expansion)
    placement_ticks = (
        country_placement.dropna(subset=["position"])
        .sort_values("position")
        .drop_duplicates("position")
    )
    placement_fig.update_yaxes(
        autorange="reversed",
        title="Placement",
        tickmode="array",
        tickvals=placement_ticks["position"].tolist(),
        ticktext=placement_ticks["placement_label"].tolist(),
    )
    set_edition_ticks(placement_fig, country_placement)
    render_plotly_chart(placement_fig)

    debutant_fig = px.bar(
        debutants,
        x="edition",
        y="debutant_count",
        color="era",
        text="debutant_count_label",
        category_orders={"era": ERA_LABELS},
        color_discrete_map=ERA_COLORS,
        labels={"edition": "Edition", "debutant_count": "Debutants"},
        title="FIFA Men's World Cup Debutants by Edition",
    )
    debutant_fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(debutant_fig, "FIFA Men's World Cup Debutants by Edition", height=500)
    add_expansion_markers(debutant_fig, filtered_expansion)
    set_edition_ticks(debutant_fig, debutants)

    if latest is not None and not latest.empty:
        latest_edition = int(latest["edition"].max())
        distribution_fig = px.treemap(
            latest,
            path=[px.Constant(f"{latest_edition} FIFA Men's World Cup"), "confederation", "country_label"],
            values="team_value",
            color="confederation",
            color_discrete_map=CONFEDERATION_COLORS,
            custom_data=["country", "participation_count", "prior_participations", "is_first_timer"],
            title=f"FIFA Men's World Cup {latest_edition} Team Distribution",
        )
        first_timers = latest.loc[latest["is_first_timer"], "country"].sort_values().tolist()
        first_timer_summary = ", ".join(first_timers) if first_timers else "None"
        distribution_fig.update_traces(
            textinfo="label+value+percent parent",
            textposition="top left",
            textfont={"color": CHART_TEXT_COLOR},
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Participation count: %{customdata[1]}<br>"
                "Previous participations: %{customdata[2]}<br>"
                "First timer: %{customdata[3]}"
                "<extra></extra>"
            ),
            marker={"cornerradius": 5},
        )
        apply_original_chart_style(distribution_fig, f"FIFA Men's World Cup {latest_edition} Team Distribution", height=640)
        distribution_fig.add_annotation(
            text=f"First timers: {first_timer_summary}. Marked with '★'",
            x=0,
            y=1.05,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font={"size": 11, "color": CHART_AXIS_COLOR},
        )
    else:
        latest_distribution = outputs["latest_distribution"]
        if selected_confederations:
            latest_distribution = latest_distribution.loc[
                latest_distribution["confederation"].isin(selected_confederations)
            ].copy()
        latest_edition = int(latest_distribution["edition"].max())
        distribution_fig = px.treemap(
            latest_distribution,
            path=[px.Constant(f"{latest_edition} FIFA Men's World Cup"), "confederation"],
            values="team_count",
            color="confederation",
            color_discrete_map=CONFEDERATION_COLORS,
            title=world_cup_chart_title(f"{latest_edition} Team Distribution"),
        )
        distribution_fig.update_traces(
            textinfo="label+value+percent parent",
            textposition="top left",
            textfont={"color": CHART_TEXT_COLOR},
            marker={"cornerradius": 5},
        )
        apply_original_chart_style(distribution_fig, f"FIFA Men's World Cup {latest_edition} Team Distribution", height=640)

    left, right = st.columns(2)
    render_column_plotly_chart(left, debutant_fig)
    render_column_plotly_chart(right, distribution_fig)


def render_goals_tab(outputs: dict[str, pd.DataFrame], participation_outputs: dict[str, pd.DataFrame]) -> None:
    tournament_goals = outputs["tournament_goals"]
    team_goals = prepare_country_goal_metrics(outputs["team_goals"])
    match_scorelines = outputs.get("match_scorelines", pd.DataFrame()).copy()
    expansion = expansion_editions(participation_outputs["participating_teams"])

    min_year = int(tournament_goals["edition"].min())
    max_year = int(tournament_goals["edition"].max())
    goal_metric_mode = st.radio(
        "Goal metric",
        ["Per game", "Totals"],
        horizontal=True,
        key="historical_eda_goal_metric_mode",
    )
    year_range = st.slider(
        "Edition range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=4,
        key="historical_eda_goals_year_range",
    )
    tournament_goals = tournament_goals.loc[tournament_goals["edition"].between(year_range[0], year_range[1])].copy()
    team_goals = team_goals.loc[team_goals["edition"].between(year_range[0], year_range[1])].copy()
    if not match_scorelines.empty:
        match_scorelines = match_scorelines.loc[
            match_scorelines["edition"].between(year_range[0], year_range[1])
        ].copy()
    expansion = expansion.loc[expansion["edition"].between(year_range[0], year_range[1])].copy()

    placement_summary = (
        team_goals.groupby("placement", as_index=False)
        .agg(
            avg_goals_for=("gf", "mean"),
            avg_goals_against=("ga", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
            rows=("country", "count"),
        )
        .round(2)
        .sort_values("avg_goal_difference", ascending=False)
        .reset_index(drop=True)
    )

    countries = sorted(team_goals["country"].dropna().unique())
    selected_country = st.selectbox(
        "Country",
        countries,
        index=countries.index("Brazil") if "Brazil" in countries else 0,
        key="historical_eda_goal_country",
    )
    if goal_metric_mode == "Totals":
        tournament_y = "total_goals"
        tournament_text = tournament_goals[tournament_y].map(lambda value: f"{value:.0f}")
        tournament_label = "Total goals"
        tournament_title = "Tournament Total Goals"
        country_y = "gf"
        country_conceded_y = "ga"
        country_label = "Goals for"
        country_conceded_label = "Goals conceded"
        country_title = country_world_cup_chart_title(selected_country, "Goals Scored")
        country_conceded_title = country_world_cup_chart_title(selected_country, "Goals Conceded")
        country_value_format = ".0f"
    else:
        tournament_y = "goals_per_match"
        tournament_text = tournament_goals[tournament_y].map(lambda value: f"{value:.2f}")
        tournament_label = "Goals per match"
        tournament_title = "Tournament Goals per Match"
        country_y = "goals_per_game"
        country_conceded_y = "goals_against_per_game"
        country_label = "Goals Scored per game"
        country_conceded_label = "Goals conceded per game"
        country_title = country_world_cup_chart_title(selected_country, "Goals Scored per Game")
        country_conceded_title = country_world_cup_chart_title(selected_country, "Goals Conceded per Game")
        country_value_format = ".2f"

    fig = px.line(
        tournament_goals,
        x="edition",
        y=tournament_y,
        markers=True,
        text=tournament_text,
        hover_data={"era": True, "total_matches": True, tournament_y: ":.3f"},
        labels={"edition": "Edition", tournament_y: tournament_label},
        title=world_cup_chart_title(tournament_title),
    )
    fig.update_traces(
        textposition="top center",
        line={"color": CHART_POSITIVE_COLOR, "width": 1.8},
        marker={"color": CHART_POSITIVE_COLOR, "size": 6},
    )
    apply_original_chart_style(fig, tournament_title)
    add_era_backgrounds(fig, tournament_goals)
    add_expansion_markers(fig, expansion)
    set_edition_ticks(fig, tournament_goals)
    render_plotly_chart(fig)

    if not match_scorelines.empty:
        scoreline_scope = st.selectbox(
            "Scoreline box plot scope",
            ["All stages", "Knockouts only"],
            key="historical_eda_scoreline_scope",
        )
        scoreline_plot = match_scorelines.copy()
        if scoreline_scope == "Knockouts only":
            scoreline_plot = scoreline_plot.loc[scoreline_plot["stage"].isin(GOALS_KNOCKOUT_STAGES)].copy()
        scoreline_plot = scoreline_plot.loc[scoreline_plot["stage"].isin(GOALS_STAGE_ORDER)].copy()

        if scoreline_plot.empty:
            st.info("No match scoreline data is available for this scope.")
        else:
            scoreline_summary = (
                scoreline_plot.groupby("stage", observed=True)
                .agg(avg_goals_per_match=("total_goals", "mean"), matches=("match_id", "count"))
                .reset_index()
            )
            mode_scorelines = (
                scoreline_plot.groupby(["stage", "scoreline"], observed=True)
                .size()
                .reset_index(name="scoreline_count")
                .sort_values(["stage", "scoreline_count", "scoreline"], ascending=[True, False, True], kind="stable")
                .drop_duplicates("stage")
                .rename(columns={"scoreline": "mode_scoreline"})
            )
            scoreline_summary = scoreline_summary.merge(mode_scorelines, on="stage", how="left")
            scoreline_fig = px.box(
                scoreline_plot,
                x="stage",
                y="total_goals",
                points="all",
                category_orders={"stage": GOALS_STAGE_ORDER},
                hover_data={
                    "edition": True,
                    "country": True,
                    "opponent": True,
                    "score": True,
                    "scoreline": True,
                    "match_id": True,
                    "scoreline_rank": False,
                },
                labels={
                    "stage": "Round",
                    "total_goals": "Total goals",
                    "country": "Team",
                    "score": "Original score",
                    "scoreline": "Canonical scoreline",
                    "match_id": "Match ID",
                },
                title=world_cup_chart_title("Match Scoreline Distribution by Round"),
            )
            scoreline_fig.update_traces(
                marker={"color": CHART_POSITIVE_COLOR, "size": 4, "opacity": 0.6},
                line={"color": CHART_AXIS_COLOR},
                jitter=0.35,
            )
            apply_original_chart_style(scoreline_fig, "Match Scoreline Distribution by Round", height=620)
            min_total_goals = int(scoreline_plot["total_goals"].min())
            max_total_goals = int(scoreline_plot["total_goals"].max())
            scoreline_fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(min_total_goals, max_total_goals + 1)),
                ticktext=[str(value) for value in range(min_total_goals, max_total_goals + 1)],
            )
            annotation_y = float(max_total_goals) + 0.9
            scoreline_fig.update_yaxes(range=[max(-0.5, min_total_goals - 0.5), annotation_y + 0.6])
            for row in scoreline_summary.itertuples(index=False):
                scoreline_fig.add_annotation(
                    x=row.stage,
                    y=annotation_y,
                    text=f"Mode {row.mode_scoreline}<br>Mean {row.avg_goals_per_match:.2f} goals",
                    showarrow=False,
                    align="center",
                    font={"size": 10, "color": CHART_AXIS_COLOR},
                )
            render_plotly_chart(scoreline_fig)

    country_goals = team_goals.loc[team_goals["country"].eq(selected_country)].sort_values("edition")
    scored_fig = render_country_goal_line(
        country_goals=country_goals,
        y_column=country_y,
        y_label=country_label,
        title=country_title,
        value_format=country_value_format,
        expansion=expansion,
        trace_color=CHART_POSITIVE_COLOR,
        compact=True,
    )
    conceded_fig = render_country_goal_line(
        country_goals=country_goals,
        y_column=country_conceded_y,
        y_label=country_conceded_label,
        title=country_conceded_title,
        value_format=country_value_format,
        expansion=expansion,
        trace_color=CHART_NEGATIVE_COLOR,
        compact=True,
    )
    scored_column, conceded_column = st.columns(2)
    render_column_plotly_chart(scored_column, scored_fig)
    render_column_plotly_chart(conceded_column, conceded_fig)

    st.dataframe(
        placement_summary,
        hide_index=True,
        width="stretch",
    )


def render_host_tab(outputs: dict[str, pd.DataFrame]) -> None:
    hosts = outputs["hosts"].copy()
    hosts["host_goal_size"] = pd.to_numeric(hosts["gf"], errors="coerce").fillna(1).clip(lower=1)
    summary = outputs["host_summary"].iloc[0]
    render_metric_row(
        {
            "Host Teams": int(summary["host_teams"]),
            "Host Titles": int(summary["titles"]),
            "Top-Four Host Finishes": int(summary["top_four_finishes"]),
            "Avg Host Finish": f'{summary["avg_position"]:.2f}',
        }
    )

    fig = px.scatter(
        hosts,
        x="edition",
        y="position",
        color="confederation",
        size="host_goal_size",
        text="country",
        hover_name="country",
        category_orders={"confederation": CONFEDERATION_ORDER},
        color_discrete_map=CONFEDERATION_COLORS,
        labels={"edition": "Edition", "position": "Final position", "gf": "Goals for"},
        title="FIFA Men's World Cup Host Nation Finishes",
    )
    fig.update_traces(textposition="top center")
    apply_original_chart_style(fig, "FIFA Men's World Cup Host Nation Finishes")
    fig.update_yaxes(autorange="reversed")
    set_edition_ticks(fig, hosts)
    render_plotly_chart(fig)

    st.dataframe(
        hosts[
            [
                "edition",
                "country",
                "confederation",
                "placement",
                "position",
                "gf",
                "ga",
                "next_edition",
                "next_placement",
            ]
        ],
        hide_index=True,
        width="stretch",
    )


def render_winner_followup_tab(winners: pd.DataFrame) -> None:
    winners = winners.copy()
    winners["next_placement_short"] = winners["next_placement"].map(PLACEMENT_SHORT_LABELS).fillna(
        winners["next_placement"].astype(str)
    )
    render_metric_row(
        {
            "Champions Tracked": len(winners),
            "Repeated as Champion": int(winners["next_placement"].eq("Winner").sum()),
            "Did Not Qualify Next": int(winners["next_placement"].eq("DNQ").sum()),
        }
    )
    fig = px.bar(
        winners,
        x="edition",
        y="next_position",
        color="next_placement",
        hover_name="country",
        text="next_placement_short",
        labels={"edition": "Title edition", "next_position": "Next edition position"},
        title=world_cup_chart_title("Champion Follow-up Performance"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, f"Winners Placement the following Edition since {int(winners['edition'].min())}")
    fig.update_yaxes(autorange="reversed", title="Final Placement")
    fig.update_xaxes(title="Tournament Edition", tickangle=45)
    set_edition_ticks(fig, winners, tickangle=45)
    render_plotly_chart(fig)
    st.dataframe(winners, hide_index=True, width="stretch")


def render_correlations_tab(outputs: dict[str, pd.DataFrame]) -> None:
    outcome = outputs["outcome_frame"].copy()
    last_k_features = outputs["last_k_features"].copy()
    last_k_summary = outputs["last_k_summary"].iloc[0]
    pre_features = [feature for feature in PRE_TOURNAMENT_FEATURES if feature in outcome.columns]
    tournament_features = [feature for feature in IN_TOURNAMENT_FEATURES if feature in outcome.columns]
    last_k_columns = [feature for feature in LAST_K_FEATURES if feature in last_k_features.columns]
    pre_corr = build_correlation_table(outcome, pre_features)
    tournament_corr = build_correlation_table(outcome, tournament_features)
    last_k_analysis = last_k_features.rename(columns={"current_finish_score": "finish_score"})
    last_k_corr = build_correlation_table(last_k_analysis, last_k_columns)
    combined_last_k_features = [*pre_features, *last_k_columns]
    combined_last_k_analysis = outcome.merge(
        last_k_analysis.drop(columns=["lookback"], errors="ignore"),
        on=["edition", "country"],
        how="left",
        suffixes=("", "_last_k"),
    )
    combined_corr = build_correlation_table(combined_last_k_analysis, combined_last_k_features)

    render_metric_row(
        {
            "Last-k Lookback": int(last_k_summary["lookback"]),
            "Last-k Rows": int(last_k_summary["rows"]),
            "Last-k Correlation": f'{last_k_summary["correlation_with_finish_score"]:.3f}',
        }
    )

    chart_tabs = st.tabs(
        [
            "Pre-Tournament",
            "Tournament Stats",
            "Last-k History",
            "Baseline + Last-k",
            "Strongest Scatters",
            "Tables",
        ]
    )

    with chart_tabs[0]:
        render_correlation_bar(
            pre_corr,
            "Pre-Tournament Feature Correlation with World Cup Finish Score",
            key="historical_corr_pre_bar",
        )
        render_correlation_heatmap(
            outcome,
            pre_features,
            "Spearman Correlation Heatmap: Outcome and Predictors",
            key="historical_corr_pre_heatmap",
        )

    with chart_tabs[1]:
        render_correlation_bar(
            tournament_corr,
            "In-Tournament Stat Correlation with World Cup Finish Score",
            key="historical_corr_tournament_bar",
        )
        render_correlation_heatmap(
            outcome,
            tournament_features,
            "Spearman Correlation Heatmap: Outcome and Tournament Stats",
            key="historical_corr_tournament_heatmap",
        )

    with chart_tabs[2]:
        render_correlation_bar(
            last_k_corr,
            f"Last-{int(last_k_summary['lookback'])} World Cup History Correlation with Finish Score",
            key="historical_corr_last_k_bar",
        )
        render_correlation_heatmap(
            last_k_analysis,
            last_k_columns,
            f"Spearman Correlation Heatmap: Outcome and Last-{int(last_k_summary['lookback'])} History",
            key="historical_corr_last_k_heatmap",
        )

    with chart_tabs[3]:
        render_correlation_bar(
            combined_corr,
            f"Pre-Tournament Predictors + Last-{int(last_k_summary['lookback'])} World Cup History Correlation with Finish Score",
            key="historical_corr_combined_bar",
        )
        render_correlation_heatmap(
            combined_last_k_analysis,
            combined_last_k_features,
            f"Spearman Correlation Heatmap: Outcome, Baseline Predictors, and Last-{int(last_k_summary['lookback'])} World Cup History",
            key="historical_corr_combined_heatmap",
        )

    with chart_tabs[4]:
        strongest_pre_features = pre_corr.head(3)["feature"].tolist()
        strongest_last_k_features = combined_corr.head(3)["feature"].tolist()
        if not strongest_pre_features and not strongest_last_k_features:
            st.info("No scatter data is available for the strongest predictors.")
        if strongest_pre_features:
            st.markdown("**Strongest Pre-Tournament Predictors**")
        for strongest_feature in strongest_pre_features:
            render_feature_scatter(
                outcome,
                strongest_feature,
                key=f"historical_corr_pre_scatter_{strongest_feature}",
            )
        if strongest_last_k_features:
            st.markdown(f"**Strongest Baseline + Last-{int(last_k_summary['lookback'])} Predictors**")
        for strongest_feature in strongest_last_k_features:
            render_feature_scatter(
                combined_last_k_analysis,
                strongest_feature,
                key=f"historical_corr_combined_scatter_{strongest_feature}",
            )

    with chart_tabs[5]:
        st.markdown("**Pre-Tournament Correlations**")
        st.dataframe(pre_corr, hide_index=True, width="stretch")
        st.markdown("**In-Tournament Correlations**")
        st.dataframe(tournament_corr, hide_index=True, width="stretch")
        st.markdown("**Last-k History Correlations**")
        st.dataframe(last_k_corr, hide_index=True, width="stretch")
        st.markdown("**Baseline + Last-k Correlations**")
        st.dataframe(combined_corr, hide_index=True, width="stretch")


def render_qualifiers_tab(outputs: dict[str, pd.DataFrame]) -> None:
    summary = outputs["summary"].copy()
    matches = outputs["matches"].copy()
    if summary.empty:
        st.info("No 2026 qualifier performance data is available for non-host qualified teams.")
        return

    confederations = [
        confederation
        for confederation in CONFEDERATION_ORDER
        if confederation in set(summary["confederation"].dropna().astype(str))
    ]
    extra_confederations = sorted(set(summary["confederation"].dropna().astype(str)).difference(confederations))
    selected_confederation = st.selectbox(
        "Confederation",
        [ALL_CONFEDERATION_FILTER, *confederations, *extra_confederations],
        key="historical_eda_qualifier_confederation",
    )
    if selected_confederation != ALL_CONFEDERATION_FILTER:
        summary = summary.loc[summary["confederation"].eq(selected_confederation)].copy()
        matches = matches.loc[matches["Confederation"].eq(selected_confederation)].copy()

    if summary.empty:
        st.info("No qualifier data is available for this confederation.")
        return

    top_team = summary.sort_values("performance_score", ascending=False, kind="stable").iloc[0]
    total_matches = int(summary["matches"].sum())
    avg_points = float(summary["points"].sum() / total_matches) if total_matches else 0.0
    avg_goals = float(summary["goals_for"].sum() / total_matches) if total_matches else 0.0
    render_metric_row(
        {
            "Teams": len(summary),
            "Matches": total_matches,
            "Avg Points/Match": f"{avg_points:.2f}",
            "Avg Goals/Match": f"{avg_goals:.2f}",
            "Top Performance Team": str(top_team["team"]),
        }
    )

    chart_limit = min(20, len(summary))
    performance_rank = summary.nlargest(chart_limit, "performance_score").sort_values("performance_score")
    performance_fig = px.bar(
        performance_rank,
        x="performance_score",
        y="team",
        orientation="h",
        color="confederation",
        color_discrete_map=CONFEDERATION_COLORS,
        text=performance_rank["performance_score"].map(lambda value: f"{value:.1f}"),
        hover_data={
            "matches": True,
            "points_per_match": ":.2f",
            "goal_difference_per_match": ":.2f",
            "elo_change_per_match": ":.2f",
            "confederation": False,
            "team": False,
        },
        labels={"performance_score": "Performance Score", "team": "Team"},
        title=world_cup_chart_title("2026 Qualifier Performance Score"),
    )
    performance_fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(performance_fig, "2026 Qualifier Performance Score", height=max(520, chart_limit * 28 + 180))
    performance_fig.update_yaxes(categoryorder="array", categoryarray=performance_rank["team"].tolist())
    render_plotly_chart(performance_fig)

    goals_mode = st.radio(
        "Goals chart values",
        ["Totals", "Per game"],
        horizontal=True,
        key="historical_eda_qualifier_goals_mode",
    )
    goals_for_metric = "goals_for" if goals_mode == "Totals" else "goals_for_per_match"
    goals_against_metric = "goals_against" if goals_mode == "Totals" else "goals_against_per_match"
    goals_for_label = "Goals For" if goals_mode == "Totals" else "Goals For per Game"
    goals_against_label = "Goals Against" if goals_mode == "Totals" else "Goals Against per Game"
    goals_text_format = ".0f" if goals_mode == "Totals" else ".2f"

    goals_for_rank = summary.nlargest(chart_limit, goals_for_metric).sort_values(goals_for_metric)
    goals_against_rank = summary.nsmallest(chart_limit, goals_against_metric).sort_values(
        goals_against_metric, ascending=False
    )
    left, right = st.columns(2)
    goals_for_fig = px.bar(
        goals_for_rank,
        x=goals_for_metric,
        y="team",
        orientation="h",
        color="confederation",
        color_discrete_map=CONFEDERATION_COLORS,
        text=goals_for_rank[goals_for_metric].map(lambda value: f"{value:{goals_text_format}}"),
        hover_data={"matches": True, "goals_for_per_match": ":.2f", "confederation": False, "team": False},
        labels={goals_for_metric: goals_for_label, "team": "Team"},
        title=world_cup_chart_title(f"2026 Qualifier {goals_for_label}"),
    )
    goals_for_fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(goals_for_fig, f"2026 Qualifier {goals_for_label}", height=560)
    goals_for_fig.update_yaxes(categoryorder="array", categoryarray=goals_for_rank["team"].tolist())
    render_column_plotly_chart(left, goals_for_fig)

    goals_against_fig = px.bar(
        goals_against_rank,
        x=goals_against_metric,
        y="team",
        orientation="h",
        color="confederation",
        color_discrete_map=CONFEDERATION_COLORS,
        text=goals_against_rank[goals_against_metric].map(lambda value: f"{value:{goals_text_format}}"),
        hover_data={"matches": True, "goals_against_per_match": ":.2f", "confederation": False, "team": False},
        labels={goals_against_metric: goals_against_label, "team": "Team"},
        title=world_cup_chart_title(f"2026 Qualifier {goals_against_label}"),
    )
    goals_against_fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(goals_against_fig, f"2026 Qualifier {goals_against_label}", height=560)
    goals_against_fig.update_yaxes(categoryorder="array", categoryarray=goals_against_rank["team"].tolist())
    render_column_plotly_chart(right, goals_against_fig)

    scatter_fig = px.scatter(
        summary,
        x="goals_for_per_match",
        y="goals_against_per_match",
        color="confederation",
        size="points_per_match",
        hover_name="team",
        color_discrete_map=CONFEDERATION_COLORS,
        hover_data={
            "matches": True,
            "performance_score": ":.1f",
            "goal_difference_per_match": ":.2f",
            "elo_change_per_match": ":.2f",
            "confederation": False,
        },
        labels={
            "goals_for_per_match": "Goals For per Match",
            "goals_against_per_match": "Goals Against per Match",
            "points_per_match": "Points per Match",
        },
        title=world_cup_chart_title("2026 Qualifier Attack vs Defense"),
    )
    apply_original_chart_style(scatter_fig, "2026 Qualifier Attack vs Defense", height=620)
    render_plotly_chart(scatter_fig)

    display_summary = summary.rename(
        columns={
            "team": "Team",
            "confederation": "Confederation",
            "qualification_path": "Qualification Path",
            "matches": "Matches",
            "wins": "Wins",
            "draws": "Draws",
            "losses": "Losses",
            "points": "Points",
            "points_per_match": "Points/Match",
            "goals_for": "Goals For",
            "goals_against": "Goals Against",
            "goal_difference": "Goal Difference",
            "goals_for_per_match": "Goals For/Match",
            "goals_against_per_match": "Goals Against/Match",
            "goal_difference_per_match": "Goal Difference/Match",
            "elo_change": "Elo Change",
            "elo_change_per_match": "Elo Change/Match",
            "performance_score": "Performance Score",
        }
    )
    st.dataframe(
        display_summary[
            [
                "Team",
                "Confederation",
                "Qualification Path",
                "Matches",
                "Wins",
                "Draws",
                "Losses",
                "Points",
                "Points/Match",
                "Goals For",
                "Goals Against",
                "Goal Difference",
                "Goals For/Match",
                "Goals Against/Match",
                "Goal Difference/Match",
                "Elo Change",
                "Elo Change/Match",
                "Performance Score",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

    with st.expander("Qualifier match details"):
        st.dataframe(
            matches.loc[
                :,
                [
                    "Date",
                    "Team",
                    "Confederation",
                    "Opponent",
                    "Result",
                    "Score",
                    "Tournament",
                    "Stage",
                    "Venue",
                    "Elo Change",
                ],
            ],
            hide_index=True,
            width="stretch",
        )


def render_2026_implications_tab(outputs: dict[str, pd.DataFrame]) -> None:
    qualified = outputs["qualified_teams"]
    distribution = outputs["confederation_distribution"]
    context = outputs["edition_context"]
    latest_context = context.iloc[0] if not context.empty else {}

    render_metric_row(
        {
            "Edition": int(latest_context.get("edition", 2026)),
            "Teams": int(distribution["team_count"].sum()),
            "Confederations": distribution["confederation"].nunique(),
        }
    )

    left, right = st.columns([1, 2])
    distribution_fig = px.pie(
        distribution,
        names="confederation",
        values="team_count",
        color="confederation",
        color_discrete_map=CONFEDERATION_COLORS,
        title=world_cup_chart_title("2026 Confederation Share"),
    )
    distribution_fig.update_traces(textinfo="label+value+percent", textposition="inside")
    apply_original_chart_style(distribution_fig, "2026 Confederation Share", height=500)
    render_column_plotly_chart(left, distribution_fig)
    right.dataframe(qualified, hide_index=True, width="stretch")


def render_historical_eda_page() -> None:
    st.title("FIFA Men's World Cup Analysis")
    st.caption(
        "An iteractive analysis companion which covers the history of the FIFA Men's World Cup"
    )

    lookback = st.sidebar.slider(
        "Correlation lookback",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        key="historical_eda_lookback",
    )
    (
        quality,
        placement,
        participation,
        goals,
        hosts,
        winners,
        correlations,
        implications_2026,
    ) = compute_historical_eda_outputs(lookback=lookback)

    with st.expander("Data quality snapshot"):
        st.dataframe(quality, hide_index=True, width="stretch")

    qualifier_outputs = load_qualifier_performance_tables()

    tabs = st.tabs(
        [
            "Participation",
            "Goals",
            "Host Effect",
            "Winner Follow-up",
            "Correlations",
            "Qualifiers",
            "2026 Implications",
        ]
    )
    with tabs[0]:
        render_participation_tab(participation, placement)
    with tabs[1]:
        render_goals_tab(goals, participation)
    with tabs[2]:
        render_host_tab(hosts)
    with tabs[3]:
        render_winner_followup_tab(winners)
    with tabs[4]:
        render_correlations_tab(correlations)
    with tabs[5]:
        render_qualifiers_tab(qualifier_outputs)
    with tabs[6]:
        render_2026_implications_tab(implications_2026)
