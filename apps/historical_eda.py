from __future__ import annotations

from pathlib import Path
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

CHART_BACKGROUND = "#EFE3CF"
CHART_TEXT_COLOR = "#3A2A1A"
CHART_AXIS_COLOR = "#5A4632"
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
    "prior_avg_finish_score",
    "prior_best_finish_score",
    "prior_avg_goals_per_match",
    "prior_avg_goal_diff_per_match",
]

IN_TOURNAMENT_FEATURES = [
    "matches_played",
    "gs",
    "ga",
    "goal_difference",
    "goals_per_match",
    "goals_against_per_match",
    "goal_difference_per_match",
    "elo_change",
]

LAST_K_FEATURES = [
    "prior_appearance_rate",
    "last_k_avg_finish_score",
]

FEATURE_LABELS = {
    "start_elo": "Starting Elo",
    "elo_rank": "Elo Rank",
    "is_host": "Host",
    "prior_world_cup_participations": "Prior WC Participations",
    "previous_finish_score": "Previous Finish Score",
    "prior_avg_finish_score": "Prior Avg Finish Score",
    "prior_best_finish_score": "Prior Best Finish Score",
    "prior_avg_goals_per_match": "Prior Avg Goals per Match",
    "prior_avg_goal_diff_per_match": "Prior Avg Goal Diff per Match",
    "matches_played": "Matches Played",
    "gs": "Goals For",
    "ga": "Goals Against",
    "goal_difference": "Goal Difference",
    "goals_per_match": "Goals per Match",
    "goals_against_per_match": "Goals Against per Match",
    "goal_difference_per_match": "Goal Difference per Match",
    "elo_change": "Elo Change",
    "prior_appearance_rate": "Last-k Appearance Rate",
    "last_k_avg_finish_score": "Last-k Avg Finish Score",
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


def render_metric_row(values: dict[str, object]) -> None:
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values.items()):
        column.metric(label, value)


def world_cup_chart_title(title: str) -> str:
    return title if TITLE_PREFIX in title else f"{TITLE_PREFIX} {title}"


def country_world_cup_chart_title(country: str, title: str) -> str:
    return f"{country}'s {TITLE_PREFIX} {title}"


def apply_original_chart_style(fig, title: str, height: int = 560):
    """Apply the chart styling from main.ipynb at commit b88051c."""
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


def render_plotly_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG)


def render_column_plotly_chart(column, fig) -> None:
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
        line={"color": CHART_TEXT_COLOR, "width": 1.5},
        marker={"color": CHART_TEXT_COLOR, "size": 4},
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


def render_correlation_bar(corr_df: pd.DataFrame, title: str) -> None:
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
    render_plotly_chart(fig)


def render_correlation_heatmap(df: pd.DataFrame, columns: list[str], title: str, target: str = "finish_score") -> None:
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
    render_plotly_chart(fig)


def render_feature_scatter(df: pd.DataFrame, feature: str, target: str = "finish_score") -> None:
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
    render_plotly_chart(fig)


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
            "Largest Field": int(filtered_participating["team_counts"].max()),
            "2026 Teams": int(participating.loc[participating["edition"].eq(2026), "team_counts"].max()),
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
    placement_summary = outputs["placement_goal_summary"]
    expansion = expansion_editions(participation_outputs["participating_teams"])

    goal_metric_mode = st.radio(
        "Goal metric",
        ["Per game", "Totals"],
        horizontal=True,
        key="historical_eda_goal_metric_mode",
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
        country_title = country_world_cup_chart_title(selected_country, "Goals For")
        country_conceded_title = country_world_cup_chart_title(selected_country, "Goals Conceded")
        country_value_format = ".0f"
    else:
        tournament_y = "goals_per_match"
        tournament_text = tournament_goals[tournament_y].map(lambda value: f"{value:.2f}")
        tournament_label = "Goals per match"
        tournament_title = "Tournament Goals per Match"
        country_y = "goals_per_game"
        country_conceded_y = "goals_against_per_game"
        country_label = "Goals for per game"
        country_conceded_label = "Goals conceded per game"
        country_title = country_world_cup_chart_title(selected_country, "Goals For per Game")
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
        line={"color": CHART_TEXT_COLOR, "width": 1.5},
        marker={"color": CHART_TEXT_COLOR, "size": 4},
    )
    apply_original_chart_style(fig, tournament_title)
    add_era_backgrounds(fig, tournament_goals)
    add_expansion_markers(fig, expansion)
    set_edition_ticks(fig, tournament_goals)
    render_plotly_chart(fig)

    country_goals = team_goals.loc[team_goals["country"].eq(selected_country)].sort_values("edition")
    scored_fig = render_country_goal_line(
        country_goals=country_goals,
        y_column=country_y,
        y_label=country_label,
        title=country_title,
        value_format=country_value_format,
        expansion=expansion,
        compact=True,
    )
    conceded_fig = render_country_goal_line(
        country_goals=country_goals,
        y_column=country_conceded_y,
        y_label=country_conceded_label,
        title=country_conceded_title,
        value_format=country_value_format,
        expansion=expansion,
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
            "Strongest Scatters",
            "Tables",
        ]
    )

    with chart_tabs[0]:
        render_correlation_bar(pre_corr, "Pre-Tournament Feature Correlation with World Cup Finish Score")
        render_correlation_heatmap(
            outcome,
            pre_features,
            "Spearman Correlation Heatmap: Outcome and Predictors",
        )

    with chart_tabs[1]:
        render_correlation_bar(tournament_corr, "In-Tournament Stat Correlation with World Cup Finish Score")
        render_correlation_heatmap(
            outcome,
            tournament_features,
            "Spearman Correlation Heatmap: Outcome and Tournament Stats",
        )

    with chart_tabs[2]:
        render_correlation_bar(
            last_k_corr,
            f"Last-{int(last_k_summary['lookback'])} World Cup History Correlation with Finish Score",
        )
        render_correlation_heatmap(
            last_k_analysis,
            last_k_columns,
            f"Spearman Correlation Heatmap: Outcome and Last-{int(last_k_summary['lookback'])} History",
        )

    with chart_tabs[3]:
        strongest_pre_features = pre_corr.head(3)["feature"].tolist()
        strongest_last_k_features = last_k_corr.head(3)["feature"].tolist()
        if not strongest_pre_features and not strongest_last_k_features:
            st.info("No scatter data is available for the strongest predictors.")
        if strongest_pre_features:
            st.markdown("**Strongest Pre-Tournament Predictors**")
        for strongest_feature in strongest_pre_features:
            render_feature_scatter(outcome, strongest_feature)
        if strongest_last_k_features:
            st.markdown(f"**Strongest Last-{int(last_k_summary['lookback'])} History Predictors**")
        for strongest_feature in strongest_last_k_features:
            render_feature_scatter(last_k_analysis, strongest_feature)

    with chart_tabs[4]:
        st.markdown("**Pre-Tournament Correlations**")
        st.dataframe(pre_corr, hide_index=True, width="stretch")
        st.markdown("**In-Tournament Correlations**")
        st.dataframe(tournament_corr, hide_index=True, width="stretch")
        st.markdown("**Last-k History Correlations**")
        st.dataframe(last_k_corr, hide_index=True, width="stretch")


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
        "Interactive companion to the executive notebook. The page uses processed datasets committed under data/processed/world_cup."
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

    tabs = st.tabs(
        [
            "Participation",
            "Goals",
            "Host Effect",
            "Winner Follow-up",
            "Correlations",
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
        render_2026_implications_tab(implications_2026)
