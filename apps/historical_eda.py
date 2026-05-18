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


@st.cache_data(show_spinner=False)
def load_historical_eda_data() -> dict[str, pd.DataFrame]:
    return load_historical_world_cup_data()


@st.cache_data(show_spinner=False)
def compute_historical_eda_outputs(
    lookback: int,
) -> tuple[
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


def apply_original_chart_style(fig, title: str, height: int = 560):
    """Apply the chart styling from main.ipynb at commit b88051c."""
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
            "showgrid": False,
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
    column.plotly_chart(fig, width="stretch", config=PLOTLY_EXPORT_CONFIG)


def render_participation_tab(outputs: dict[str, pd.DataFrame]) -> None:
    participating = outputs["participating_teams"]
    confed = outputs["confederation_by_edition"]
    debutants = outputs["debutants_by_edition"]
    latest = outputs["latest_distribution"]

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
    filtered_participating = participating[
        participating["edition"].between(year_range[0], year_range[1])
    ].copy()
    filtered_confed = confed[confed["edition"].between(year_range[0], year_range[1])].copy()
    filtered_participating["team_count_label"] = filtered_participating["team_counts"].astype(int).astype(str)
    filtered_confed["participant_count_label"] = filtered_confed["participant_count"].astype(int).astype(str)
    debutants = debutants.copy()
    debutants["debutant_count_label"] = debutants["debutant_count"].astype(int).astype(str)

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
        labels={"edition": "Edition", "team_counts": "Teams", "tournament_count": "Field size"},
        title="World Cup Field Size by Edition",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, "World Cup Field Size by Edition")
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
        title="Participation by Confederation",
    )
    fig.update_traces(textposition="top center")
    apply_original_chart_style(fig, "Participation by Confederation")
    render_plotly_chart(fig)

    debutant_fig = px.bar(
        debutants,
        x="edition",
        y="debutant_count",
        color="era",
        text="debutant_count_label",
        category_orders={"era": ERA_LABELS},
        color_discrete_map=ERA_COLORS,
        labels={"edition": "Edition", "debutant_count": "Debutants"},
        title="Debutants by Edition",
    )
    debutant_fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(debutant_fig, "Debutants by Edition", height=500)

    distribution_fig = px.treemap(
        latest,
        path=["confederation"],
        values="team_count",
        color="confederation",
        color_discrete_map=CONFEDERATION_COLORS,
        title="2026 Team Distribution",
    )
    distribution_fig.update_traces(
        textinfo="label+value+percent parent",
        textfont={"color": CHART_TEXT_COLOR},
    )
    apply_original_chart_style(distribution_fig, "2026 Team Distribution", height=500)

    left, right = st.columns(2)
    render_column_plotly_chart(left, debutant_fig)
    render_column_plotly_chart(right, distribution_fig)


def render_goals_tab(outputs: dict[str, pd.DataFrame]) -> None:
    tournament_goals = outputs["tournament_goals"]
    team_goals = outputs["team_goals"]
    placement_summary = outputs["placement_goal_summary"]

    countries = sorted(team_goals["country"].dropna().unique())
    selected_country = st.selectbox(
        "Country",
        countries,
        index=countries.index("Brazil") if "Brazil" in countries else 0,
        key="historical_eda_goal_country",
    )

    fig = px.line(
        tournament_goals,
        x="edition",
        y="goals_per_match",
        markers=True,
        color="era",
        text=tournament_goals["goals_per_match"].map(lambda value: f"{value:.2f}"),
        category_orders={"era": ERA_LABELS},
        color_discrete_map=ERA_COLORS,
        labels={"edition": "Edition", "goals_per_match": "Goals per match"},
        title="Tournament Goals per Match",
    )
    fig.update_traces(textposition="top center")
    apply_original_chart_style(fig, "Tournament Goals per Match")
    render_plotly_chart(fig)

    country_goals = team_goals.loc[team_goals["country"].eq(selected_country)].sort_values("edition")
    fig = px.bar(
        country_goals,
        x="edition",
        y="gf",
        color="placement",
        text="gf",
        labels={"edition": "Edition", "gf": "Goals for"},
        title=f"{selected_country} Goals by Edition",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, f"{selected_country} Goals by Edition")
    render_plotly_chart(fig)

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
        title="Host Nation Finishes",
    )
    fig.update_traces(textposition="top center")
    apply_original_chart_style(fig, "Host Nation Finishes")
    fig.update_yaxes(autorange="reversed")
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
        title="Champion Follow-up Performance",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, f"World Cup Winners Placement the following Edition since {int(winners['edition'].min())}")
    fig.update_yaxes(autorange="reversed", title="Final Placement")
    fig.update_xaxes(title="Tournament Edition", tickangle=45)
    render_plotly_chart(fig)
    st.dataframe(winners, hide_index=True, width="stretch")


def render_correlations_tab(outputs: dict[str, pd.DataFrame]) -> None:
    summary = outputs["correlation_summary"]
    last_k_summary = outputs["last_k_summary"].iloc[0]
    render_metric_row(
        {
            "Last-k Lookback": int(last_k_summary["lookback"]),
            "Last-k Rows": int(last_k_summary["rows"]),
            "Last-k Correlation": f'{last_k_summary["correlation_with_finish_score"]:.3f}',
        }
    )
    fig = px.bar(
        summary,
        x="correlation_with_finish_score",
        y="feature",
        orientation="h",
        text=summary["correlation_with_finish_score"].map(lambda value: f"{value:.2f}"),
        labels={"correlation_with_finish_score": "Correlation", "feature": "Feature"},
        title="Correlation with Normalized Finish Score",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_original_chart_style(fig, "Correlation with Normalized Finish Score")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    render_plotly_chart(fig)
    st.dataframe(summary, hide_index=True, width="stretch")


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
        title="2026 Confederation Share",
    )
    distribution_fig.update_traces(textinfo="label+value+percent", textposition="inside")
    apply_original_chart_style(distribution_fig, "2026 Confederation Share", height=500)
    render_column_plotly_chart(left, distribution_fig)
    right.dataframe(qualified, hide_index=True, width="stretch")


def render_historical_eda_page() -> None:
    st.title("Analysis")
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
        render_participation_tab(participation)
    with tabs[1]:
        render_goals_tab(goals)
    with tabs[2]:
        render_host_tab(hosts)
    with tabs[3]:
        render_winner_followup_tab(winners)
    with tabs[4]:
        render_correlations_tab(correlations)
    with tabs[5]:
        render_2026_implications_tab(implications_2026)
