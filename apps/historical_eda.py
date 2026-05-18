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
    ]
    filtered_confed = confed[confed["edition"].between(year_range[0], year_range[1])]

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
        labels={"edition": "Edition", "team_counts": "Teams", "tournament_count": "Field size"},
        title="World Cup Field Size by Edition",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        filtered_confed,
        x="edition",
        y="participant_count",
        color="confederation",
        markers=True,
        category_orders={"confederation": CONFEDERATION_ORDER},
        color_discrete_map=CONFEDERATION_COLORS,
        labels={"edition": "Edition", "participant_count": "Teams"},
        title="Participation by Confederation",
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            debutants,
            x="edition",
            y="debutant_count",
            color="era",
            labels={"edition": "Edition", "debutant_count": "Debutants"},
            title="Debutants by Edition",
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        px.treemap(
            latest,
            path=["confederation"],
            values="team_count",
            color="confederation",
            color_discrete_map=CONFEDERATION_COLORS,
            title="2026 Team Distribution",
        ),
        use_container_width=True,
    )


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
        labels={"edition": "Edition", "goals_per_match": "Goals per match"},
        title="Tournament Goals per Match",
    )
    st.plotly_chart(fig, use_container_width=True)

    country_goals = team_goals.loc[team_goals["country"].eq(selected_country)].sort_values("edition")
    fig = px.bar(
        country_goals,
        x="edition",
        y="gf",
        color="placement",
        labels={"edition": "Edition", "gf": "Goals for"},
        title=f"{selected_country} Goals by Edition",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        placement_summary,
        hide_index=True,
        use_container_width=True,
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
        hover_name="country",
        color_discrete_map=CONFEDERATION_COLORS,
        labels={"edition": "Edition", "position": "Final position", "gf": "Goals for"},
        title="Host Nation Finishes",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
    )


def render_winner_followup_tab(winners: pd.DataFrame) -> None:
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
        labels={"edition": "Title edition", "next_position": "Next edition position"},
        title="Champion Follow-up Performance",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(winners, hide_index=True, use_container_width=True)


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
        labels={"correlation_with_finish_score": "Correlation", "feature": "Feature"},
        title="Correlation with Normalized Finish Score",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary, hide_index=True, use_container_width=True)


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
    left.plotly_chart(
        px.pie(
            distribution,
            names="confederation",
            values="team_count",
            color="confederation",
            color_discrete_map=CONFEDERATION_COLORS,
            title="2026 Confederation Share",
        ),
        use_container_width=True,
    )
    right.dataframe(qualified, hide_index=True, use_container_width=True)


def render_historical_eda_page() -> None:
    st.title("Historical World Cup EDA")
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
        st.dataframe(quality, hide_index=True, use_container_width=True)

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
