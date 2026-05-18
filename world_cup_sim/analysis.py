from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_cup_sim.constants import WORLD_CUP_ROOT


ERA_BINS = [1929, 1950, 1970, 1990, 2010, 2026]
ERA_LABELS = ["Early Era", "Golden Age", "Modern Era", "Contemporary", "Recent"]
CONFEDERATION_ORDER = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"]
PLACEMENT_SCORE_MAP = {
    "Winner": 7,
    "Runner-up": 6,
    "Third Place": 5,
    "Fourth Place": 4,
    "Semi-final": 4,
    "Quarter-final": 3,
    "Round of 16": 2,
    "Group Stage": 1,
}
HOST_NAME_ALIASES = {
    "usa": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "west germany": "germany",
}


def _world_nations_path(root: Path) -> Path:
    return root.parent / "world_nations" / "yearly_counts_1930_2026.csv"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_country_name(value: object) -> str:
    normalized = str(value).strip().lower()
    return HOST_NAME_ALIASES.get(normalized, normalized)


def parse_host_countries(value: object) -> set[str]:
    host_text = str(value)
    for separator in [" / ", "/", " & ", " and ", ","]:
        host_text = host_text.replace(separator, ",")
    return {
        normalize_country_name(host_country)
        for host_country in host_text.split(",")
        if host_country.strip()
    }


def add_era_column(df: pd.DataFrame, edition_column: str = "edition") -> pd.DataFrame:
    output = df.copy()
    output["era"] = pd.cut(
        pd.to_numeric(output[edition_column], errors="coerce"),
        bins=ERA_BINS,
        labels=ERA_LABELS,
    )
    return output


def load_historical_world_cup_data(root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load and normalize the historical datasets used by the notebook and EDA app."""
    data_root = Path(root) if root is not None else WORLD_CUP_ROOT
    history = pd.read_csv(data_root / "fifa_world_cup_history.csv").rename(
        columns={"Year": "edition", "Teams": "teams", "Host_Country": "host"}
    )
    teams = pd.read_csv(data_root / "all_editions" / "teams.csv").rename(
        columns={"year": "edition", "team": "country"}
    )
    app_2026_teams = _read_csv_if_exists(data_root / "2026" / "teams.csv").rename(
        columns={"year": "edition", "team": "country"}
    )
    if not app_2026_teams.empty:
        app_2026_columns = [
            column
            for column in [
                "team_id",
                "group_code",
                "is_host",
                "canonical_name",
                "flag_icon_code",
                "qualification_path",
                "world_cup_participations",
                "weighted_world_cup_participations",
                "weighted_world_cup_placement_score",
            ]
            if column in app_2026_teams.columns
        ]
        teams = teams.merge(
            app_2026_teams[app_2026_columns].drop_duplicates("team_id"),
            on="team_id",
            how="left",
            suffixes=("", "_2026"),
            validate="many_to_one",
        )
        for column in app_2026_columns:
            if column == "team_id":
                continue
            enriched_column = f"{column}_2026"
            if enriched_column in teams.columns:
                if column in teams.columns:
                    teams[column] = teams[column].fillna(teams[enriched_column])
                else:
                    teams[column] = teams[enriched_column]
                teams = teams.drop(columns=enriched_column)
    placement = pd.read_csv(data_root / "all_editions" / "placement.csv")
    results = pd.read_csv(data_root / "all_editions" / "results.csv").rename(
        columns={"country": "host_country", "team": "country"}
    )
    globe = _read_csv_if_exists(_world_nations_path(data_root)).rename(
        columns={
            "year": "edition",
            "official_state_count": "global_count",
            "official_state_count_AFC": "afc_count",
            "official_state_count_CAF": "caf_count",
            "official_state_count_CONCACAF": "concacaf_count",
            "official_state_count_CONMEBOL": "conmebol_count",
            "official_state_count_OFC": "ofc_count",
            "official_state_count_UEFA": "uefa_count",
        }
    )

    for frame in (history, teams, placement, results, globe):
        if "edition" in frame.columns:
            frame["edition"] = pd.to_numeric(frame["edition"], errors="coerce").astype("Int64")

    placement_context = history[["edition", "teams", "host"]].drop_duplicates("edition")
    placement = placement.merge(
        placement_context,
        on="edition",
        how="left",
        suffixes=("", "_history"),
        validate="many_to_one",
    )
    for column in ["teams", "host"]:
        history_column = f"{column}_history"
        if history_column in placement.columns:
            placement[column] = placement[column].fillna(placement[history_column])
            placement = placement.drop(columns=history_column)

    teams = add_era_column(teams)
    if "is_host" in teams.columns:
        teams["is_host"] = teams["is_host"].astype(str).str.upper().eq("TRUE")
    results = add_era_column(results)
    placement = add_era_column(placement)
    history = add_era_column(history)

    if {"edition", "tournament_id"}.issubset(placement.columns) and "start_elo" in placement.columns:
        placement["elo_rank"] = (
            placement.groupby(["edition", "tournament_id"])["start_elo"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )

    return {
        "history": history,
        "teams": teams,
        "placement": placement,
        "results": results,
        "globe": globe,
    }


def build_data_quality_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, df in datasets.items():
        if df.empty:
            rows.append(
                {
                    "dataset": name,
                    "rows": 0,
                    "columns": 0,
                    "duplicate_rows": 0,
                    "missing_cells": 0,
                    "missing_pct": 0.0,
                }
            )
            continue
        total_cells = int(df.shape[0] * df.shape[1])
        missing_cells = int(df.isna().sum().sum())
        rows.append(
            {
                "dataset": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": missing_cells,
                "missing_pct": round((missing_cells / total_cells) * 100, 2) if total_cells else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def build_participation_metrics(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    teams = datasets["teams"].copy()
    globe = datasets.get("globe", pd.DataFrame()).copy()

    participating_teams = (
        teams.dropna(subset=["edition", "country"])
        .groupby(["edition", "era", "tournament_id"], dropna=False, as_index=False, observed=True)
        .agg(team_counts=("country", "nunique"))
        .sort_values("edition")
        .reset_index(drop=True)
    )
    participating_teams["tournament_count"] = pd.cut(
        participating_teams["team_counts"],
        bins=[0, 16, 24, 32, 48, 100],
        labels=["<16-teams", "16-teams", "24-teams", "32-teams", "48-teams"],
        right=False,
        include_lowest=True,
    )
    if not globe.empty:
        globe_columns = [
            column
            for column in [
                "edition",
                "global_count",
                "afc_count",
                "caf_count",
                "concacaf_count",
                "conmebol_count",
                "ofc_count",
                "uefa_count",
            ]
            if column in globe.columns
        ]
        participating_teams = participating_teams.merge(
            globe[globe_columns],
            on="edition",
            how="left",
            validate="many_to_one",
        )
        participating_teams["global_participation_pct"] = (
            participating_teams["team_counts"] / participating_teams["global_count"] * 100
        ).round(2)

    confederation_by_edition = (
        teams.dropna(subset=["edition", "confederation", "country"])
        .groupby(["edition", "era", "confederation"], as_index=False, observed=True)
        .agg(participant_count=("country", "nunique"))
        .sort_values(["edition", "confederation"])
        .reset_index(drop=True)
    )
    avg_by_confederation = (
        confederation_by_edition.groupby("confederation", as_index=False, observed=True)
        .agg(
            avg_participant_count=("participant_count", "mean"),
            min_participant_count=("participant_count", "min"),
            max_participant_count=("participant_count", "max"),
            editions_with_participants=("edition", "nunique"),
        )
        .sort_values("avg_participant_count", ascending=False)
        .reset_index(drop=True)
    )
    avg_by_confederation["avg_participant_count"] = avg_by_confederation["avg_participant_count"].round(2)

    debutants = teams.dropna(subset=["country", "edition"]).copy()
    debutants["first_edition"] = debutants.groupby("country")["edition"].transform("min")
    debutants_by_edition = (
        debutants.loc[debutants["edition"].eq(debutants["first_edition"])]
        .groupby(["edition", "era"], as_index=False, observed=True)
        .agg(debutant_count=("country", "nunique"))
        .sort_values("edition")
        .reset_index(drop=True)
    )

    cumulative_participation = (
        confederation_by_edition.sort_values(["confederation", "edition"]).copy()
    )
    cumulative_participation["cumulative_participants"] = cumulative_participation.groupby(
        "confederation", observed=True
    )["participant_count"].cumsum()

    edition_confed = confederation_by_edition.merge(
        participating_teams[["edition", "team_counts"]],
        on="edition",
        how="left",
        validate="many_to_one",
    )
    edition_confed["field_share_pct"] = (
        edition_confed["participant_count"] / edition_confed["team_counts"] * 100
    ).round(2)

    latest_edition = int(teams["edition"].dropna().max())
    latest_distribution = (
        teams.loc[teams["edition"].eq(latest_edition)]
        .groupby("confederation", as_index=False, observed=True)
        .agg(team_count=("country", "nunique"))
        .sort_values("team_count", ascending=False)
        .reset_index(drop=True)
    )
    latest_distribution["edition"] = latest_edition

    team_distribution = teams.dropna(subset=["edition", "country", "confederation"]).copy()
    team_distribution["edition"] = pd.to_numeric(team_distribution["edition"], errors="coerce").astype("Int64")
    team_distribution = team_distribution.sort_values(["country", "edition"]).reset_index(drop=True)
    team_distribution["prior_participations"] = team_distribution.groupby("country", observed=True).cumcount()
    team_distribution["participation_count"] = team_distribution["prior_participations"] + 1
    team_distribution["is_first_timer"] = team_distribution["prior_participations"].eq(0)
    team_distribution["country_label"] = team_distribution["country"].where(
        ~team_distribution["is_first_timer"],
        team_distribution["country"] + " ★",
    )
    team_distribution["team_value"] = 1
    latest_team_distribution = (
        team_distribution.loc[team_distribution["edition"].eq(latest_edition)]
        .sort_values(["confederation", "country"])
        .reset_index(drop=True)
    )

    return {
        "participating_teams": participating_teams,
        "confederation_by_edition": confederation_by_edition,
        "avg_by_confederation": avg_by_confederation,
        "debutants_by_edition": debutants_by_edition,
        "cumulative_participation": cumulative_participation,
        "edition_confederation_share": edition_confed,
        "latest_distribution": latest_distribution,
        "latest_team_distribution": latest_team_distribution,
    }


def build_goal_metrics(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    results = datasets["results"].copy()
    placement = datasets["placement"].copy()

    for column in ["team_score", "opponent_score"]:
        results[column] = pd.to_numeric(results[column], errors="coerce").fillna(0)

    matches = (
        results.groupby(["edition", "era", "tournament_id"], dropna=False, as_index=False, observed=True)
        .agg(total_matches=("match_id", "nunique"))
    )
    team_goals = (
        results.groupby(["edition", "era", "tournament_id", "country"], dropna=False, as_index=False, observed=True)
        .agg(
            gf=("team_score", "sum"),
            ga=("opponent_score", "sum"),
            team_matches=("match_id", "nunique"),
        )
    )
    team_goals["goal_difference"] = team_goals["gf"] - team_goals["ga"]
    team_goals["goals_per_game"] = (team_goals["gf"] / team_goals["team_matches"]).round(3)
    team_goals["goals_against_per_game"] = (team_goals["ga"] / team_goals["team_matches"]).round(3)
    team_goals["gls_rank"] = team_goals.groupby(["edition", "tournament_id"])["gf"].rank(
        method="min", ascending=False
    )
    team_goals = team_goals.merge(
        placement[["edition", "country", "placement", "position"]],
        on=["edition", "country"],
        how="left",
    )
    team_goals = team_goals.merge(matches[["edition", "total_matches"]], on="edition", how="left")

    tournament_goals = (
        results.drop_duplicates(["edition", "match_id"])
        .groupby(["edition", "era", "tournament_id"], as_index=False, observed=True)
        .agg(
            total_goals=("team_score", lambda series: np.nan),
            total_matches=("match_id", "nunique"),
        )
    )
    totals = (
        results.groupby(["edition", "era", "tournament_id"], as_index=False, observed=True)
        .agg(total_goals=("team_score", "sum"))
    )
    tournament_goals = tournament_goals.drop(columns=["total_goals"]).merge(
        totals, on=["edition", "era", "tournament_id"], how="left"
    )
    tournament_goals["goals_per_match"] = (
        tournament_goals["total_goals"] / tournament_goals["total_matches"]
    ).round(3)

    placement_goal_summary = (
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

    return {
        "team_goals": team_goals,
        "tournament_goals": tournament_goals.sort_values("edition").reset_index(drop=True),
        "placement_goal_summary": placement_goal_summary,
    }


def build_host_effect_metrics(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    teams = datasets["teams"].copy()
    history = datasets["history"].copy()
    goal_metrics = build_goal_metrics(datasets)
    team_goals = goal_metrics["team_goals"]

    host_lookup = history[["edition", "host"]].drop_duplicates("edition").copy()
    host_lookup["host_names"] = host_lookup["host"].apply(parse_host_countries)
    hosts = teams.copy()
    hosts["normalized_country"] = hosts["country"].apply(normalize_country_name)
    hosts = hosts.merge(host_lookup[["edition", "host", "host_names"]], on="edition", how="left")
    parsed_host_flag = hosts.apply(
        lambda row: row["normalized_country"] in row["host_names"]
        if isinstance(row["host_names"], set)
        else False,
        axis=1,
    )
    existing_host_flag = (
        hosts["is_host"].fillna(False).astype(bool)
        if "is_host" in hosts.columns
        else pd.Series(False, index=hosts.index)
    )
    hosts["is_host"] = existing_host_flag | parsed_host_flag
    hosts = hosts.loc[hosts["is_host"]].copy()

    editions = sorted(teams["edition"].dropna().astype(int).unique())
    next_edition_map = dict(zip(editions[:-1], editions[1:]))
    hosts["next_edition"] = hosts["edition"].astype(int).map(next_edition_map)
    next_lookup = teams[["edition", "country", "placement", "position"]].rename(
        columns={
            "edition": "next_edition",
            "placement": "next_placement",
            "position": "next_position",
        }
    )
    hosts = hosts.merge(next_lookup, on=["country", "next_edition"], how="left")
    hosts.loc[hosts["next_edition"].notna() & hosts["next_placement"].isna(), "next_placement"] = "DNQ"
    hosts.loc[hosts["next_edition"].isna(), "next_placement"] = "TBD"
    hosts["next_position"] = pd.to_numeric(hosts["next_position"], errors="coerce").fillna(0)
    hosts = hosts.merge(
        team_goals[["edition", "country", "gf", "ga", "goal_difference"]],
        on=["edition", "country"],
        how="left",
    )

    host_summary = pd.DataFrame(
        [
            {
                "host_editions": int(hosts["edition"].nunique()),
                "host_teams": int(len(hosts)),
                "avg_position": round(float(pd.to_numeric(hosts["position"], errors="coerce").mean()), 2),
                "median_position": round(float(pd.to_numeric(hosts["position"], errors="coerce").median()), 2),
                "avg_goals_for": round(float(pd.to_numeric(hosts["gf"], errors="coerce").mean()), 2),
                "avg_goals_against": round(float(pd.to_numeric(hosts["ga"], errors="coerce").mean()), 2),
                "titles": int(hosts["placement"].eq("Winner").sum()),
                "top_four_finishes": int(pd.to_numeric(hosts["position"], errors="coerce").le(4).sum()),
            }
        ]
    )
    return {
        "hosts": hosts.sort_values(["edition", "country"]).reset_index(drop=True),
        "host_summary": host_summary,
    }


def build_winner_followup_metrics(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    placement = datasets["placement"].copy()
    winners = placement.loc[placement["placement"].eq("Winner")].copy()
    columns = [
        "edition",
        "country",
        "position",
        "next_edition",
        "next_placement",
        "next_position",
        "start_elo",
        "finish_elo",
        "elo_change",
    ]
    available_columns = [column for column in columns if column in winners.columns]
    winners = winners[available_columns].sort_values("edition").reset_index(drop=True)
    if "next_position" in winners.columns:
        winners["next_position"] = pd.to_numeric(winners["next_position"], errors="coerce")
    return winners


def _build_outcome_frame(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    placement = datasets["placement"].copy().sort_values(["country", "edition"]).reset_index(drop=True)
    placement["position"] = pd.to_numeric(placement["position"], errors="coerce")
    placement["edition_team_count"] = placement.groupby("edition")["country"].transform("count")
    placement["edition_finish_scale"] = placement.groupby("edition")["position"].transform("max")
    placement["edition_finish_scale"] = placement[["edition_team_count", "edition_finish_scale"]].max(axis=1)
    placement["finish_score"] = 1 - (
        (placement["position"] - 1) / (placement["edition_finish_scale"] - 1)
    )
    placement["placement_score"] = placement["placement"].map(PLACEMENT_SCORE_MAP)
    for column in ["matches_played", "gs", "ga", "start_elo", "finish_elo", "elo_change"]:
        if column in placement.columns:
            placement[column] = pd.to_numeric(placement[column], errors="coerce")
    placement["goal_difference"] = placement["gs"] - placement["ga"]
    placement["goals_per_match"] = placement["gs"] / placement["matches_played"]
    placement["goals_against_per_match"] = placement["ga"] / placement["matches_played"]
    placement["goal_difference_per_match"] = placement["goal_difference"] / placement["matches_played"]

    host_metrics = build_host_effect_metrics(datasets)
    host_pairs = set(host_metrics["hosts"][["edition", "country"]].itertuples(index=False, name=None))
    placement["is_host"] = [
        int((edition, country) in host_pairs)
        for edition, country in zip(placement["edition"], placement["country"])
    ]

    team_history = placement.groupby("country", group_keys=False)
    placement["prior_world_cup_participations"] = team_history.cumcount()
    placement["previous_finish_score"] = team_history["finish_score"].shift(1)
    placement["prior_avg_finish_score"] = team_history["finish_score"].transform(
        lambda series: series.shift(1).expanding().mean()
    )
    placement["prior_best_finish_score"] = team_history["finish_score"].transform(
        lambda series: series.shift(1).expanding().max()
    )
    placement["prior_avg_goals_per_match"] = team_history["goals_per_match"].transform(
        lambda series: series.shift(1).expanding().mean()
    )
    placement["prior_avg_goal_diff_per_match"] = team_history["goal_difference_per_match"].transform(
        lambda series: series.shift(1).expanding().mean()
    )
    return placement


def build_correlation_metrics(
    datasets: dict[str, pd.DataFrame],
    lookback: int = 5,
) -> dict[str, pd.DataFrame]:
    outcome = _build_outcome_frame(datasets)
    feature_columns = [
        "start_elo",
        "elo_rank",
        "is_host",
        "prior_world_cup_participations",
        "previous_finish_score",
        "prior_avg_finish_score",
        "prior_best_finish_score",
        "prior_avg_goals_per_match",
        "prior_avg_goal_diff_per_match",
        "goals_per_match",
        "goal_difference_per_match",
        "elo_change",
    ]
    available = [column for column in feature_columns if column in outcome.columns]
    corr_rows = []
    for column in available:
        valid = outcome[[column, "finish_score"]].apply(pd.to_numeric, errors="coerce").dropna()
        corr_rows.append(
            {
                "feature": column,
                "correlation_with_finish_score": round(float(valid[column].corr(valid["finish_score"])), 3)
                if len(valid) > 2
                else np.nan,
                "rows": int(len(valid)),
            }
        )
    correlation_summary = (
        pd.DataFrame(corr_rows)
        .assign(abs_correlation=lambda frame: frame["correlation_with_finish_score"].abs())
        .sort_values("abs_correlation", ascending=False)
        .drop(columns=["abs_correlation"])
        .reset_index(drop=True)
    )

    history = outcome.copy()
    all_editions = sorted(history["edition"].dropna().astype(int).unique().tolist())
    team_edition_lookup = {
        (int(row.edition), str(row.country)): row
        for row in history.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for row in history[["edition", "country", "finish_score", "position"]].itertuples(index=False):
        edition = int(row.edition)
        country = str(row.country)
        prior_editions = [prior for prior in all_editions if prior < edition][-lookback:]
        prior_rows = [team_edition_lookup.get((prior, country)) for prior in prior_editions]
        appeared_rows = [prior_row for prior_row in prior_rows if prior_row is not None]
        finish_scores = [prior_row.finish_score for prior_row in appeared_rows]
        rows.append(
            {
                "edition": edition,
                "country": country,
                "lookback": lookback,
                "prior_editions": len(prior_editions),
                "prior_appearances": len(appeared_rows),
                "prior_appearance_rate": round(len(appeared_rows) / len(prior_editions), 3)
                if prior_editions
                else np.nan,
                "last_k_avg_finish_score": float(np.nanmean(finish_scores)) if finish_scores else np.nan,
                "current_finish_score": row.finish_score,
                "current_position": row.position,
            }
        )
    last_k = pd.DataFrame(rows)
    valid_last_k = last_k[["last_k_avg_finish_score", "current_finish_score"]].dropna()
    last_k_summary = pd.DataFrame(
        [
            {
                "lookback": lookback,
                "rows": int(len(valid_last_k)),
                "correlation_with_finish_score": round(
                    float(valid_last_k["last_k_avg_finish_score"].corr(valid_last_k["current_finish_score"])), 3
                )
                if len(valid_last_k) > 2
                else np.nan,
            }
        ]
    )
    return {
        "outcome_frame": outcome,
        "correlation_summary": correlation_summary,
        "last_k_features": last_k,
        "last_k_summary": last_k_summary,
    }


def build_2026_implication_tables(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    teams = datasets["teams"].copy()
    latest_edition = int(teams["edition"].dropna().max())
    latest = teams.loc[teams["edition"].eq(latest_edition)].copy()
    columns = [
        "team_id",
        "country",
        "confederation",
        "group_code",
        "placement",
        "position",
        "matches_played",
    ]
    available = [column for column in columns if column in latest.columns]
    latest = latest[available].sort_values(["confederation", "country"]).reset_index(drop=True)

    participation = build_participation_metrics(datasets)
    distribution = participation["latest_distribution"]
    history = datasets["history"].copy()
    latest_history = history.loc[history["edition"].eq(latest_edition)].copy()

    return {
        "qualified_teams": latest,
        "confederation_distribution": distribution,
        "edition_context": latest_history.reset_index(drop=True),
    }
