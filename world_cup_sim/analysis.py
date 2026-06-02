from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_cup_sim.constants import WORLD_CUP_ROOT
from world_cup_sim.shared import normalize_historical_team_name


ERA_BINS = [1929, 1950, 1970, 1990, 2010, 2026]
ERA_LABELS = ["Early Era", "Golden Age", "Modern Era", "Contemporary", "Recent"]
CONFEDERATION_ORDER = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"]
MAIN_BRACKET_STAGES = ["Round of 16", "Quarter-final", "Semi-final", "Final"]
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
PARTICIPATION_TEAM_ID_ALIASES = {
    "che": "sui",
    "dza": "alg",
    "hrv": "cro",
    "hti": "hai",
    "prt": "por",
    "pry": "par",
    "ury": "uru",
    "zaf": "rsa",
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


def participation_identity_key(row: pd.Series) -> str:
    """Return a stable team identity for historical participation counts."""
    team_id = row.get("team_id", "")
    if pd.notna(team_id) and str(team_id).strip():
        normalized_team_id = str(team_id).strip().lower()
        return PARTICIPATION_TEAM_ID_ALIASES.get(normalized_team_id, normalized_team_id)

    for column_name in ("canonical_name", "country"):
        value = row.get(column_name, "")
        if pd.notna(value) and str(value).strip():
            return normalize_historical_team_name(str(value))
    return ""


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
    app_2026_elo = _read_csv_if_exists(data_root / "2026" / "elo.csv").rename(
        columns={"year": "edition", "team": "country", "elo_start": "start_elo"}
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

    for frame in (history, teams, placement, results, globe, app_2026_elo):
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
        "elo_2026": app_2026_elo,
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

    teams_with_identity = teams.dropna(subset=["country", "edition"]).copy()
    teams_with_identity["participation_key"] = teams_with_identity.apply(participation_identity_key, axis=1)
    teams_with_identity = teams_with_identity.loc[teams_with_identity["participation_key"].ne("")].copy()

    debutants = teams_with_identity.copy()
    debutants["first_edition"] = debutants.groupby("participation_key")["edition"].transform("min")
    debutants_by_edition = (
        debutants.loc[debutants["edition"].eq(debutants["first_edition"])]
        .groupby(["edition", "era"], as_index=False, observed=True)
        .agg(debutant_count=("participation_key", "nunique"))
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

    team_distribution = teams_with_identity.dropna(subset=["edition", "country", "confederation"]).copy()
    team_distribution["edition"] = pd.to_numeric(team_distribution["edition"], errors="coerce").astype("Int64")
    team_distribution = team_distribution.sort_values(["participation_key", "edition"]).reset_index(drop=True)
    team_distribution["prior_participations"] = team_distribution.groupby(
        "participation_key", observed=True
    ).cumcount()
    team_distribution["participation_count"] = team_distribution["prior_participations"] + 1
    team_distribution["is_first_timer"] = team_distribution["prior_participations"].eq(0)
    team_distribution["country_label"] = team_distribution["country"].where(
        ~team_distribution["is_first_timer"],
        team_distribution["country"] + " *",
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

    match_scorelines = results.drop_duplicates(["edition", "match_id"]).copy()
    match_scorelines["score_low"] = match_scorelines[["team_score", "opponent_score"]].min(axis=1).astype(int)
    match_scorelines["score_high"] = match_scorelines[["team_score", "opponent_score"]].max(axis=1).astype(int)
    match_scorelines["total_goals"] = match_scorelines["score_low"] + match_scorelines["score_high"]
    match_scorelines["scoreline"] = (
        match_scorelines["score_low"].astype(str) + "-" + match_scorelines["score_high"].astype(str)
    )
    scoreline_order = (
        match_scorelines[["scoreline", "total_goals", "score_low", "score_high"]]
        .drop_duplicates()
        .sort_values(["total_goals", "score_low", "score_high"], kind="stable")
        .reset_index(drop=True)
    )
    scoreline_order["scoreline_rank"] = scoreline_order.index + 1
    match_scorelines = match_scorelines.merge(
        scoreline_order[["scoreline", "scoreline_rank"]],
        on="scoreline",
        how="left",
        validate="many_to_one",
    )
    match_scorelines["score"] = (
        match_scorelines["team_score"].astype(int).astype(str)
        + "-"
        + match_scorelines["opponent_score"].astype(int).astype(str)
    )
    match_scoreline_columns = [
        "edition",
        "era",
        "tournament_id",
        "match_id",
        "match_number",
        "date",
        "stage",
        "country",
        "opponent",
        "team_score",
        "opponent_score",
        "score",
        "score_low",
        "score_high",
        "total_goals",
        "scoreline",
        "scoreline_rank",
    ]
    match_scorelines = (
        match_scorelines[[column for column in match_scoreline_columns if column in match_scorelines.columns]]
        .sort_values(["edition", "match_number", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    winner_lookup = (
        placement.loc[placement["placement"].eq("Winner"), ["edition", "country"]]
        .drop_duplicates()
        .assign(is_winner=True)
    )
    winner_match_scorelines = results.merge(
        winner_lookup,
        on=["edition", "country"],
        how="inner",
        validate="many_to_one",
    ).copy()
    winner_match_scorelines["score_low"] = winner_match_scorelines[["team_score", "opponent_score"]].min(axis=1).astype(int)
    winner_match_scorelines["score_high"] = winner_match_scorelines[["team_score", "opponent_score"]].max(axis=1).astype(int)
    winner_match_scorelines["total_goals"] = winner_match_scorelines["score_low"] + winner_match_scorelines["score_high"]
    winner_match_scorelines["scoreline"] = (
        winner_match_scorelines["score_low"].astype(str) + "-" + winner_match_scorelines["score_high"].astype(str)
    )
    winner_match_scorelines = winner_match_scorelines.merge(
        scoreline_order[["scoreline", "scoreline_rank"]],
        on="scoreline",
        how="left",
        validate="many_to_one",
    )
    winner_match_scorelines["score"] = (
        winner_match_scorelines["team_score"].astype(int).astype(str)
        + "-"
        + winner_match_scorelines["opponent_score"].astype(int).astype(str)
    )
    winner_match_scorelines = (
        winner_match_scorelines[[column for column in match_scoreline_columns if column in winner_match_scorelines.columns]]
        .sort_values(["edition", "match_number", "match_id"], kind="stable")
        .reset_index(drop=True)
    )

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
    placement_goal_columns = [
        column
        for column in ["edition", "country", "placement", "position", "elo_rank"]
        if column in placement.columns
    ]
    team_goals = team_goals.merge(
        placement[placement_goal_columns],
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
        "match_scorelines": match_scorelines,
        "winner_match_scorelines": winner_match_scorelines,
    }


def build_host_effect_metrics(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    teams = datasets["teams"].copy()
    history = datasets["history"].copy()
    placement = datasets["placement"].copy()
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
    has_host_metadata = hosts["host_names"].apply(lambda value: isinstance(value, set) and bool(value))
    existing_host_flag = (
        hosts["is_host"].fillna(False).astype(bool)
        if "is_host" in hosts.columns
        else pd.Series(False, index=hosts.index)
    )
    hosts["is_host"] = parsed_host_flag | (existing_host_flag & ~has_host_metadata)
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
    hosts.loc[hosts["next_edition"].notna() & hosts["next_placement"].isna(), "next_placement"] = "DNP"
    hosts.loc[hosts["next_edition"].isna(), "next_placement"] = "TBD"
    hosts["next_position"] = pd.to_numeric(hosts["next_position"], errors="coerce").fillna(0)
    hosts = hosts.merge(
        team_goals[["edition", "country", "gf", "ga", "goal_difference"]],
        on=["edition", "country"],
        how="left",
    )
    if "start_elo" in placement.columns:
        hosts = hosts.merge(
            placement[["edition", "country", "start_elo"]].drop_duplicates(["edition", "country"]),
            on=["edition", "country"],
            how="left",
        )
    if "start_elo" not in hosts.columns:
        hosts["start_elo"] = pd.NA

    elo_2026 = datasets.get("elo_2026", pd.DataFrame()).copy()
    if not elo_2026.empty and {"edition", "team_id", "start_elo"}.issubset(elo_2026.columns):
        hosts = hosts.merge(
            elo_2026[["edition", "team_id", "start_elo"]]
            .drop_duplicates(["edition", "team_id"])
            .rename(columns={"start_elo": "start_elo_2026"}),
            on=["edition", "team_id"],
            how="left",
        )
        hosts["start_elo"] = hosts["start_elo"].fillna(hosts["start_elo_2026"])
        hosts = hosts.drop(columns="start_elo_2026")

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
        "era",
        "country",
        "position",
        "next_edition",
        "next_placement",
        "next_position",
        "start_elo",
        "elo_rank",
        "finish_elo",
        "elo_change",
    ]
    available_columns = [column for column in columns if column in winners.columns]
    winners = winners[available_columns].sort_values("edition").reset_index(drop=True)
    if "next_position" in winners.columns:
        winners["next_position"] = pd.to_numeric(winners["next_position"], errors="coerce")
    return winners


def _build_match_elo_frame(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = datasets["results"].copy()
    placement = datasets["placement"].copy()
    if results.empty or placement.empty:
        return pd.DataFrame()

    match_columns = [
        "edition",
        "era",
        "tournament_id",
        "match_id",
        "match_number",
        "stage",
        "country",
        "opponent",
        "team_score",
        "opponent_score",
        "decided_by_shootout",
        "shootout_winner",
    ]
    matches = (
        results[[column for column in match_columns if column in results.columns]]
        .drop_duplicates(["edition", "match_id"])
        .copy()
    )
    for column in ["team_score", "opponent_score"]:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")

    elo_lookup = placement[["edition", "country", "start_elo", "elo_rank"]].drop_duplicates(["edition", "country"])
    matches = matches.merge(
        elo_lookup.rename(
            columns={
                "country": "country",
                "start_elo": "country_start_elo",
                "elo_rank": "country_elo_rank",
            }
        ),
        on=["edition", "country"],
        how="left",
        validate="many_to_one",
    )
    matches = matches.merge(
        elo_lookup.rename(
            columns={
                "country": "opponent",
                "start_elo": "opponent_start_elo",
                "elo_rank": "opponent_elo_rank",
            }
        ),
        on=["edition", "opponent"],
        how="left",
        validate="many_to_one",
    )
    for column in ["country_start_elo", "opponent_start_elo", "country_elo_rank", "opponent_elo_rank"]:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

    shootout_winner = matches.get("shootout_winner", pd.Series(pd.NA, index=matches.index)).fillna("")
    decided_by_shootout = (
        matches.get("decided_by_shootout", pd.Series(False, index=matches.index))
        .fillna(False)
        .astype(str)
        .str.upper()
        .isin({"TRUE", "1", "YES"})
    )
    score_winner = np.select(
        [
            matches["team_score"].gt(matches["opponent_score"]),
            matches["team_score"].lt(matches["opponent_score"]),
        ],
        [matches["country"], matches["opponent"]],
        default="",
    )
    matches["winner"] = np.where(decided_by_shootout & shootout_winner.ne(""), shootout_winner, score_winner)
    winner = matches["winner"].fillna("")
    country = matches["country"].fillna("")
    opponent = matches["opponent"].fillna("")
    matches["loser"] = np.select(
        [
            winner.eq(country),
            winner.eq(opponent),
        ],
        [matches["opponent"], matches["country"]],
        default="",
    )
    matches["actual_score"] = np.select(
        [
            winner.eq(country),
            winner.eq(opponent),
            matches["team_score"].eq(matches["opponent_score"]),
        ],
        [1.0, 0.0, 0.5],
        default=np.nan,
    )
    matches["elo_difference"] = matches["country_start_elo"] - matches["opponent_start_elo"]
    matches["winner_elo_rank"] = np.select(
        [
            winner.eq(country),
            winner.eq(opponent),
        ],
        [matches["country_elo_rank"], matches["opponent_elo_rank"]],
        default=np.nan,
    )
    matches["loser_elo_rank"] = np.select(
        [
            winner.eq(country),
            winner.eq(opponent),
        ],
        [matches["opponent_elo_rank"], matches["country_elo_rank"]],
        default=np.nan,
    )
    matches["winner"] = matches["winner"].replace("", pd.NA)
    matches["loser"] = matches["loser"].replace("", pd.NA)
    matches["is_upset"] = matches["winner_elo_rank"] > matches["loser_elo_rank"]
    return matches.sort_values(["edition", "match_number", "match_id"], kind="stable").reset_index(drop=True)


def _summarize_upsets_by_stage(match_elo_frame: pd.DataFrame) -> pd.DataFrame:
    if match_elo_frame.empty:
        return pd.DataFrame(columns=["stage", "matches", "upsets", "upset_pct"])
    knockouts = match_elo_frame.loc[
        match_elo_frame["stage"].isin(MAIN_BRACKET_STAGES)
        & match_elo_frame["winner"].notna()
        & match_elo_frame["winner_elo_rank"].notna()
        & match_elo_frame["loser_elo_rank"].notna()
    ].copy()
    summary = (
        knockouts.groupby("stage", as_index=False)
        .agg(matches=("match_id", "nunique"), upsets=("is_upset", "sum"))
    )
    summary["upset_pct"] = (summary["upsets"] / summary["matches"] * 100).round(1)
    stage_order = pd.Categorical(summary["stage"], categories=MAIN_BRACKET_STAGES, ordered=True)
    return summary.assign(stage_order=stage_order).sort_values("stage_order").drop(columns="stage_order").reset_index(drop=True)


def _summarize_champion_elo_rank(placement: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    winners = placement.loc[placement["placement"].eq("Winner"), ["edition", "country", "elo_rank"]].copy()
    winners["top_elo_won"] = pd.to_numeric(winners["elo_rank"], errors="coerce").eq(1)
    winners["result"] = np.where(winners["top_elo_won"], "Top Elo won", "Other champion")
    by_edition = winners.sort_values("edition").reset_index(drop=True)
    rows = []
    total_editions = int(len(by_edition))
    for result in ["Top Elo won", "Other champion"]:
        editions = by_edition.loc[by_edition["result"].eq(result), "edition"].dropna().astype(int).tolist()
        rows.append(
            {
                "result": result,
                "editions": ", ".join(map(str, editions)),
                "edition_count": len(editions),
                "pct": round((len(editions) / total_editions * 100), 1) if total_editions else 0.0,
            }
        )
    return by_edition, pd.DataFrame(rows)


def _summarize_elo_predictive_power(match_elo_frame: pd.DataFrame) -> pd.DataFrame:
    if match_elo_frame.empty:
        return pd.DataFrame(columns=["edition", "era", "matches", "elo_result_corr"])
    valid = match_elo_frame.dropna(subset=["elo_difference", "actual_score"]).copy()
    rows = []
    for (edition, era), rows_for_edition in valid.groupby(["edition", "era"], dropna=False, observed=True):
        correlation = (
            rows_for_edition["elo_difference"].corr(rows_for_edition["actual_score"], method="pearson")
            if len(rows_for_edition) > 2 and rows_for_edition["elo_difference"].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "edition": int(edition),
                "era": era,
                "matches": int(len(rows_for_edition)),
                "elo_result_corr": correlation,
            }
        )
    return pd.DataFrame(rows).sort_values("edition").reset_index(drop=True)


def build_elo_metrics(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return tournament-start Elo analysis tables for historical World Cups."""
    match_elo_frame = _build_match_elo_frame(datasets)
    champion_by_edition, champion_summary = _summarize_champion_elo_rank(datasets["placement"].copy())
    return {
        "match_elo_frame": match_elo_frame,
        "upset_by_stage": _summarize_upsets_by_stage(match_elo_frame),
        "champion_elo_rank_by_edition": champion_by_edition,
        "champion_elo_rank_summary": champion_summary,
        "elo_predictive_power_by_edition": _summarize_elo_predictive_power(match_elo_frame),
    }


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
    placement["previous_position"] = team_history["position"].shift(1)
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
    placement = _add_lead_in_form_features(placement)
    return placement


def _safe_mean(values: list[Any] | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = numeric.dropna()
    return float(valid.mean()) if not valid.empty else np.nan


def _safe_sum(values: list[Any] | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = numeric.dropna()
    return float(valid.sum()) if not valid.empty else np.nan


def _weighted_mean(values: list[Any] | pd.Series, weights: list[Any] | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    weight_values = pd.Series(weights, index=numeric.index, dtype=float)
    valid = numeric.notna()
    return float(np.average(numeric[valid], weights=weight_values[valid])) if valid.any() else np.nan


def _add_lead_in_form_features(outcome: pd.DataFrame, match_window: int = 10) -> pd.DataFrame:
    results_root = WORLD_CUP_ROOT / "by_confederation"
    edition_start_dates: dict[int, pd.Timestamp] = {}
    for results_file in sorted(WORLD_CUP_ROOT.glob("[0-9][0-9][0-9][0-9]/results.csv")):
        edition = int(results_file.parent.name)
        edition_results = pd.read_csv(results_file, usecols=["date"])
        edition_start_dates[edition] = pd.to_datetime(edition_results["date"], errors="coerce").min()

    feature_columns = [
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
    if not results_root.exists():
        for column in feature_columns:
            outcome[column] = np.nan
        outcome["edition_start_date"] = outcome["edition"].map(edition_start_dates)
        return outcome

    team_result_columns = [
        "date",
        "team",
        "team_score",
        "opponent_score",
        "result",
        "team_elo_delta",
    ]
    team_result_frames = []
    for team_results_file in sorted(results_root.glob("*/*/results.csv")):
        team_results = pd.read_csv(team_results_file, usecols=team_result_columns)
        if not team_results.empty:
            team_result_frames.append(team_results)
    if not team_result_frames:
        for column in feature_columns:
            outcome[column] = np.nan
        outcome["edition_start_date"] = outcome["edition"].map(edition_start_dates)
        return outcome

    team_results_history = pd.concat(team_result_frames, ignore_index=True)
    team_results_history["date"] = pd.to_datetime(team_results_history["date"], errors="coerce")
    for column in ["team_score", "opponent_score", "team_elo_delta"]:
        team_results_history[column] = pd.to_numeric(team_results_history[column], errors="coerce")

    normalized_result = (
        team_results_history["result"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"w": "win", "d": "draw", "l": "loss", "win": "win", "draw": "draw", "loss": "loss"})
    )
    score_based_result = pd.Series(
        np.select(
            [
                team_results_history["team_score"] > team_results_history["opponent_score"],
                team_results_history["team_score"] == team_results_history["opponent_score"],
                team_results_history["team_score"] < team_results_history["opponent_score"],
            ],
            ["win", "draw", "loss"],
            default=None,
        ),
        index=team_results_history.index,
        dtype="object",
    )
    team_results_history["normalized_result"] = normalized_result.fillna(score_based_result)
    team_results_history["actual_score"] = team_results_history["normalized_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0})
    team_results_history["win_indicator"] = team_results_history["normalized_result"].eq("win").astype(float)
    team_results_history["goal_difference"] = team_results_history["team_score"] - team_results_history["opponent_score"]
    team_results_history = team_results_history.dropna(
        subset=["date", "team", "team_score", "opponent_score", "actual_score"]
    ).copy()
    team_results_lookup = {
        str(team): matches.sort_values("date", kind="stable").reset_index(drop=True)
        for team, matches in team_results_history.groupby("team", sort=False)
    }
    country_form_aliases = {"China PR": "China", "DR Congo": "Zaire"}

    form_rows = []
    for row in outcome[["edition", "country"]].itertuples(index=False):
        edition_start_date = edition_start_dates.get(int(row.edition))
        lookup_country = country_form_aliases.get(str(row.country), str(row.country))
        team_matches = team_results_lookup.get(lookup_country)
        recent = (
            pd.DataFrame(columns=team_results_history.columns)
            if team_matches is None or pd.isna(edition_start_date)
            else team_matches[team_matches["date"] < edition_start_date].tail(match_window).copy()
        )
        form_row = {"edition": int(row.edition), "country": str(row.country), **{column: np.nan for column in feature_columns}}
        if not recent.empty:
            match_count = len(recent)
            weights = pd.Series(np.arange(1, match_count + 1, dtype=float), index=recent.index)
            goals_for = recent["team_score"].sum()
            goals_against = recent["opponent_score"].sum()
            goal_difference = goals_for - goals_against
            elo_change = recent["team_elo_delta"].sum(min_count=1)
            form_row.update(
                {
                    "form_l10_matches": int(match_count),
                    "form_l10_win_pct": float(recent["win_indicator"].mean()),
                    "form_l10_goals_for": float(goals_for),
                    "form_l10_goals_against": float(goals_against),
                    "form_l10_goal_difference": float(goal_difference),
                    "form_l10_goals_per_match": float(goals_for / match_count),
                    "form_l10_goals_against_per_match": float(goals_against / match_count),
                    "form_l10_goal_difference_per_match": float(goal_difference / match_count),
                    "form_l10_elo_change": float(elo_change) if pd.notna(elo_change) else np.nan,
                    "form_l10_elo_change_per_match": _safe_mean(recent["team_elo_delta"]),
                    "weighted_form_l10_result_score": _weighted_mean(recent["actual_score"], weights),
                    "weighted_form_l10_win_pct": _weighted_mean(recent["win_indicator"], weights),
                    "weighted_form_l10_goals_for_per_match": _weighted_mean(recent["team_score"], weights),
                    "weighted_form_l10_goals_against_per_match": _weighted_mean(recent["opponent_score"], weights),
                    "weighted_form_l10_goal_difference_per_match": _weighted_mean(recent["goal_difference"], weights),
                    "weighted_form_l10_elo_change_per_match": _weighted_mean(recent["team_elo_delta"], weights),
                }
            )
        form_rows.append(form_row)

    outcome = outcome.merge(pd.DataFrame(form_rows), on=["edition", "country"], how="left")
    outcome["edition_start_date"] = outcome["edition"].map(edition_start_dates)
    return outcome


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
        "finish_elo",
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
    for row in history.itertuples(index=False):
        edition = int(row.edition)
        country = str(row.country)
        prior_editions = [prior for prior in all_editions if prior < edition][-lookback:]
        prior_rows = [team_edition_lookup.get((prior, country)) for prior in prior_editions]
        appeared_rows = [prior_row for prior_row in prior_rows if prior_row is not None]
        appeared_weights = pd.Series(np.arange(1, len(appeared_rows) + 1, dtype=float))
        appearance_flags = [1.0 if prior_row is not None else 0.0 for prior_row in prior_rows]
        finish_scores = [prior_row.finish_score for prior_row in appeared_rows]
        positions = [prior_row.position for prior_row in appeared_rows]
        goals_for = [prior_row.gs for prior_row in appeared_rows]
        goals_against = [prior_row.ga for prior_row in appeared_rows]
        goal_differences = [prior_row.goal_difference for prior_row in appeared_rows]
        goals_per_match = [prior_row.goals_per_match for prior_row in appeared_rows]
        goals_against_per_match = [prior_row.goals_against_per_match for prior_row in appeared_rows]
        goal_difference_per_match = [prior_row.goal_difference_per_match for prior_row in appeared_rows]
        elo_changes = [prior_row.elo_change for prior_row in appeared_rows]
        rows.append(
            {
                "edition": edition,
                "country": country,
                "placement": row.placement,
                "lookback": lookback,
                "prior_editions": len(prior_editions),
                "prior_appearances": len(appeared_rows),
                "prior_appearance_rate": round(len(appeared_rows) / len(prior_editions), 3)
                if prior_editions
                else np.nan,
                "last_k_appearances": len(appeared_rows),
                "last_k_appearance_rate": round(len(appeared_rows) / len(prior_editions), 3)
                if prior_editions
                else np.nan,
                "last_k_avg_finish_score": _safe_mean(finish_scores),
                "last_k_best_finish_score": float(pd.to_numeric(pd.Series(finish_scores), errors="coerce").max())
                if finish_scores
                else np.nan,
                "last_k_avg_position": _safe_mean(positions),
                "last_k_goals_for": _safe_sum(goals_for),
                "last_k_goals_against": _safe_sum(goals_against),
                "last_k_goal_difference": _safe_sum(goal_differences),
                "last_k_goals_per_match": _safe_mean(goals_per_match),
                "last_k_goals_against_per_match": _safe_mean(goals_against_per_match),
                "last_k_goal_difference_per_match": _safe_mean(goal_difference_per_match),
                "last_k_elo_change": _safe_sum(elo_changes),
                "last_k_elo_change_per_appearance": _safe_mean(elo_changes),
                "weighted_last_k_appearance_rate": _weighted_mean(
                    appearance_flags,
                    pd.Series(np.arange(1, len(prior_editions) + 1, dtype=float)),
                )
                if prior_editions
                else np.nan,
                "weighted_last_k_finish_score": _weighted_mean(finish_scores, appeared_weights)
                if finish_scores
                else np.nan,
                "weighted_last_k_position": _weighted_mean(positions, appeared_weights) if positions else np.nan,
                "weighted_last_k_goals_per_match": _weighted_mean(goals_per_match, appeared_weights)
                if goals_per_match
                else np.nan,
                "weighted_last_k_goals_against_per_match": _weighted_mean(goals_against_per_match, appeared_weights)
                if goals_against_per_match
                else np.nan,
                "weighted_last_k_goal_difference_per_match": _weighted_mean(goal_difference_per_match, appeared_weights)
                if goal_difference_per_match
                else np.nan,
                "weighted_last_k_elo_change_per_appearance": _weighted_mean(elo_changes, appeared_weights)
                if elo_changes
                else np.nan,
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
