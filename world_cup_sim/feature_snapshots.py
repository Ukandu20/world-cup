from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


_PREPARED_LEAD_IN_CACHE: dict[tuple[int, int, tuple[str, ...]], dict[str, object]] = {}
_WC_HISTORY_CACHE: dict[tuple[int, int, int, int, int, int, tuple[str, ...], tuple[str, ...]], pd.DataFrame] = {}


TOURNAMENT_WEIGHT_MAP = {
    "FIFA World Cup": 3.0,
    "UEFA Euro": 2.5,
    "Copa America": 2.5,
    "Copa América": 2.5,
    "Africa Cup of Nations": 2.5,
    "AFC Asian Cup": 2.5,
    "CONCACAF Gold Cup": 2.5,
    "OFC Nations Cup": 2.5,
    "FIFA World Cup qualification": 2.0,
    "UEFA Euro qualification": 2.0,
}
FRIENDLY_WEIGHT = 1.0
COMPETITIVE_WEIGHT = 1.5


def _require_cutoff(cutoff_date: date) -> None:
    if cutoff_date is None:
        raise AssertionError("cutoff_date must be explicit; never use a default in validation paths")


def get_competition_weight(tournament_name: str) -> float:
    if pd.isna(tournament_name):
        return FRIENDLY_WEIGHT
    value = str(tournament_name)
    for key, weight in TOURNAMENT_WEIGHT_MAP.items():
        if key.lower() in value.lower():
            return weight
    if "friendly" in value.lower():
        return FRIENDLY_WEIGHT
    return COMPETITIVE_WEIGHT


def get_elo_expected_points(team_elo: float, opp_elo: float) -> float:
    win_expectation = 1.0 / (1.0 + 10.0 ** ((float(opp_elo) - float(team_elo)) / 400.0))
    return 3.0 * win_expectation


def _prepared_lead_in(lead_in: pd.DataFrame) -> dict[str, object]:
    cache_key = (id(lead_in), len(lead_in), tuple(map(str, lead_in.columns)))
    cached = _PREPARED_LEAD_IN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    prepared = lead_in.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for column in ["team_score", "opponent_score", "goal_difference", "team_elo_start", "opponent_elo_start"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    points_map = {"w": 3.0, "win": 3.0, "d": 1.0, "draw": 1.0, "l": 0.0, "loss": 0.0}
    prepared["_points"] = prepared["result"].astype(str).str.lower().map(points_map).fillna(0.0).astype(float)
    prepared = prepared.dropna(subset=["date", "qualified_team_id"]).sort_values(["qualified_team_id", "date"], kind="stable")

    team_rows: dict[str, pd.DataFrame] = {}
    team_dates: dict[str, np.ndarray] = {}
    for team_id, group in prepared.groupby(prepared["qualified_team_id"].astype(str), sort=False):
        group = group.reset_index(drop=True)
        team_rows[str(team_id)] = group
        team_dates[str(team_id)] = group["date"].to_numpy(dtype="datetime64[ns]")

    all_elo = prepared.dropna(subset=["team_elo_start"]).sort_values("date", kind="stable")
    elo_values = all_elo["team_elo_start"].to_numpy(dtype=float)
    payload = {
        "team_rows": team_rows,
        "team_dates": team_dates,
        "all_elo_dates": all_elo["date"].to_numpy(dtype="datetime64[ns]"),
        "all_elo_cumsum": np.cumsum(elo_values, dtype=float),
        "all_elo_count": np.arange(1, len(elo_values) + 1, dtype=float),
    }
    _PREPARED_LEAD_IN_CACHE[cache_key] = payload
    return payload


def _global_elo_mean_before(prepared: dict[str, object], cutoff_date: date) -> float:
    dates = prepared["all_elo_dates"]
    if len(dates) == 0:
        return 0.0
    position = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cutoff_date)), side="left"))
    if position <= 0:
        return 0.0
    cumsum = prepared["all_elo_cumsum"]
    counts = prepared["all_elo_count"]
    return float(cumsum[position - 1] / counts[position - 1])


def get_elo_at_cutoff(team_ids: list[str], cutoff_date: date, lead_in: pd.DataFrame) -> pd.Series:
    _require_cutoff(cutoff_date)
    prepared = _prepared_lead_in(lead_in)
    team_rows = prepared["team_rows"]
    team_dates = prepared["team_dates"]
    global_mean = _global_elo_mean_before(prepared, cutoff_date)
    values: list[float] = []
    cutoff_ts = np.datetime64(pd.Timestamp(cutoff_date))
    for team_id in team_ids:
        key = str(team_id)
        dates = team_dates.get(key)
        rows = team_rows.get(key)
        if dates is None or rows is None:
            values.append(global_mean)
            continue
        position = int(np.searchsorted(dates, cutoff_ts, side="left"))
        prior_elo = pd.to_numeric(rows.iloc[:position]["team_elo_start"], errors="coerce").dropna()
        values.append(float(prior_elo.iloc[-1]) if not prior_elo.empty else global_mean)
    return pd.Series(values, index=pd.Index(team_ids, name="qualified_team_id"))


def get_lead_in_form(
    team_ids: list[str],
    cutoff_date: date,
    lead_in: pd.DataFrame,
    k: int = 10,
    quadratic_weights: bool = True,
    time_decay_halflife: int | None = 1095,
) -> pd.DataFrame:
    _require_cutoff(cutoff_date)
    records: list[dict[str, object]] = []
    prepared = _prepared_lead_in(lead_in)
    team_rows = prepared["team_rows"]
    team_dates = prepared["team_dates"]
    global_elo = _global_elo_mean_before(prepared, cutoff_date)
    cutoff_ts = np.datetime64(pd.Timestamp(cutoff_date))

    for team_id in team_ids:
        rows = team_rows.get(str(team_id))
        dates = team_dates.get(str(team_id))
        if rows is None or dates is None:
            team_rows_at_cutoff = pd.DataFrame()
        else:
            position = int(np.searchsorted(dates, cutoff_ts, side="left"))
            team_rows_at_cutoff = rows.iloc[:position].tail(k).reset_index(drop=True)
        n_rows = len(team_rows_at_cutoff)
        if n_rows == 0:
            records.append(
                {
                    "team_id": team_id,
                    "results_form": 0.0,
                    "gd_form": 0.0,
                    "goals_for": 0.0,
                    "goals_against": 0.0,
                    "perf_vs_exp": 0.0,
                    "pre_tournament_elo": global_elo,
                    "lead_in_match_count": 0,
                }
            )
            continue

        indices = np.arange(1, n_rows + 1, dtype=float)
        weights = indices**2 if quadratic_weights else indices
        if time_decay_halflife is not None:
            days_before = np.array([(cutoff_date - pd.Timestamp(value).date()).days for value in team_rows_at_cutoff["date"]])
            weights = weights * (0.5 ** (days_before / float(time_decay_halflife)))
        weights = weights / weights.sum()

        points = team_rows_at_cutoff["_points"].fillna(0.0).to_numpy(dtype=float)
        expected_points = np.array(
            [
                get_elo_expected_points(row.team_elo_start, row.opponent_elo_start)
                for row in team_rows_at_cutoff.itertuples(index=False)
            ],
            dtype=float,
        )
        records.append(
            {
                "team_id": team_id,
                "results_form": float(np.dot(weights, points)),
                "gd_form": float(np.dot(weights, team_rows_at_cutoff["goal_difference"].fillna(0.0).to_numpy(dtype=float))),
                "goals_for": float(np.dot(weights, team_rows_at_cutoff["team_score"].fillna(0.0).to_numpy(dtype=float))),
                "goals_against": float(np.dot(weights, team_rows_at_cutoff["opponent_score"].fillna(0.0).to_numpy(dtype=float))),
                "perf_vs_exp": float(np.dot(weights, points - expected_points)),
                "pre_tournament_elo": float(team_rows_at_cutoff.iloc[-1]["team_elo_start"]),
                "lead_in_match_count": n_rows,
            }
        )

    return pd.DataFrame(records).set_index("team_id")


def get_wc_history_features(
    team_ids: list[str],
    holdout_year: int,
    teams: pd.DataFrame,
    results: pd.DataFrame,
    last_n_editions: int = 5,
) -> pd.DataFrame:
    cache_key = (
        id(teams),
        id(results),
        len(teams),
        len(results),
        int(holdout_year),
        int(last_n_editions),
        tuple(map(str, teams.columns)),
        tuple(map(str, results.columns)),
    )
    cached = _WC_HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached.reindex(pd.Index(team_ids, name="team_id")).copy()

    prior_teams = teams[pd.to_numeric(teams["year"], errors="coerce") < int(holdout_year)].copy()
    prior_results = results[pd.to_numeric(results["edition"], errors="coerce") < int(holdout_year)].copy()

    appearances = prior_teams.groupby("team_id").size().rename("appearance_count")
    best_pos = prior_teams.groupby("team_id")["position"].min().rename("best_position")
    recent_pos = prior_teams.sort_values("year").groupby("team_id")["position"].last().rename("last_placement")

    prior_results["team_score"] = pd.to_numeric(prior_results["team_score"], errors="coerce")
    prior_results["opponent_score"] = pd.to_numeric(prior_results["opponent_score"], errors="coerce")
    prior_results["gd"] = prior_results["team_score"] - prior_results["opponent_score"]
    last_years = sorted(pd.to_numeric(prior_results["edition"], errors="coerce").dropna().astype(int).unique())[-last_n_editions:]
    l5 = prior_results[pd.to_numeric(prior_results["edition"], errors="coerce").isin(last_years)]
    wc_l5_gd = l5.groupby("team_id")["gd"].sum().rename("wc_l5_goal_diff")
    has_l5 = l5.groupby("team_id").size().ge(1).astype(int).rename("has_wc_l5_history")

    all_team_ids = sorted(
        set(teams.get("team_id", pd.Series(dtype=object)).dropna().astype(str))
        | set(results.get("team_id", pd.Series(dtype=object)).dropna().astype(str))
    )
    out = pd.DataFrame(index=pd.Index(all_team_ids, name="team_id"))
    for column, series in [
        ("appearance_count", appearances),
        ("best_position", best_pos),
        ("last_placement", recent_pos),
        ("wc_l5_goal_diff", wc_l5_gd),
        ("has_wc_l5_history", has_l5),
    ]:
        out[column] = series.reindex(out.index)

    default_pos = float(pd.to_numeric(prior_teams["position"], errors="coerce").max() + 1) if not prior_teams.empty else 99.0
    out["appearance_count"] = out["appearance_count"].fillna(0).astype(int)
    out["has_wc_l5_history"] = out["has_wc_l5_history"].fillna(0).astype(int)
    out["best_position"] = out["best_position"].fillna(default_pos)
    out["last_placement"] = out["last_placement"].fillna(default_pos)
    _WC_HISTORY_CACHE[cache_key] = out.copy()
    return out.reindex(pd.Index(team_ids, name="team_id")).copy()


def get_team_features_at_date(
    team_ids: list[str],
    holdout_year: int,
    cutoff_date: date,
    data: dict[str, pd.DataFrame],
    k: int = 10,
    quadratic_weights: bool = True,
    time_decay_halflife: int | None = 1095,
) -> pd.DataFrame:
    _require_cutoff(cutoff_date)
    elo = get_elo_at_cutoff(team_ids, cutoff_date, data["lead_in"])
    form = get_lead_in_form(
        team_ids,
        cutoff_date,
        data["lead_in"],
        k=k,
        quadratic_weights=quadratic_weights,
        time_decay_halflife=time_decay_halflife,
    )
    wc_hist = get_wc_history_features(team_ids, holdout_year, data["teams"], data["results"])

    features = form.copy()
    features["start_elo"] = elo
    features = features.join(wc_hist, how="left")

    if features["wc_l5_goal_diff"].isna().any():
        try:
            bins = pd.qcut(features["start_elo"], q=5, labels=False, duplicates="drop")
            features["wc_l5_goal_diff"] = (
                features.groupby(bins)["wc_l5_goal_diff"].transform(lambda values: values.fillna(values.mean())).fillna(0.0)
            )
        except ValueError:
            features["wc_l5_goal_diff"] = features["wc_l5_goal_diff"].fillna(0.0)

    if features.index.duplicated().any():
        raise ValueError("Duplicate team_ids in feature output")
    if features.shape[0] != len(team_ids):
        raise ValueError(f"Expected {len(team_ids)} rows, got {features.shape[0]}")
    return features
