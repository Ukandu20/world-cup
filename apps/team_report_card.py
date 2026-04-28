from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from apps import home
from world_cup_sim.constants import (
    WEIGHTED_FORM_ELO_BOUNDS,
    WEIGHTED_FORM_GD_BOUNDS,
    WEIGHTED_FORM_PERF_BOUNDS,
)
from world_cup_sim.shared import (
    clip_scale,
    compute_elo_expected_score,
    normalize_historical_team_name,
    normalize_weighted_form_result,
)
import world_cup_simulation as simulation


ROOT = Path(__file__).resolve().parents[1]
WORLD_CUP_ROOT = simulation.WORLD_CUP_ROOT
SUBJECT_ORDER = (
    "Overall Strength",
    "Attack",
    "Defense",
    "Recent Form",
    "World Cup History",
    "Tournament Outlook",
)
PENDING_SUBJECTS = ("Squad Quality", "Qualification Strength")
PENDING_IDENTITY_FIELDS = ("Coach", "Captain")
GRADE_BANDS = (
    (9.5, "A+"),
    (8.8, "A"),
    (7.5, "B"),
    (6.0, "C"),
    (4.5, "D"),
    (-float("inf"), "F"),
)
SUBJECT_NOTE_TEMPLATES = {
    "Overall Strength": (
        "Outstanding all-round outlook compared with the field.",
        "Strong overall profile with only minor soft spots.",
        "Solid baseline, but not yet among the elite favorites.",
        "Mixed profile with clear volatility in the model.",
        "Undersized overall profile for a deep run.",
    ),
    "Attack": (
        "Creates one of the sharpest attacking profiles in the tournament.",
        "Carries reliable scoring upside against most opponents.",
        "Can score, but not consistently enough to dominate.",
        "Needs favorable game states to create clear attacking value.",
        "Attack is a major limiter in the current model.",
    ),
    "Defense": (
        "Defensive profile is among the most reliable in the field.",
        "Usually suppresses chances well enough to stay in control.",
        "Functional defense, though not especially dominant.",
        "Can be exposed when the match tempo rises.",
        "Defensive resilience is a major concern.",
    ),
    "Recent Form": (
        "Lead-in form is excellent and supports the projection.",
        "Recent results meaningfully strengthen the baseline.",
        "Recent form is acceptable but not carrying the team.",
        "Lead-in form is uneven and adds some downside.",
        "Recent form is working against the projection.",
    ),
    "World Cup History": (
        "Tournament pedigree is elite by any historical standard.",
        "History adds meaningful credibility to the forecast.",
        "History is respectable without being a major edge.",
        "Limited recent pedigree reduces the margin for error.",
        "World Cup history offers little support here.",
    ),
    "Tournament Outlook": (
        "Model gives this team a genuine title-level outlook.",
        "Knockout advancement chances are consistently strong.",
        "Projection points to a plausible but imperfect run.",
        "Needs several outcomes to break correctly for a deep run.",
        "Current tournament path is difficult in the model.",
    ),
}
DRIVER_LABELS = {
    "elo_rating": "Elite Elo base",
    "results_form": "Strong recent results",
    "gd_form": "Healthy recent goal difference",
    "placement_metric": "Strong World Cup history",
    "goals_for": "Reliable attacking output",
    "host_flag": "Host advantage",
}


def score_to_grade(score: float) -> str:
    """Map a 1-10 score to the fixed report-card grade bands."""
    numeric = float(score)
    for minimum, grade in GRADE_BANDS:
        if numeric >= minimum:
            return grade
    return "F"


def score_to_verdict(score: float) -> str:
    """Translate a report-card score into one short outlook verdict."""
    numeric = float(score)
    if numeric >= 9.0:
        return "Contender"
    if numeric >= 8.0:
        return "Strong knockout candidate"
    if numeric >= 7.0:
        return "Dangerous outsider"
    if numeric >= 6.0:
        return "Competitive but vulnerable"
    return "Likely group-stage struggler"


def choose_column(df: pd.DataFrame, candidates: Iterable[str], fallback: str = "") -> str:
    """Return the first available column name from a list of candidates."""
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    return fallback


def build_display_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Return a simple team-id to display-name lookup."""
    return (
        df.loc[:, ["team_id", "display_name"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .assign(team_id=lambda frame: frame["team_id"].astype(str), display_name=lambda frame: frame["display_name"].astype(str))
        .set_index("team_id")["display_name"]
        .to_dict()
    )


def build_flag_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Return a simple team-id to flag-icon-code lookup."""
    return (
        df.loc[:, ["team_id", "flag_icon_code"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .assign(team_id=lambda frame: frame["team_id"].astype(str), flag_icon_code=lambda frame: frame["flag_icon_code"].fillna("").astype(str))
        .set_index("team_id")["flag_icon_code"]
        .to_dict()
    )


def series_to_report_scores(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Scale a metric into deterministic 1-10 report-card scores via rank percentiles."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.full(len(series), 1.0), index=series.index, dtype=float)
    if len(valid) == 1:
        scores = pd.Series(np.full(len(series), 10.0), index=series.index, dtype=float)
        scores.loc[numeric.isna()] = 1.0
        return scores

    ranking_source = -valid if reverse else valid
    rank = ranking_source.rank(method="average", ascending=True)
    scaled = 1.0 + 9.0 * ((rank - 1.0) / (len(valid) - 1.0))
    scores = pd.Series(1.0, index=series.index, dtype=float)
    scores.loc[valid.index] = scaled.astype(float)
    return scores.clip(lower=1.0, upper=10.0).round(1)


def describe_subject_score(subject: str, score: float) -> str:
    """Return a short teacher-style note for one subject score."""
    elite, strong, solid, mixed, weak = SUBJECT_NOTE_TEMPLATES[subject]
    if score >= 9.0:
        return elite
    if score >= 8.0:
        return strong
    if score >= 6.5:
        return solid
    if score >= 5.0:
        return mixed
    return weak


def add_report_card_metrics(dashboard_df: pd.DataFrame) -> pd.DataFrame:
    """Add the raw and scored report-card metrics to the V3 simulation table."""
    df = dashboard_df.copy()
    appearance_max = max(float(pd.to_numeric(df["appearance"], errors="coerce").fillna(0.0).max()), 1.0)
    df["appearance_norm"] = pd.to_numeric(df["appearance"], errors="coerce").fillna(0.0) / appearance_max
    df["recent_form_metric"] = (
        0.5 * pd.to_numeric(df["results_form"], errors="coerce").fillna(0.0)
        + 0.3 * pd.to_numeric(df["gd_form"], errors="coerce").fillna(0.0)
        + 0.2 * pd.to_numeric(df["perf_vs_exp"], errors="coerce").fillna(0.0)
    )
    df["history_metric"] = (
        0.7 * pd.to_numeric(df["placement"], errors="coerce").fillna(0.0)
        + 0.3 * pd.to_numeric(df["appearance_norm"], errors="coerce").fillna(0.0)
    )
    df["outlook_metric"] = (
        0.10 * pd.to_numeric(df["ko_prob"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(df["r16_prob"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["qf_prob"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["sf_prob"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["final_prob"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(df["champion_prob"], errors="coerce").fillna(0.0)
    )

    raw_subject_metrics = {
        "Overall Strength": pd.to_numeric(df["team_strength"], errors="coerce").fillna(0.0),
        "Attack": pd.to_numeric(df["goals_for"], errors="coerce").fillna(0.0),
        "Defense": pd.to_numeric(df["goals_against"], errors="coerce").fillna(0.0),
        "Recent Form": pd.to_numeric(df["recent_form_metric"], errors="coerce").fillna(0.0),
        "World Cup History": pd.to_numeric(df["history_metric"], errors="coerce").fillna(0.0),
        "Tournament Outlook": pd.to_numeric(df["outlook_metric"], errors="coerce").fillna(0.0),
    }
    reverse_subjects = {"Defense"}
    for subject, metric_series in raw_subject_metrics.items():
        score_column = subject.lower().replace(" ", "_").replace("-", "_") + "_score"
        df[score_column] = series_to_report_scores(metric_series, reverse=subject in reverse_subjects)

    overall_columns = [subject.lower().replace(" ", "_").replace("-", "_") + "_score" for subject in SUBJECT_ORDER]
    df["overall_report_score"] = df.loc[:, overall_columns].mean(axis=1).round(1)
    df["overall_grade"] = df["overall_report_score"].map(score_to_grade)
    df["overall_verdict"] = df["overall_report_score"].map(score_to_verdict)
    return df


def build_subject_rows(team_row: pd.Series) -> list[dict[str, str | float]]:
    """Build the subject-score rows shown in the report card."""
    rows: list[dict[str, str | float]] = []
    for subject in SUBJECT_ORDER:
        score_column = subject.lower().replace(" ", "_").replace("-", "_") + "_score"
        score = float(team_row[score_column])
        rows.append(
            {
                "subject": subject,
                "score": round(score, 1),
                "grade": score_to_grade(score),
                "note": describe_subject_score(subject, score),
            }
        )
    return rows


def build_pending_subject_rows() -> list[dict[str, str]]:
    """Return the unsupported subject rows for the MVP."""
    return [{"subject": subject, "value": "Pending data"} for subject in PENDING_SUBJECTS]


def build_identity_rows(team_row: pd.Series, best_finish: str) -> list[dict[str, str]]:
    """Return the key identity facts for the selected team."""
    appearances_value = team_row.get("world_cup_participations", "")
    if pd.isna(appearances_value) or appearances_value == "":
        appearances_value = ""
    else:
        appearances_value = f"{int(float(appearances_value))}"

    rows = [
        {"label": "Confederation", "value": str(team_row.get("confederation", ""))},
        {"label": "Group", "value": f"Group {team_row.get('group_code', '')}"},
        {"label": "FIFA Rank", "value": f"{int(float(team_row['world_rank']))}" if pd.notna(team_row.get("world_rank")) else "N/A"},
        {"label": "Elo Rating", "value": f"{int(round(float(team_row['elo_rating'])))}" if pd.notna(team_row.get("elo_rating")) else "N/A"},
        {"label": "World Cup Appearances", "value": str(appearances_value or "N/A")},
        {"label": "Best Finish", "value": best_finish},
    ]
    rows.extend({"label": field, "value": "Pending data"} for field in PENDING_IDENTITY_FIELDS)
    return rows


def normalize_team_best_finish(placement_df: pd.DataFrame, team_names: Iterable[str]) -> str:
    """Resolve the best historical finish for a current team across name aliases."""
    normalized_names = {
        normalize_historical_team_name(name)
        for name in team_names
        if isinstance(name, str) and name.strip()
    }
    if not normalized_names:
        return "No appearances"
    team_rows = placement_df[placement_df["team_key"].isin(normalized_names)].copy()
    if team_rows.empty:
        return "No appearances"
    best_row = team_rows.sort_values(["position", "edition"], ascending=[True, True], kind="stable").iloc[0]
    return str(best_row["placement"])


def build_best_finish_lookup(base_df: pd.DataFrame) -> dict[str, str]:
    """Build a team-id to best-finish lookup from historical placement data."""
    placement_df = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "placement.csv")
    placement_df["position"] = pd.to_numeric(placement_df["position"], errors="coerce")
    placement_df["team_key"] = placement_df["country"].map(normalize_historical_team_name)
    canonical_column = choose_column(base_df, ("canonical_name", "canonical_name_y", "canonical_name_x"), fallback="")

    lookup: dict[str, str] = {}
    for row in base_df.drop_duplicates(subset=["team_id"], keep="first").itertuples(index=False):
        team_names = [
            getattr(row, "display_name", ""),
            getattr(row, "tournament_name", ""),
            getattr(row, "team_name", ""),
            getattr(row, canonical_column, "") if canonical_column else "",
        ]
        lookup[str(getattr(row, "team_id"))] = normalize_team_best_finish(placement_df, team_names)
    return lookup


def build_recent_matches_table(lead_in_df: pd.DataFrame, team_id: str, match_window: int = 10) -> pd.DataFrame:
    """Return the selected team's latest matches with report-card performance grades."""
    team_matches = lead_in_df[lead_in_df["qualified_team_id"].astype(str) == str(team_id)].copy()
    if team_matches.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Opponent",
                "Competition",
                "Result",
                "Score",
                "Elo Change",
                "Performance Score",
                "Grade",
            ]
        )

    team_matches["date"] = pd.to_datetime(team_matches["date"], errors="coerce")
    for column_name in ("team_score", "opponent_score", "team_elo_start", "opponent_elo_start", "team_elo_delta"):
        team_matches[column_name] = pd.to_numeric(team_matches[column_name], errors="coerce")
    team_matches = team_matches.sort_values(["date", "lead_in_id"], kind="stable").tail(match_window).copy()
    team_matches["normalized_result"] = normalize_weighted_form_result(
        team_matches["result"],
        team_matches["team_score"],
        team_matches["opponent_score"],
    )
    team_matches["actual_score"] = team_matches["normalized_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0}).astype(float)
    team_matches["goal_difference"] = (team_matches["team_score"] - team_matches["opponent_score"]).fillna(0.0)
    team_matches["expected_score"] = compute_elo_expected_score(
        team_matches["team_elo_start"],
        team_matches["opponent_elo_start"],
    ).astype(float)
    team_matches["perf_vs_exp"] = team_matches["actual_score"] - team_matches["expected_score"]
    team_matches["gd_score"] = clip_scale(team_matches["goal_difference"].clip(lower=-4.0, upper=4.0), *WEIGHTED_FORM_GD_BOUNDS)
    team_matches["perf_score"] = clip_scale(team_matches["perf_vs_exp"], *WEIGHTED_FORM_PERF_BOUNDS)
    team_matches["elo_score"] = clip_scale(team_matches["team_elo_delta"], *WEIGHTED_FORM_ELO_BOUNDS)
    team_matches["performance_index"] = (
        0.4 * team_matches["actual_score"]
        + 0.25 * team_matches["gd_score"]
        + 0.25 * team_matches["perf_score"]
        + 0.10 * team_matches["elo_score"]
    )
    team_matches["Performance Score"] = (1.0 + 9.0 * team_matches["performance_index"]).round(1)
    team_matches["Grade"] = team_matches["Performance Score"].map(score_to_grade)
    team_matches["Date"] = team_matches["date"].dt.strftime("%Y-%m-%d")
    team_matches["Opponent"] = team_matches["opponent_name"].fillna("").astype(str)
    team_matches["Competition"] = team_matches["tournament"].fillna("").astype(str)
    team_matches["Result"] = team_matches["normalized_result"].map({"win": "W", "draw": "D", "loss": "L"}).fillna("")
    team_matches["Score"] = (
        team_matches["team_score"].fillna(0).astype(int).astype(str)
        + "-"
        + team_matches["opponent_score"].fillna(0).astype(int).astype(str)
    )
    team_matches["Elo Change"] = team_matches["team_elo_delta"].map(lambda value: f"{float(value):+0.1f}")
    team_matches["post_match_elo"] = team_matches["team_elo_start"].fillna(0.0) + team_matches["team_elo_delta"].fillna(0.0)
    return team_matches.sort_values(["date", "lead_in_id"], ascending=[False, False], kind="stable").reset_index(drop=True)


def build_group_fixtures_table(fixtures_df: pd.DataFrame, team_id: str, display_lookup: dict[str, str]) -> pd.DataFrame:
    """Return the selected team's upcoming group-stage fixtures."""
    df = fixtures_df.copy()
    df["match_number"] = pd.to_numeric(df["match_number"], errors="coerce")
    df["kickoff_datetime_utc"] = pd.to_datetime(df["kickoff_datetime_utc"], errors="coerce", utc=True)
    team_fixtures = df[
        (df["round_code"] == "GS")
        & (
            df["home_team_id"].astype(str).eq(str(team_id))
            | df["away_team_id"].astype(str).eq(str(team_id))
        )
    ].copy()
    if team_fixtures.empty:
        return pd.DataFrame(columns=["Date", "Opponent", "Stage", "Venue"])

    team_fixtures["Date"] = team_fixtures["kickoff_datetime_utc"].dt.strftime("%Y-%m-%d")
    team_fixtures["Opponent"] = np.where(
        team_fixtures["home_team_id"].astype(str).eq(str(team_id)),
        team_fixtures["away_team_id"].map(display_lookup).fillna(team_fixtures["away_tournament_name"]),
        team_fixtures["home_team_id"].map(display_lookup).fillna(team_fixtures["home_tournament_name"]),
    )
    team_fixtures["Stage"] = "Group Stage"
    team_fixtures["Venue"] = team_fixtures["venue_name"].fillna("").astype(str)
    return team_fixtures.sort_values(["kickoff_datetime_utc", "match_number"], kind="stable").loc[:, ["Date", "Opponent", "Stage", "Venue"]].reset_index(drop=True)


def build_knockout_path_table(bracket_data: dict[str, Any], team_id: str, display_lookup: dict[str, str]) -> pd.DataFrame:
    """Return the selected team's projected knockout path from the deterministic bracket."""
    rows: list[dict[str, str | float]] = []
    for round_data in bracket_data.get("rounds", []):
        for match in round_data.get("matches", []):
            home_team_id = str(match.get("home_team_id", ""))
            away_team_id = str(match.get("away_team_id", ""))
            if str(team_id) not in {home_team_id, away_team_id}:
                continue
            is_home = str(team_id) == home_team_id
            opponent_id = away_team_id if is_home else home_team_id
            matchup_win_prob = float(match.get("home_win_prob" if is_home else "away_win_prob", 0.0))
            rows.append(
                {
                    "Stage": str(match.get("round_label", match.get("round_code", ""))),
                    "Opponent": display_lookup.get(opponent_id, opponent_id),
                    "Matchup Win %": round(matchup_win_prob, 1),
                    "Projected Winner": display_lookup.get(str(match.get("winner_team_id", "")), str(match.get("winner_team_id", ""))),
                }
            )
            break
    return pd.DataFrame(rows)


def build_probability_tables(team_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact probability tables for group finish and knockout stages."""
    group_finish = pd.DataFrame(
        [
            {"Finish": "1st", "Probability": float(team_row.get("prob_1", 0.0))},
            {"Finish": "2nd", "Probability": float(team_row.get("prob_2", 0.0))},
            {"Finish": "3rd", "Probability": float(team_row.get("prob_3", 0.0))},
            {"Finish": "4th", "Probability": float(team_row.get("prob_4", 0.0))},
        ]
    )
    stage_progression = pd.DataFrame(
        [
            {"Stage": "Round of 32", "Probability": float(team_row.get("ko_prob", 0.0))},
            {"Stage": "Round of 16", "Probability": float(team_row.get("r16_prob", 0.0))},
            {"Stage": "Quarter-finals", "Probability": float(team_row.get("qf_prob", 0.0))},
            {"Stage": "Semi-finals", "Probability": float(team_row.get("sf_prob", 0.0))},
            {"Stage": "Final", "Probability": float(team_row.get("final_prob", 0.0))},
            {"Stage": "Winner", "Probability": float(team_row.get("champion_prob", 0.0))},
        ]
    )
    return group_finish, stage_progression


def build_model_reason_bullets(team_row: pd.Series, full_df: pd.DataFrame) -> list[str]:
    """Return a short set of model-friendly reasons for the selected team."""
    metric_frame = pd.DataFrame(
        {
            "team_id": full_df["team_id"].astype(str),
            "elo_rating": pd.to_numeric(full_df["elo_rating"], errors="coerce").fillna(0.0),
            "results_form": pd.to_numeric(full_df["results_form"], errors="coerce").fillna(0.0),
            "gd_form": pd.to_numeric(full_df["gd_form"], errors="coerce").fillna(0.0),
            "placement_metric": pd.to_numeric(full_df["history_metric"], errors="coerce").fillna(0.0),
            "goals_for": pd.to_numeric(full_df["goals_for"], errors="coerce").fillna(0.0),
            "host_flag": pd.to_numeric(full_df["host_flag"], errors="coerce").fillna(0.0),
        }
    ).drop_duplicates(subset=["team_id"], keep="first").set_index("team_id")
    ranked_rows: list[tuple[float, str]] = []
    team_index = str(team_row.get("team_id", team_row.name))
    if team_index not in metric_frame.index:
        return []
    for column_name, label in DRIVER_LABELS.items():
        if column_name == "host_flag" and float(team_row.get("host_flag", 0.0)) <= 0:
            continue
        score = float(series_to_report_scores(metric_frame[column_name]).loc[team_index])
        ranked_rows.append((score, label))
    ranked_rows.sort(reverse=True)
    return [label for _, label in ranked_rows[:3]]


@st.cache_data(show_spinner=False)
def build_report_card_dataset(simulations: int, match_window: int) -> dict[str, Any]:
    """Build and cache the shared V3 report-card dataset for the active controls."""
    base_df, fixtures_df, lead_in_df, metadata = home.load_data()
    base_team_lookup = (
        base_df.drop_duplicates(subset=["team_id"], keep="first")
        .set_index("team_id")
        .to_dict("index")
    )
    model_bundle = home.load_v3_poisson_model(match_window)
    dashboard_df = home.simulate_probabilities_v3_dashboard(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=simulations,
        match_window=match_window,
    )
    dashboard_df = home.ensure_dashboard_probability_columns(dashboard_df)
    dashboard_df = add_report_card_metrics(dashboard_df)
    bracket_data = simulation.build_deterministic_bracket_v3(
        dashboard_df,
        fixtures_df,
        dashboard_df,
        model_bundle,
        head_to_head_simulations=home.BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    return {
        "base_df": base_df,
        "fixtures_df": fixtures_df,
        "lead_in_df": lead_in_df,
        "metadata": metadata,
        "dashboard_df": dashboard_df,
        "bracket_data": bracket_data,
        "display_lookup": build_display_lookup(base_df),
        "flag_lookup": build_flag_lookup(base_df),
        "best_finish_lookup": build_best_finish_lookup(base_df),
        "base_team_lookup": base_team_lookup,
    }


def select_report_card_context(dataset: dict[str, Any], team_id: str, recent_match_count: int = 10) -> dict[str, Any]:
    """Filter the shared dataset into one selected-team context payload."""
    dashboard_df = dataset["dashboard_df"]
    team_row = dashboard_df.loc[dashboard_df["team_id"].astype(str) == str(team_id)].copy()
    if team_row.empty:
        raise ValueError(f"Unknown team_id: {team_id}")
    team_row = team_row.iloc[0].copy()
    team_row.name = str(team_row["team_id"])
    base_snapshot = dataset["base_team_lookup"].get(str(team_id), {})
    for column_name, value in base_snapshot.items():
        if column_name not in team_row.index or pd.isna(team_row[column_name]):
            team_row[column_name] = value

    recent_matches = build_recent_matches_table(dataset["lead_in_df"], str(team_id), match_window=recent_match_count)
    group_fixtures = build_group_fixtures_table(dataset["fixtures_df"], str(team_id), dataset["display_lookup"])
    knockout_path = build_knockout_path_table(dataset["bracket_data"], str(team_id), dataset["display_lookup"])
    group_finish_table, stage_probability_table = build_probability_tables(team_row)
    subject_rows = build_subject_rows(team_row)
    overall_summary = {
        "score": float(team_row["overall_report_score"]),
        "grade": str(team_row["overall_grade"]),
        "verdict": str(team_row["overall_verdict"]),
    }
    strongest = max(subject_rows, key=lambda row: float(row["score"]))
    weakest = min(subject_rows, key=lambda row: float(row["score"]))
    overall_summary["summary"] = f"{strongest['subject']} leads this profile, while {weakest['subject']} is the main pressure point."
    return {
        "team_row": team_row,
        "identity_rows": build_identity_rows(team_row, dataset["best_finish_lookup"].get(str(team_id), "No appearances")),
        "subject_rows": subject_rows,
        "pending_subject_rows": build_pending_subject_rows(),
        "recent_matches": recent_matches,
        "group_fixtures": group_fixtures,
        "knockout_path": knockout_path,
        "first_knockout_match": knockout_path.iloc[0].to_dict() if not knockout_path.empty else None,
        "group_finish_table": group_finish_table,
        "stage_probability_table": stage_probability_table,
        "overall_summary": overall_summary,
        "model_reason_bullets": build_model_reason_bullets(team_row, dashboard_df),
        "display_lookup": dataset["display_lookup"],
        "flag_lookup": dataset["flag_lookup"],
        "metadata": dataset["metadata"],
        "simulation_count": len(dashboard_df),
    }


def report_card_css() -> str:
    """Return custom CSS used by the team report-card page."""
    return """
    .trc-shell {
        display: grid;
        gap: 18px;
        margin-top: 0.5rem;
    }
    .trc-hero {
        border: 1px solid #dbe4ef;
        border-radius: 24px;
        padding: 22px 24px;
        background:
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 28%),
            linear-gradient(135deg, #f8fbff 0%, #eef4fb 100%);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }
    .trc-hero-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        flex-wrap: wrap;
    }
    .trc-title {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .trc-title .fi {
        font-size: 1.8rem;
        border-radius: 999px;
        box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.12);
    }
    .trc-title h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
    }
    .trc-subhead {
        margin-top: 0.35rem;
        color: #475569;
        font-weight: 600;
    }
    .trc-grade-panel {
        min-width: 220px;
        border-radius: 20px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: #f8fafc;
        text-align: center;
    }
    .trc-grade-kicker {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: #bfdbfe;
        margin-bottom: 0.45rem;
    }
    .trc-grade {
        font-size: 2.9rem;
        font-weight: 900;
        line-height: 1;
    }
    .trc-score {
        margin-top: 0.35rem;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .trc-verdict {
        margin-top: 0.55rem;
        font-size: 0.95rem;
        color: #dbeafe;
    }
    .trc-facts {
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
    }
    .trc-fact {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 14px 15px;
        background: rgba(255, 255, 255, 0.84);
    }
    .trc-fact-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.7rem;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .trc-fact-value {
        font-weight: 700;
        color: #0f172a;
        line-height: 1.3;
    }
    .trc-subject-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
    }
    .trc-subject-card, .trc-pending-card {
        border: 1px solid #dbe4ef;
        border-radius: 20px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        padding: 16px 17px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }
    .trc-subject-card h3, .trc-pending-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #0f172a;
    }
    .trc-grade-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 2.3rem;
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        background: #0f172a;
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 800;
    }
    .trc-subject-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
    .trc-subject-score {
        margin-top: 0.7rem;
        font-size: 2rem;
        line-height: 1;
        font-weight: 900;
        color: #0f172a;
    }
    .trc-subject-note {
        margin-top: 0.7rem;
        color: #475569;
        line-height: 1.45;
        font-size: 0.92rem;
    }
    .trc-pending-value {
        margin-top: 0.75rem;
        color: #64748b;
        font-weight: 700;
    }
    """


def get_query_team_param() -> str | None:
    """Read the active team query parameter from the Streamlit page."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        value = query_params.get("team")
        if isinstance(value, list):
            return str(value[0]) if value else None
        return str(value) if value else None
    getter = getattr(st, "experimental_get_query_params", None)
    if getter is None:
        return None
    values = getter().get("team", [])
    return str(values[0]) if values else None


def set_query_team_param(team_id: str) -> None:
    """Set the active team query parameter on the Streamlit page."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params["team"] = str(team_id)
        return
    setter = getattr(st, "experimental_set_query_params", None)
    if setter is not None:
        setter(team=str(team_id))


def format_percent(value: float) -> str:
    """Format one probability value for display."""
    return f"{float(value):.1f}%"


def render_identity_header(context: dict[str, Any]) -> None:
    """Render the top hero block for the selected team."""
    team_row = context["team_row"]
    overall = context["overall_summary"]
    flag_code = str(team_row.get("flag_icon_code", "") or "")
    flag_html = f'<span class="fi fi-{flag_code}"></span>' if flag_code else ""
    st.markdown(
        f"""
        <div class="trc-hero">
          <div class="trc-hero-top">
            <div>
              <div class="trc-title">
                {flag_html}
                <div>
                  <h1>{team_row['display_name']}</h1>
                  <div class="trc-subhead">Group {team_row['group_code']} · {team_row['confederation']}</div>
                </div>
              </div>
            </div>
            <div class="trc-grade-panel">
              <div class="trc-grade-kicker">Overall Report Card</div>
              <div class="trc-grade">{overall['grade']}</div>
              <div class="trc-score">{overall['score']:.1f} / 10</div>
              <div class="trc-verdict">{overall['verdict']}</div>
            </div>
          </div>
          <div class="trc-facts">
            {"".join(
                f'<div class="trc-fact"><div class="trc-fact-label">{row["label"]}</div><div class="trc-fact-value">{row["value"]}</div></div>'
                for row in context["identity_rows"]
            )}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(overall["summary"])


def render_subject_cards(context: dict[str, Any]) -> None:
    """Render the scored and pending subject cards."""
    scored_html = "".join(
        (
            f'<div class="trc-subject-card">'
            f'<div class="trc-subject-head"><h3>{row["subject"]}</h3><span class="trc-grade-chip">{row["grade"]}</span></div>'
            f'<div class="trc-subject-score">{row["score"]:.1f} / 10</div>'
            f'<div class="trc-subject-note">{row["note"]}</div>'
            f"</div>"
        )
        for row in context["subject_rows"]
    )
    pending_html = "".join(
        (
            f'<div class="trc-pending-card">'
            f"<h3>{row['subject']}</h3>"
            f'<div class="trc-pending-value">{row["value"]}</div>'
            f"</div>"
        )
        for row in context["pending_subject_rows"]
    )
    st.markdown(f'<div class="trc-subject-grid">{scored_html}{pending_html}</div>', unsafe_allow_html=True)


def build_plotly_figure_library() -> tuple[Any, Any]:
    """Import Plotly lazily so tests do not require it at module import time."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


def render_charts(context: dict[str, Any]) -> None:
    """Render the report-card charts."""
    recent_matches = context["recent_matches"].copy()
    if recent_matches.empty:
        st.info("No recent match history is available for this team.")
        return

    go, _ = build_plotly_figure_library()
    chart_df = recent_matches.sort_values(["date", "lead_in_id"], kind="stable").copy()
    labels = chart_df["Date"] + " vs " + chart_df["Opponent"]

    elo_fig = go.Figure()
    elo_fig.add_trace(go.Scatter(x=labels, y=chart_df["post_match_elo"], mode="lines+markers", name="Post-match Elo"))
    elo_fig.update_layout(title="Recent Elo Trend", margin=dict(l=20, r=20, t=44, b=20), height=320)

    perf_fig = go.Figure()
    perf_fig.add_trace(go.Scatter(x=labels, y=chart_df["actual_score"], mode="lines+markers", name="Actual Score"))
    perf_fig.add_trace(go.Scatter(x=labels, y=chart_df["expected_score"], mode="lines+markers", name="Expected Score"))
    perf_fig.update_layout(title="Expected vs Actual Performance", margin=dict(l=20, r=20, t=44, b=20), height=320)

    goal_fig = go.Figure()
    goal_fig.add_trace(go.Bar(x=labels, y=chart_df["team_score"], name="Goals For"))
    goal_fig.add_trace(go.Bar(x=labels, y=chart_df["opponent_score"], name="Goals Against"))
    goal_fig.update_layout(title="Goals For vs Goals Against", barmode="group", margin=dict(l=20, r=20, t=44, b=20), height=320)

    breakdown_fig = go.Figure(
        go.Pie(
            labels=["Wins", "Draws", "Losses"],
            values=[
                int(chart_df["normalized_result"].eq("win").sum()),
                int(chart_df["normalized_result"].eq("draw").sum()),
                int(chart_df["normalized_result"].eq("loss").sum()),
            ],
            hole=0.55,
        )
    )
    breakdown_fig.update_layout(title="Win / Draw / Loss Breakdown", margin=dict(l=20, r=20, t=44, b=20), height=320)

    stage_prob = context["stage_probability_table"].copy()
    prob_fig = go.Figure(go.Bar(x=stage_prob["Probability"], y=stage_prob["Stage"], orientation="h"))
    prob_fig.update_layout(title="Tournament Probability Breakdown", margin=dict(l=20, r=20, t=44, b=20), height=320)

    radar_rows = context["subject_rows"]
    radar_labels = [row["subject"] for row in radar_rows]
    radar_scores = [float(row["score"]) for row in radar_rows]
    radar_labels.append(radar_labels[0])
    radar_scores.append(radar_scores[0])
    radar_fig = go.Figure(
        go.Scatterpolar(
            r=radar_scores,
            theta=radar_labels,
            fill="toself",
            name=context["team_row"]["display_name"],
        )
    )
    radar_fig.update_layout(
        title="Team Profile Radar",
        polar=dict(radialaxis=dict(visible=True, range=[1, 10])),
        margin=dict(l=20, r=20, t=44, b=20),
        height=320,
        showlegend=False,
    )

    top_cols = st.columns(2)
    with top_cols[0]:
        st.plotly_chart(elo_fig, use_container_width=True)
    with top_cols[1]:
        st.plotly_chart(perf_fig, use_container_width=True)

    middle_cols = st.columns(2)
    with middle_cols[0]:
        st.plotly_chart(goal_fig, use_container_width=True)
    with middle_cols[1]:
        st.plotly_chart(breakdown_fig, use_container_width=True)

    bottom_cols = st.columns(2)
    with bottom_cols[0]:
        st.plotly_chart(prob_fig, use_container_width=True)
    with bottom_cols[1]:
        st.plotly_chart(radar_fig, use_container_width=True)


def render_recent_performance(context: dict[str, Any]) -> None:
    """Render the recent performance section."""
    recent_table = context["recent_matches"].loc[
        :,
        ["Date", "Opponent", "Competition", "Result", "Score", "Elo Change", "Performance Score", "Grade"],
    ]
    st.subheader("Recent Performance")
    st.dataframe(recent_table, use_container_width=True, hide_index=True)


def render_prediction_outlook(context: dict[str, Any]) -> None:
    """Render the probability tables and model explanation."""
    st.subheader("Prediction Outlook")
    cols = st.columns(2)
    with cols[0]:
        group_table = context["group_finish_table"].copy()
        group_table["Probability"] = group_table["Probability"].map(format_percent)
        st.caption("Group Finish Probabilities")
        st.dataframe(group_table, use_container_width=True, hide_index=True)
    with cols[1]:
        stage_table = context["stage_probability_table"].copy()
        stage_table["Probability"] = stage_table["Probability"].map(format_percent)
        st.caption("Tournament Stage Probabilities")
        st.dataframe(stage_table, use_container_width=True, hide_index=True)

    st.caption("Why the model likes this team")
    for bullet in context["model_reason_bullets"]:
        st.write(f"- {bullet}")


def render_fixtures_and_path(context: dict[str, Any]) -> None:
    """Render upcoming fixtures and projected knockout path."""
    st.subheader("Fixtures And Path")
    cols = st.columns(2)
    with cols[0]:
        st.caption("Group Stage Fixtures")
        st.dataframe(context["group_fixtures"], use_container_width=True, hide_index=True)
    with cols[1]:
        first_knockout = context["first_knockout_match"]
        if first_knockout is None:
            st.caption("Projected Knockout Entry")
            st.info("The modal bracket currently projects a group-stage exit.")
        else:
            st.caption("Projected First Knockout Match")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Stage": first_knockout["Stage"],
                            "Opponent": first_knockout["Opponent"],
                            "Matchup Win %": format_percent(first_knockout["Matchup Win %"]),
                            "Projected Winner": first_knockout["Projected Winner"],
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.caption("Projected Knockout Path")
    if context["knockout_path"].empty:
        st.info("No knockout path is projected for this team in the modal bracket.")
    else:
        path_table = context["knockout_path"].copy()
        path_table["Matchup Win %"] = path_table["Matchup Win %"].map(format_percent)
        st.dataframe(path_table, use_container_width=True, hide_index=True)


def render_team_report_card_page() -> None:
    """Render the dedicated V3 team report-card page."""
    home.inject_styles()
    st.markdown(f"<style>{report_card_css()}</style>", unsafe_allow_html=True)

    default_settings = home.default_simulation_settings()
    simulation_options = list(home.SIMULATION_OPTIONS.keys())
    default_simulation_label = str(default_settings["simulation_label"])
    default_simulation_index = simulation_options.index(default_simulation_label) if default_simulation_label in simulation_options else 0

    st.sidebar.subheader("Report Card Controls")
    simulation_label = st.sidebar.radio(
        "Simulations",
        simulation_options,
        index=default_simulation_index,
        key="team_report_card_simulation_label",
    )
    form_match_window = st.sidebar.slider(
        "Recent-match window",
        min_value=home.FORM_WINDOW_MIN,
        max_value=home.FORM_WINDOW_MAX,
        value=int(default_settings["form_match_window"]),
        key="team_report_card_form_match_window",
    )

    simulations = home.SIMULATION_OPTIONS[simulation_label]
    dataset = build_report_card_dataset(simulations=simulations, match_window=form_match_window)
    team_choices = (
        dataset["dashboard_df"]
        .loc[:, ["team_id", "display_name", "group_code"]]
        .drop_duplicates(subset=["team_id"], keep="first")
        .sort_values(["group_code", "display_name"], kind="stable")
        .reset_index(drop=True)
    )
    team_ids = team_choices["team_id"].astype(str).tolist()
    labels = [f'{row.display_name} (Group {row.group_code})' for row in team_choices.itertuples(index=False)]
    query_team_id = get_query_team_param()
    selected_index = team_ids.index(query_team_id) if query_team_id in team_ids else 0
    selected_team_id = st.sidebar.selectbox(
        "Team",
        team_ids,
        index=selected_index,
        format_func=lambda value: labels[team_ids.index(value)],
        key="team_report_card_team_id",
    )
    set_query_team_param(selected_team_id)

    home.render_dashboard_header(
        home.load_world_cup_logo_data_uri(),
        dataset["metadata"],
        simulations,
        title="World Cup 2026 Team Report Card",
        model_version=home.V3_MODEL_VERSION,
        model_label=home.V3_MODEL_LABEL,
    )
    st.caption(
        f"Model {home.V3_MODEL_VERSION}: {home.V3_MODEL_SUMMARY}. "
        f"This report card uses the V3 Poisson model, the last {form_match_window} Elo-rated matches, "
        "historical World Cup pedigree, and the modal deterministic bracket."
    )

    context = select_report_card_context(dataset, selected_team_id)
    render_identity_header(context)
    render_subject_cards(context)
    render_charts(context)
    render_recent_performance(context)
    render_prediction_outlook(context)
    render_fixtures_and_path(context)
