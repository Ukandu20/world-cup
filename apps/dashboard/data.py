from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import CHAMPION_TROPHY_PATH, DATA_DIR, ROOT, WORLD_CUP_LOGO_PATH

VALIDATION_ARTIFACT_PATH = ROOT / "data" / "processed" / "validation" / "aggregate_validation.json"


def _empty_validation_artifacts(path: Path, warning: str) -> dict[str, Any]:
    empty = pd.DataFrame()
    return {
        "available": False,
        "artifact_path": str(path),
        "warning": warning,
        "aggregate_rows": empty.copy(),
        "per_fold_rows": empty.copy(),
        "aggregate_model_rows": empty.copy(),
        "calibration_rows": empty.copy(),
        "aggregate_display": empty.copy(),
        "per_fold_display": empty.copy(),
        "calibration_display": empty.copy(),
        "headline_findings": {},
        "anomaly_notes": [],
        "metadata": {"artifact_path": str(path)},
    }


def _scope_label(scope: object) -> str:
    value = "" if pd.isna(scope) else str(scope)
    return value.replace("_", " ")


def _model_family(model_id: object, model_label: object = "") -> str:
    value = str(model_id or model_label or "").lower()
    if value.startswith("baseline"):
        return "Elo"
    if value.startswith("v2"):
        return "V2"
    if value.startswith("v3"):
        return "V3"
    if value.startswith("v4"):
        return "V4"
    return "Other"


def _format_mean_std(mean_value: object, std_value: object, decimals: int = 4) -> str:
    mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    std_numeric = pd.to_numeric(pd.Series([std_value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric) or pd.isna(std_numeric):
        return ""
    return f"{float(mean_numeric):.{decimals}f} +/- {float(std_numeric):.{decimals}f}"


def validation_headline_findings(aggregate_model_rows: pd.DataFrame) -> dict[str, str]:
    """Return the headline winners used by README, model card, and dashboard."""
    if aggregate_model_rows.empty:
        return {}
    rows = aggregate_model_rows.copy()
    log_loss_idx = pd.to_numeric(rows["multiclass_log_loss_mean"], errors="coerce").idxmin()
    brier_idx = pd.to_numeric(rows["multiclass_brier_score_mean"], errors="coerce").idxmin()
    top1_idx = pd.to_numeric(rows["top1_match_accuracy_mean"], errors="coerce").idxmax()
    return {
        "best_log_loss": str(rows.loc[log_loss_idx, "model_label"]),
        "best_brier": str(rows.loc[brier_idx, "model_label"]),
        "best_top1": str(rows.loc[top1_idx, "model_label"]),
        "dominance": "No model consistently dominates all headline metrics.",
    }


def validation_anomaly_notes(per_fold_rows: pd.DataFrame, aggregate_model_rows: pd.DataFrame) -> list[str]:
    """Build validation anomaly notes from the same thresholds used in the model card."""
    notes: list[str] = []
    if not aggregate_model_rows.empty:
        for row in aggregate_model_rows.to_dict("records"):
            label = str(row.get("model_label", row.get("model_id", "model")))
            scope = _scope_label(row.get("training_scope", ""))
            log_std = pd.to_numeric(pd.Series([row.get("multiclass_log_loss_std")]), errors="coerce").iloc[0]
            brier_std = pd.to_numeric(pd.Series([row.get("multiclass_brier_score_std")]), errors="coerce").iloc[0]
            draw_pred = pd.to_numeric(pd.Series([row.get("draw_rate_predicted_mean")]), errors="coerce").iloc[0]
            draw_actual = pd.to_numeric(pd.Series([row.get("draw_rate_actual_mean")]), errors="coerce").iloc[0]
            if pd.notna(log_std) and float(log_std) > 0.02:
                notes.append(f"{label} ({scope}) has log loss std {float(log_std):.4f} across folds.")
            if pd.notna(brier_std) and float(brier_std) > 0.02:
                notes.append(f"{label} ({scope}) has Brier std {float(brier_std):.4f} across folds.")
            if pd.notna(draw_pred) and pd.notna(draw_actual) and abs(float(draw_pred) - float(draw_actual)) > 3.0:
                notes.append(
                    f"{label} ({scope}) predicts draws at {float(draw_pred):.1f}% vs actual {float(draw_actual):.1f}%."
                )

    if not per_fold_rows.empty:
        rows = per_fold_rows.copy()
        rows["model_family"] = rows["model"].map(lambda value: _model_family(value))
        for (fold_year, scope), group in rows.groupby(["fold_year", "scope"], dropna=False):
            v4_rows = group[group["model_family"] == "V4"]
            simpler_rows = group[group["model_family"].isin(["Elo", "V2", "V3"])]
            if v4_rows.empty or simpler_rows.empty:
                continue
            v4_best_log = pd.to_numeric(v4_rows["log_loss"], errors="coerce").min()
            v4_best_brier = pd.to_numeric(v4_rows["brier"], errors="coerce").min()
            for simpler in simpler_rows.to_dict("records"):
                log_loss = pd.to_numeric(pd.Series([simpler.get("log_loss")]), errors="coerce").iloc[0]
                brier = pd.to_numeric(pd.Series([simpler.get("brier")]), errors="coerce").iloc[0]
                if pd.notna(log_loss) and pd.notna(brier) and float(log_loss) < v4_best_log and float(brier) < v4_best_brier:
                    notes.append(
                        f"{int(fold_year)} {str(scope).replace('_', ' ')}: {simpler.get('model')} beats V4 on log loss and Brier."
                    )
    return notes


def _aggregate_display_frame(aggregate_model_rows: pd.DataFrame) -> pd.DataFrame:
    if aggregate_model_rows.empty:
        return pd.DataFrame()
    rows = aggregate_model_rows.copy()
    rows["model_family"] = rows.apply(lambda row: _model_family(row.get("model_id"), row.get("model_label")), axis=1)
    rows["scope_label"] = rows["training_scope"].map(_scope_label)
    return pd.DataFrame(
        {
            "model": rows["model_label"],
            "scope": rows["scope_label"],
            "log_loss mean+/-std": [
                _format_mean_std(mean_value, std_value, 4)
                for mean_value, std_value in zip(rows["multiclass_log_loss_mean"], rows["multiclass_log_loss_std"])
            ],
            "brier mean+/-std": [
                _format_mean_std(mean_value, std_value, 4)
                for mean_value, std_value in zip(rows["multiclass_brier_score_mean"], rows["multiclass_brier_score_std"])
            ],
            "top1_acc mean+/-std": [
                _format_mean_std(mean_value, std_value, 1)
                for mean_value, std_value in zip(rows["top1_match_accuracy_mean"], rows["top1_match_accuracy_std"])
            ],
            "champion_hits/3": rows["champion_hits"].map(lambda value: f"{int(value)}/3" if pd.notna(value) else ""),
            "model_family": rows["model_family"],
            "model_id": rows["model_id"],
            "training_scope": rows["training_scope"],
        }
    )


def _per_fold_display_frame(per_fold_rows: pd.DataFrame) -> pd.DataFrame:
    if per_fold_rows.empty:
        return pd.DataFrame()
    rows = per_fold_rows.copy()
    rows["model_family"] = rows["model"].map(lambda value: _model_family(value))
    rows["scope_label"] = rows["scope"].map(_scope_label)
    columns = {
        "fold_year": "fold_year",
        "model": "model",
        "scope_label": "scope",
        "log_loss": "log_loss",
        "brier": "brier",
        "top1_acc_pct": "top1_acc",
        "draw_pred_pct": "draw_pred",
        "draw_actual_pct": "draw_actual",
        "r16_hits": "r16_hits",
        "sf_hits": "sf_hits",
        "champion_hit": "champion_hit",
        "model_family": "model_family",
    }
    display = rows.loc[:, [column for column in columns if column in rows.columns]].rename(columns=columns)
    for column in ("log_loss", "brier"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    for column in ("top1_acc", "draw_pred", "draw_actual"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: f"{value:.1f}%" if pd.notna(value) else "")
    return display


def _calibration_display_frame(calibration_rows: pd.DataFrame, aggregate_model_rows: pd.DataFrame) -> pd.DataFrame:
    if calibration_rows.empty:
        return pd.DataFrame()
    label_lookup = (
        aggregate_model_rows.set_index("model_id")["model_label"].to_dict()
        if not aggregate_model_rows.empty and {"model_id", "model_label"}.issubset(aggregate_model_rows.columns)
        else {}
    )
    grouped = (
        calibration_rows[calibration_rows["target"].isin(["home_win", "draw", "away_win"])]
        .assign(ece=lambda frame: pd.to_numeric(frame["ece"], errors="coerce"))
        .groupby(["model_id", "target"], as_index=False)["ece"]
        .mean()
    )
    if grouped.empty:
        return pd.DataFrame()
    pivot = grouped.pivot(index="model_id", columns="target", values="ece").reset_index()
    pivot.insert(1, "model", pivot["model_id"].map(lambda value: label_lookup.get(value, value)))
    pivot.insert(2, "model_family", pivot["model_id"].map(lambda value: _model_family(value)))
    for column in ("home_win", "draw", "away_win"):
        if column in pivot.columns:
            pivot[f"{column}_ece"] = pivot[column].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
            pivot = pivot.drop(columns=[column])
        else:
            pivot[f"{column}_ece"] = ""
    return pivot.loc[:, ["model", "home_win_ece", "draw_ece", "away_win_ece", "model_family", "model_id"]]


@st.cache_data(show_spinner=False)
def load_validation_artifacts(path: str | Path | None = None) -> dict[str, Any]:
    """Load committed multi-fold validation artifacts for dashboard display."""
    artifact_path = Path(path) if path is not None else VALIDATION_ARTIFACT_PATH
    if not artifact_path.exists():
        return _empty_validation_artifacts(
            artifact_path,
            f"Validation artifact not found at {artifact_path}. Run `python scripts/run_multi_fold_validation.py` to generate it.",
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _empty_validation_artifacts(
            artifact_path,
            f"Validation artifact at {artifact_path} could not be loaded: {exc}",
        )

    aggregate_rows = pd.DataFrame(payload.get("aggregate_rows", []))
    per_fold_rows = pd.DataFrame(payload.get("per_fold_rows", []))
    aggregate_model_rows = pd.DataFrame(payload.get("aggregate_model_rows", []))
    calibration_rows = pd.DataFrame(payload.get("calibration", []))
    metadata = {
        "artifact_path": str(artifact_path),
        "generated_at_utc": payload.get("generated_at_utc", ""),
        "run_settings": payload.get("run_settings", {}),
    }
    return {
        "available": True,
        "artifact_path": str(artifact_path),
        "warning": "",
        "aggregate_rows": aggregate_rows,
        "per_fold_rows": per_fold_rows,
        "aggregate_model_rows": aggregate_model_rows,
        "calibration_rows": calibration_rows,
        "aggregate_display": _aggregate_display_frame(aggregate_model_rows),
        "per_fold_display": _per_fold_display_frame(per_fold_rows),
        "calibration_display": _calibration_display_frame(calibration_rows, aggregate_model_rows),
        "headline_findings": validation_headline_findings(aggregate_model_rows),
        "anomaly_notes": validation_anomaly_notes(per_fold_rows, aggregate_model_rows),
        "metadata": metadata,
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
def load_svg_data_uri(svg_path: str) -> str:
    """Load a local SVG file as a data URI for inline display and export."""
    path = Path(svg_path)
    if not path.exists():
        return ""
    svg_bytes = path.read_bytes()
    encoded = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_world_cup_logo_data_uri() -> str:
    """Load the dashboard World Cup logo as a data URI."""
    return load_svg_data_uri(str(WORLD_CUP_LOGO_PATH))


@st.cache_data(show_spinner=False)
def load_champion_trophy_data_uri() -> str:
    """Load the champion trophy SVG as a data URI."""
    return load_svg_data_uri(str(CHAMPION_TROPHY_PATH))


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load the dashboard inputs: teams, ratings, fixtures, lead-in form, and metadata."""
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    groups = pd.read_csv(DATA_DIR / "groups.csv")
    fifa = pd.read_csv(DATA_DIR / "fifa_rank_snapshots.csv")
    elo = pd.read_csv(DATA_DIR / "elo_snapshots.csv")
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    lead_in = pd.read_csv(DATA_DIR / "team_results_lead_in.csv")
    manifest = pd.read_json(DATA_DIR / "manifest.json", typ="series").to_dict()

    text_columns = ["team", "canonical_name", "tournament_name"]
    for frame in (teams, groups, fifa, elo, fixtures, lead_in):
        for column in text_columns:
            if column in frame.columns:
                frame[column] = frame[column].map(fix_mojibake)
    groups["team_name"] = groups["team_name"].map(fix_mojibake)
    if "qualified_team_name" in lead_in.columns:
        lead_in["qualified_team_name"] = lead_in["qualified_team_name"].map(fix_mojibake)
    if "opponent_name" in lead_in.columns:
        lead_in["opponent_name"] = lead_in["opponent_name"].map(fix_mojibake)

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

    team_columns = [
        "team_id",
        "team",
        "tournament_name",
        "canonical_name",
        "flag_icon_code",
        "group_code",
        "confederation",
        "is_host",
        "world_cup_participations",
        "weighted_world_cup_participations",
        "weighted_world_cup_placement_score",
    ]
    available_team_columns = [column_name for column_name in team_columns if column_name in teams.columns]

    merged = (
        groups.merge(
            teams.loc[:, available_team_columns],
            on=["team_id", "group_code"],
            how="left",
        )
        .merge(latest_fifa, on="team_id", how="left")
        .merge(latest_elo, on="team_id", how="left")
    )

    display_name_source = (
        merged["team"]
        if "team" in merged.columns
        else pd.Series(pd.NA, index=merged.index, dtype="object")
    )
    merged["display_name"] = (
        display_name_source.fillna(merged.get("tournament_name")).fillna(merged["team_name"]).map(fix_mojibake)
    )
    merged["world_rank"] = pd.to_numeric(merged["world_rank"], errors="coerce")
    merged["fifa_points"] = pd.to_numeric(merged["fifa_points"], errors="coerce")
    merged["elo_rating"] = pd.to_numeric(merged["elo_rating"], errors="coerce")
    merged["elo_rank"] = pd.to_numeric(merged["elo_rank"], errors="coerce")
    if "world_cup_participations" in merged.columns:
        merged["world_cup_participations"] = pd.to_numeric(merged["world_cup_participations"], errors="coerce")
    if "weighted_world_cup_participations" in merged.columns:
        merged["weighted_world_cup_participations"] = pd.to_numeric(
            merged["weighted_world_cup_participations"],
            errors="coerce",
        )
    if "weighted_world_cup_placement_score" in merged.columns:
        merged["weighted_world_cup_placement_score"] = pd.to_numeric(
            merged["weighted_world_cup_placement_score"],
            errors="coerce",
        )

    metadata = {
        "build_date": manifest.get("build_date", ""),
        "fifa_snapshot_date": latest_fifa["fifa_snapshot_date"].dropna().max(),
        "elo_snapshot_date": latest_elo["elo_snapshot_date"].dropna().max(),
    }
    return merged, fixtures, lead_in, metadata
