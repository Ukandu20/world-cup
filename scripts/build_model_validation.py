from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_cup_sim.constants import (
    RECENT_MATCH_WINDOW,
    SAMPLE_WEIGHT_POLICY,
    TRAINING_SCOPE_ALL_INTERNATIONAL,
    TRAINING_SCOPE_WORLD_CUP_ONLY,
    V2_FEATURE_COLUMNS,
    V2_OUTCOME_LABELS,
    V3_FEATURE_COLUMNS,
    V4_FEATURE_COLUMNS,
)
from world_cup_sim.shared import (
    build_historical_world_cup_backtest_data,
    outcome_label_from_scoreline,
    resolve_training_anchor_date,
    tournament_holdout_label,
    validation_artifact_filenames,
)
from world_cup_sim.calibration import CalibrationResult, compute_calibration
from world_cup_sim.validation_data import load_all_data
from world_cup_sim.validation import FoldResult, validate_all_folds
from world_cup_sim.validation_folds import build_all_folds
from world_cup_sim.v2 import build_v2_training_frame, run_v2_historical_backtest
from world_cup_sim.v3 import run_v3_historical_backtest
from world_cup_sim.v4 import run_v4_historical_backtest


VALIDATION_DIR = ROOT / "data" / "processed" / "validation"
MODEL_CARD_PATH = ROOT / "docs" / "model_card.md"
DEFAULT_SIMULATIONS = 20000
DEFAULT_SEED = 20260403
DEFAULT_MATCH_WINDOW = RECENT_MATCH_WINDOW
DEFAULT_FOLD_YEARS = (2014, 2018, 2022)
METRIC_FIELDS = (
    "multiclass_log_loss",
    "multiclass_brier_score",
    "top1_match_accuracy",
    "draw_rate_actual",
    "draw_rate_predicted",
    "round_of_16_hit_count",
    "semifinal_hit_count",
    "exact_champion_hit",
)


def build_model_runner_registry() -> dict[str, dict[str, object]]:
    """Return validation model runners plus artifact metadata."""
    return {
        "baseline_elo": {
            "runner": lambda **kwargs: run_elo_baseline_historical(**kwargs),
            "scopes": [TRAINING_SCOPE_ALL_INTERNATIONAL],
            "label": "Elo-only baseline",
            "type": "Multinomial logistic regression",
            "feature_columns": ["elo_diff"],
            "tournament_simulated": False,
        },
        "v2_world_cup_only": {
            "runner": lambda **kwargs: run_v2_historical_backtest(**kwargs),
            "scopes": [TRAINING_SCOPE_WORLD_CUP_ONLY],
            "label": "V2 World Cup only",
            "type": "Historical World Cup multinomial regression + Monte Carlo",
            "feature_columns": list(V2_FEATURE_COLUMNS),
            "tournament_simulated": True,
        },
        "v2_all_international_since_anchor": {
            "runner": lambda **kwargs: run_v2_historical_backtest(**kwargs),
            "scopes": [TRAINING_SCOPE_ALL_INTERNATIONAL],
            "label": "V2 all international since anchor",
            "type": "Historical international multinomial regression + Monte Carlo",
            "feature_columns": list(V2_FEATURE_COLUMNS),
            "tournament_simulated": True,
        },
        "v3_world_cup_only": {
            "runner": lambda **kwargs: run_v3_historical_backtest(**kwargs),
            "scopes": [TRAINING_SCOPE_WORLD_CUP_ONLY],
            "label": "V3 World Cup only",
            "type": "Historical World Cup Poisson regression + Monte Carlo",
            "feature_columns": list(V3_FEATURE_COLUMNS),
            "tournament_simulated": True,
        },
        "v3_all_international_since_anchor": {
            "runner": lambda **kwargs: run_v3_historical_backtest(**kwargs),
            "scopes": [TRAINING_SCOPE_ALL_INTERNATIONAL],
            "label": "V3 all international since anchor",
            "type": "Historical international Poisson regression + Monte Carlo",
            "feature_columns": list(V3_FEATURE_COLUMNS),
            "tournament_simulated": True,
        },
        "v4_world_cup_only": {
            "runner": lambda **kwargs: run_v4_historical_backtest(**kwargs),
            "scopes": [TRAINING_SCOPE_WORLD_CUP_ONLY],
            "label": "V4 World Cup only",
            "type": "Enhanced World Cup Poisson regression + Monte Carlo",
            "feature_columns": list(V4_FEATURE_COLUMNS),
            "tournament_simulated": True,
        },
        "v4_all_international_since_anchor": {
            "runner": lambda **kwargs: run_v4_historical_backtest(**kwargs),
            "scopes": [TRAINING_SCOPE_ALL_INTERNATIONAL],
            "label": "V4 all international since anchor",
            "type": "Enhanced Poisson regression + Monte Carlo",
            "feature_columns": list(V4_FEATURE_COLUMNS),
            "tournament_simulated": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-holdout and multi-fold model-validation artifacts.")
    parser.add_argument("--match-window", type=int, default=DEFAULT_MATCH_WINDOW)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-year", type=int, default=2022)
    parser.add_argument("--output-dir", default=str(VALIDATION_DIR))
    parser.add_argument("--model-card-path", default=str(MODEL_CARD_PATH))
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Write JSON/CSV artifacts only.",
    )
    return parser.parse_args()


def metric_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def pct(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}%"


def decimal(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def compute_match_metrics(match_predictions: pd.DataFrame) -> dict[str, float]:
    y_true = np.array(
        [
            [
                1.0 if label == "home_win" else 0.0,
                1.0 if label == "draw" else 0.0,
                1.0 if label == "away_win" else 0.0,
            ]
            for label in match_predictions["actual_outcome"].astype(str)
        ],
        dtype=float,
    )
    y_pred = match_predictions.loc[:, ["home_win_prob", "draw_prob", "away_win_prob"]].astype(float).to_numpy()
    true_class_indices = np.argmax(y_true, axis=1)
    epsilon = 1e-15
    return {
        "multiclass_log_loss": float(
            -np.mean(np.log(np.clip(y_pred[np.arange(len(y_pred)), true_class_indices], epsilon, 1.0)))
        ),
        "multiclass_brier_score": float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1))),
        "top1_match_accuracy": float(match_predictions["top1_correct"].mean() * 100.0),
        "draw_rate_actual": float(y_true[:, 1].mean() * 100.0),
        "draw_rate_predicted": float(y_pred[:, 1].mean() * 100.0),
    }


def run_elo_baseline_2022(match_window: int = DEFAULT_MATCH_WINDOW, seed: int = DEFAULT_SEED) -> dict[str, object]:
    return run_elo_baseline_historical(holdout_year=2022, match_window=match_window, seed=seed)


def run_elo_baseline_historical(
    holdout_year: int = 2022,
    match_window: int = DEFAULT_MATCH_WINDOW,
    simulations: int = 0,
    seed: int = DEFAULT_SEED,
    training_scope: str = TRAINING_SCOPE_ALL_INTERNATIONAL,
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    holdout_year = int(holdout_year)
    dataset = build_historical_world_cup_backtest_data(holdout_year, data=data)
    edition_start = pd.to_datetime(pd.DataFrame(dataset["results_df"])["date"], errors="coerce").min()
    training_end_date = str((pd.Timestamp(edition_start) - pd.Timedelta(days=1)).date())
    training_df = build_v2_training_frame(
        match_window=match_window,
        exclude_editions=(holdout_year,),
        training_scope=TRAINING_SCOPE_ALL_INTERNATIONAL,
        reference_edition_year=holdout_year,
        end_date=training_end_date,
        data=data,
    )
    if "date" in training_df.columns and (pd.to_datetime(training_df["date"], errors="coerce") >= pd.Timestamp(edition_start)).any():
        raise ValueError(f"Elo baseline training data leaked {holdout_year} matches")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for the Elo baseline validation") from exc

    X = training_df.loc[:, ["elo_diff"]].astype(float)
    y = training_df["outcome_label"].astype(str)
    sample_weight = training_df["sample_weight"].astype(float).to_numpy()
    scaler = StandardScaler()
    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=5000,
        random_state=seed,
    )
    model.fit(scaler.fit_transform(X), y, sample_weight=sample_weight)

    results_df = pd.DataFrame(dataset["results_df"]).copy()
    rows: list[dict[str, object]] = []
    for match in results_df.sort_values(["match_number"], kind="stable").itertuples(index=False):
        elo_diff = float(match.home_elo_start) - float(match.away_elo_start)
        X_match = pd.DataFrame({"elo_diff": [elo_diff]})
        probability_by_class = dict(zip(model.classes_, model.predict_proba(scaler.transform(X_match))[0]))
        probability_triplet = tuple(float(probability_by_class.get(label, 0.0)) for label in V2_OUTCOME_LABELS)
        predicted_outcome = V2_OUTCOME_LABELS[int(np.argmax(probability_triplet))]
        actual_outcome = outcome_label_from_scoreline(int(match.home_score), int(match.away_score))
        rows.append(
            {
                "model_id": "baseline_elo",
                "model_label": "Elo-only baseline",
                "match_number": int(match.match_number),
                "stage": str(match.stage),
                "home_team": str(match.home_team),
                "away_team": str(match.away_team),
                "home_score": int(match.home_score),
                "away_score": int(match.away_score),
                "home_win_prob": probability_triplet[0],
                "draw_prob": probability_triplet[1],
                "away_win_prob": probability_triplet[2],
                "predicted_outcome": predicted_outcome,
                "actual_outcome": actual_outcome,
                "top1_correct": predicted_outcome == actual_outcome,
            }
        )

    match_predictions = pd.DataFrame(rows)
    summary_metrics = compute_match_metrics(match_predictions)
    summary_metrics.update(
        {
            "round_of_16_hit_count": 0.0,
            "semifinal_hit_count": 0.0,
            "exact_champion_hit": 0.0,
        }
    )
    return {
        "summary_metrics": summary_metrics,
        "match_predictions": match_predictions,
        "feature_columns": ["elo_diff"],
        "tournament_simulated": False,
        "training_metadata": {
            "training_scope": TRAINING_SCOPE_ALL_INTERNATIONAL,
            "anchor_year": int(training_df["anchor_year"].iloc[0]) if "anchor_year" in training_df.columns else 1998,
            "anchor_date": resolve_training_anchor_date(holdout_year).strftime("%Y-%m-%d"),
            "training_start_date": pd.to_datetime(training_df["date"], errors="coerce").min().strftime("%Y-%m-%d"),
            "training_end_date": pd.to_datetime(training_df["date"], errors="coerce").max().strftime("%Y-%m-%d"),
            "training_match_count": int(len(training_df)),
            "sample_weight_policy": SAMPLE_WEIGHT_POLICY,
        },
    }


def normalize_backtest_match_predictions(backtest: dict[str, object], model_id: str, model_label: str) -> pd.DataFrame:
    match_predictions = pd.DataFrame(backtest["match_predictions"]).copy()
    match_predictions.insert(0, "model_label", model_label)
    match_predictions.insert(0, "model_id", model_id)
    keep_columns = [
        "model_id",
        "model_label",
        "match_number",
        "stage",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "predicted_outcome",
        "actual_outcome",
        "top1_correct",
    ]
    return match_predictions.loc[:, keep_columns]


def normalize_team_backtest(backtest: dict[str, object], model_id: str, model_label: str) -> pd.DataFrame:
    team_table = pd.DataFrame(backtest["team_backtest_table"]).copy()
    team_table.insert(0, "model_label", model_label)
    team_table.insert(0, "model_id", model_id)
    keep_columns = [
        "model_id",
        "model_label",
        "team_id",
        "display_name",
        "group_code",
        "r32_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "champion_prob",
        "actual_stage",
        "actual_r16",
        "actual_sf",
        "actual_final",
        "actual_champion",
    ]
    return team_table.loc[:, [column for column in keep_columns if column in team_table.columns]]


def build_calibration_results(
    match_predictions: pd.DataFrame,
    team_backtest: pd.DataFrame,
    holdout_year: int,
) -> list[CalibrationResult]:
    results: list[CalibrationResult] = []
    if not match_predictions.empty and "model_id" in match_predictions.columns:
        for model_id, group in match_predictions.groupby("model_id", sort=False):
            for target in ("home_win", "draw", "away_win"):
                results.append(
                    compute_calibration(
                        group,
                        target,
                        model=str(model_id),
                        holdout_year=holdout_year,
                    )
                )
    if not team_backtest.empty and {"model_id", "champion_prob", "actual_champion"}.issubset(team_backtest.columns):
        for model_id, group in team_backtest.groupby("model_id", sort=False):
            results.append(
                compute_calibration(
                    group,
                    "champion",
                    model=str(model_id),
                    holdout_year=holdout_year,
                )
            )
    return results


def flatten_calibration_results(results: list[CalibrationResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        for bin_row in result.bins:
            rows.append(
                {
                    "model_id": result.model,
                    "holdout_year": result.holdout_year,
                    "target": result.target,
                    "bin_count": result.bin_count,
                    "brier_score": result.brier_score,
                    "ece": result.ece,
                    "sample_count": result.sample_count,
                    **bin_row,
                }
            )
    return pd.DataFrame(rows)


def calibration_summary_rows(results: list[CalibrationResult]) -> list[dict[str, object]]:
    return [
        {
            "model_id": result.model,
            "holdout_year": result.holdout_year,
            "target": result.target,
            "bin_count": result.bin_count,
            "brier_score": result.brier_score,
            "ece": result.ece,
            "sample_count": result.sample_count,
        }
        for result in results
    ]


def build_summary_row(
    model_id: str,
    model_label: str,
    model_type: str,
    backtest: dict[str, object],
    feature_columns: list[str],
    tournament_simulated: bool,
    training_metadata: dict[str, object],
    holdout_label: str,
) -> dict[str, object]:
    metrics = dict(backtest["summary_metrics"])
    if "draw_rate_actual" not in metrics or "draw_rate_predicted" not in metrics:
        match_metrics = compute_match_metrics(pd.DataFrame(backtest["match_predictions"]))
        metrics.setdefault("draw_rate_actual", match_metrics["draw_rate_actual"])
        metrics.setdefault("draw_rate_predicted", match_metrics["draw_rate_predicted"])
    row: dict[str, object] = {
        "model_id": model_id,
        "model_label": model_label,
        "model_type": model_type,
        "holdout": holdout_label,
        "feature_columns": feature_columns,
        "tournament_simulated": bool(tournament_simulated),
        **training_metadata,
    }
    for field in METRIC_FIELDS:
        row[field] = metric_float(metrics.get(field))
    row["predicted_champion_team_id"] = metrics.get("predicted_champion_team_id", "")
    row["actual_champion_team_id"] = metrics.get("actual_champion_team_id", "")
    return row


def artifacts_from_fold_results(
    fold_results: list[FoldResult],
    *,
    match_window: int,
    simulations: int,
    seed: int,
    holdout_years: tuple[int, ...],
    include_holdout_year: bool = False,
) -> dict[str, object]:
    specs = build_model_runner_registry()
    model_rows: list[dict[str, object]] = []
    match_frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    validation_windows: list[dict[str, object]] = []

    for fr in fold_results:
        spec = dict(specs[fr.model_name])
        backtest = fr.raw_backtest or {}
        holdout_label = tournament_holdout_label(fr.fold_year)
        if not any(int(window["holdout_year"]) == int(fr.fold_year) for window in validation_windows):
            validation_windows.append(
                {
                    "holdout": holdout_label,
                    "holdout_year": int(fr.fold_year),
                    "match_window": int(match_window),
                    "simulations": int(simulations),
                    "seed": int(seed),
                }
            )
        row = build_summary_row(
            fr.model_name,
            str(spec["label"]),
            str(spec["type"]),
            backtest,
            list(spec["feature_columns"]),
            bool(spec["tournament_simulated"]),
            dict(backtest.get("training_metadata", fr.metadata)),
            holdout_label,
        )
        row["fold_year"] = int(fr.fold_year)
        row["multiclass_log_loss"] = float(fr.log_loss_score)
        row["multiclass_brier_score"] = float(fr.brier_score)
        row["top1_match_accuracy"] = float(fr.top1_accuracy * 100.0)
        row["draw_rate_predicted"] = float(fr.mean_draw_prediction * 100.0)
        row["draw_rate_actual"] = float(fr.actual_draw_rate * 100.0)
        row["round_of_16_hit_count"] = float(fr.r16_hits)
        row["semifinal_hit_count"] = float(fr.sf_hits)
        row["exact_champion_hit"] = 1.0 if fr.champion_hit else 0.0
        model_rows.append(row)

        if fr.model_name == "baseline_elo":
            match_predictions = fr.match_predictions_df.copy()
            if "model_id" not in match_predictions.columns:
                match_predictions.insert(0, "model_id", fr.model_name)
            if "model_label" not in match_predictions.columns:
                match_predictions.insert(1, "model_label", str(spec["label"]))
        else:
            match_predictions = normalize_backtest_match_predictions(backtest, fr.model_name, str(spec["label"]))
        if include_holdout_year:
            match_predictions.insert(0, "holdout_year", int(fr.fold_year))
        match_frames.append(match_predictions)

        fold_team_backtest = pd.DataFrame()
        if bool(spec["tournament_simulated"]) and not fr.team_backtest_df.empty:
            team_backtest = normalize_team_backtest(backtest, fr.model_name, str(spec["label"]))
            if include_holdout_year:
                team_backtest.insert(0, "holdout_year", int(fr.fold_year))
            fold_team_backtest = team_backtest.copy()
            team_frames.append(team_backtest)

        fold_calibration_results = build_calibration_results(
            match_predictions.drop(columns=["holdout_year"], errors="ignore"),
            fold_team_backtest.drop(columns=["holdout_year"], errors="ignore"),
            fr.fold_year,
        )
        calibration_rows.extend(calibration_summary_rows(fold_calibration_results))
        calibration_bins = flatten_calibration_results(fold_calibration_results)
        if not calibration_bins.empty:
            calibration_frames.append(calibration_bins)

    holdout_years = tuple(int(year) for year in holdout_years)
    if len(holdout_years) == 1:
        validation_window = {
            "holdout": tournament_holdout_label(holdout_years[0]),
            "holdout_year": holdout_years[0],
            "match_window": int(match_window),
            "simulations": int(simulations),
            "seed": int(seed),
        }
    else:
        validation_window = {
            "holdout": "/".join(str(year) for year in holdout_years) + " FIFA World Cup folds",
            "holdout_years": list(holdout_years),
            "match_window": int(match_window),
            "simulations": int(simulations),
            "seed": int(seed),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "validation_window": validation_window,
        "fold_windows": validation_windows,
        "models": model_rows,
        "aggregate_models": aggregate_model_rows(model_rows) if len(holdout_years) > 1 else [],
        "match_predictions": pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame(),
        "team_backtest": pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame(),
        "calibration": calibration_rows,
        "calibration_bins": pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame(),
    }


def build_validation_artifacts(
    match_window: int = DEFAULT_MATCH_WINDOW,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    holdout_year: int = 2022,
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    holdout_year = int(holdout_year)
    if data is None:
        data = load_all_data()
    validation_result = validate_all_folds(
        build_model_runner_registry(),
        [],
        data,
        folds=build_all_folds(data, holdout_years=(holdout_year,)),
        n_simulations=simulations,
        seed=seed,
        match_window=match_window,
    )
    return artifacts_from_fold_results(
        list(validation_result["fold_results"]),
        match_window=match_window,
        simulations=simulations,
        seed=seed,
        holdout_years=(holdout_year,),
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_validation_artifacts(artifacts: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    holdout_year = int(dict(artifacts["validation_window"]).get("holdout_year", 2022))
    filenames = validation_artifact_filenames(holdout_year)
    json_path = output_dir / filenames["json"]
    match_predictions_path = output_dir / filenames["match_predictions"]
    team_backtest_path = output_dir / filenames["team_backtest"]
    calibration_path = output_dir / f"calibration_{holdout_year}.csv"

    payload = {
        "generated_at_utc": artifacts["generated_at_utc"],
        "validation_window": artifacts["validation_window"],
        "models": artifacts["models"],
        "calibration": artifacts.get("calibration", []),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(match_predictions_path, pd.DataFrame(artifacts["match_predictions"]))
    write_csv(team_backtest_path, pd.DataFrame(artifacts["team_backtest"]))
    write_csv(calibration_path, pd.DataFrame(artifacts.get("calibration_bins", [])))
    return {
        "json": json_path,
        "match_predictions": match_predictions_path,
        "team_backtest": team_backtest_path,
        "calibration": calibration_path,
    }


def aggregate_model_rows(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(model_rows)
    metric_columns = [
        "multiclass_log_loss",
        "multiclass_brier_score",
        "top1_match_accuracy",
        "draw_rate_predicted",
        "draw_rate_actual",
        "round_of_16_hit_count",
        "semifinal_hit_count",
        "exact_champion_hit",
    ]
    aggregate_rows: list[dict[str, object]] = []
    for (model_id, model_label, training_scope), group in frame.groupby(
        ["model_id", "model_label", "training_scope"],
        dropna=False,
        sort=False,
    ):
        row: dict[str, object] = {
            "model_id": model_id,
            "model_label": model_label,
            "training_scope": training_scope,
            "fold_count": int(group["holdout"].nunique()),
            "champion_hits": int(pd.to_numeric(group["exact_champion_hit"], errors="coerce").fillna(0.0).sum()),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if not values.empty else 0.0
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(row)
    return aggregate_rows


def build_multi_fold_validation_artifacts(
    match_window: int = DEFAULT_MATCH_WINDOW,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    holdout_years: tuple[int, ...] = DEFAULT_FOLD_YEARS,
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    if data is None:
        data = load_all_data()
    holdout_years = tuple(int(year) for year in holdout_years)
    validation_result = validate_all_folds(
        build_model_runner_registry(),
        [],
        data,
        folds=build_all_folds(data, holdout_years=holdout_years),
        n_simulations=simulations,
        seed=seed,
        match_window=match_window,
    )
    return artifacts_from_fold_results(
        list(validation_result["fold_results"]),
        match_window=match_window,
        simulations=simulations,
        seed=seed,
        holdout_years=holdout_years,
        include_holdout_year=True,
    )


def write_multi_fold_validation_artifacts(artifacts: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_validation_folds.json"
    fold_results_path = output_dir / "fold_results.csv"
    match_predictions_path = output_dir / "match_predictions_folds.csv"
    team_backtest_path = output_dir / "team_backtest_folds.csv"
    calibration_path = output_dir / "calibration_folds.csv"
    payload = {
        "generated_at_utc": artifacts["generated_at_utc"],
        "validation_window": artifacts["validation_window"],
        "fold_windows": artifacts["fold_windows"],
        "models": artifacts["models"],
        "aggregate_models": artifacts["aggregate_models"],
        "calibration": artifacts.get("calibration", []),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(fold_results_path, pd.DataFrame(artifacts["models"]))
    write_csv(match_predictions_path, pd.DataFrame(artifacts["match_predictions"]))
    write_csv(team_backtest_path, pd.DataFrame(artifacts["team_backtest"]))
    write_csv(calibration_path, pd.DataFrame(artifacts.get("calibration_bins", [])))
    return {
        "json": json_path,
        "fold_results": fold_results_path,
        "match_predictions": match_predictions_path,
        "team_backtest": team_backtest_path,
        "calibration": calibration_path,
    }


def markdown_metric_table(models: list[dict[str, object]]) -> str:
    rows = [
        "| Model | Scope | Matches | Log Loss | Brier | Top-1 Acc. | Draw Pred./Actual | R16 Hits | SF Hits | Champion Hit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        rows.append(
            "| {label} | {scope} | {matches} | {log_loss} | {brier} | {accuracy} | {draw_pred} / {draw_actual} | {r16} | {sf} | {champion} |".format(
                label=model["model_label"],
                scope=model.get("training_scope", ""),
                matches=int(model.get("training_match_count", 0)),
                log_loss=decimal(float(model["multiclass_log_loss"])),
                brier=decimal(float(model["multiclass_brier_score"])),
                accuracy=pct(float(model["top1_match_accuracy"])),
                draw_pred=pct(float(model["draw_rate_predicted"])),
                draw_actual=pct(float(model["draw_rate_actual"])),
                r16=int(float(model["round_of_16_hit_count"])),
                sf=int(float(model["semifinal_hit_count"])),
                champion="Yes" if float(model["exact_champion_hit"]) >= 1.0 else "No",
            )
        )
    return "\n".join(rows)


def markdown_aggregate_metric_table(aggregate_models: list[dict[str, object]]) -> str:
    rows = [
        "| model | scope | log_loss mean+/-std | brier mean+/-std | top1_acc mean+/-std | champion_hits/3 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in aggregate_models:
        rows.append(
            "| {label} | {scope} | {log_loss} +/- {log_loss_std} | {brier} +/- {brier_std} | {accuracy} +/- {accuracy_std} | {champion_hits}/3 |".format(
                label=model["model_label"],
                scope=model.get("training_scope", ""),
                log_loss=decimal(float(model["multiclass_log_loss_mean"])),
                log_loss_std=decimal(float(model["multiclass_log_loss_std"])),
                brier=decimal(float(model["multiclass_brier_score_mean"])),
                brier_std=decimal(float(model["multiclass_brier_score_std"])),
                accuracy=pct(float(model["top1_match_accuracy_mean"])),
                accuracy_std=pct(float(model["top1_match_accuracy_std"])),
                champion_hits=int(model.get("champion_hits", 0)),
            )
        )
    return "\n".join(rows)


def markdown_per_fold_validation_table(models: list[dict[str, object]]) -> str:
    rows = [
        "| fold_year | model | scope | log_loss | brier | top1_acc | draw_pred | draw_actual | r16_hits | sf_hits | champion_hit |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in sorted(models, key=lambda row: (int(row.get("fold_year", 0)), str(row.get("model_id", "")))):
        rows.append(
            "| {fold_year} | {label} | {scope} | {log_loss} | {brier} | {accuracy} | {draw_pred} | {draw_actual} | {r16} | {sf} | {champion} |".format(
                fold_year=int(model.get("fold_year", 0)),
                label=model["model_label"],
                scope=model.get("training_scope", ""),
                log_loss=decimal(float(model["multiclass_log_loss"])),
                brier=decimal(float(model["multiclass_brier_score"])),
                accuracy=pct(float(model["top1_match_accuracy"])),
                draw_pred=pct(float(model["draw_rate_predicted"])),
                draw_actual=pct(float(model["draw_rate_actual"])),
                r16=int(float(model["round_of_16_hit_count"])),
                sf=int(float(model["semifinal_hit_count"])),
                champion="Yes" if float(model["exact_champion_hit"]) >= 1.0 else "No",
            )
        )
    return "\n".join(rows)


def markdown_tournament_fold_table(models: list[dict[str, object]]) -> str:
    rows = [
        "| Model | Scope | Holdout | R16 Hits | SF Hits | Champion Hit |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for model in models:
        if not bool(model.get("tournament_simulated", False)):
            continue
        rows.append(
            "| {label} | {scope} | {holdout} | {r16} | {sf} | {champion} |".format(
                label=model["model_label"],
                scope=model.get("training_scope", ""),
                holdout=model.get("holdout", ""),
                r16=int(float(model["round_of_16_hit_count"])),
                sf=int(float(model["semifinal_hit_count"])),
                champion="Yes" if float(model["exact_champion_hit"]) >= 1.0 else "No",
            )
        )
    return "\n".join(rows)


def markdown_calibration_table(calibration_rows: list[dict[str, object]]) -> str:
    return markdown_calibration_ece_table(calibration_rows)


def markdown_calibration_ece_table(calibration_rows: list[dict[str, object]]) -> str:
    rows = [
        "| model | home_win ECE | draw ECE | away_win ECE |",
        "| --- | ---: | ---: | ---: |",
    ]
    frame = pd.DataFrame(calibration_rows)
    if frame.empty:
        rows.append("| n/a | 0.0000 | 0.0000 | 0.0000 |")
        return "\n".join(rows)
    frame = frame[frame["target"].isin(["home_win", "draw", "away_win"])].copy()
    for model_id, group in frame.groupby("model_id", sort=False):
        ece_by_target = group.groupby("target")["ece"].mean().to_dict()
        rows.append(
            "| {model} | {home_win} | {draw} | {away_win} |".format(
                model=model_id,
                home_win=decimal(float(ece_by_target.get("home_win", 0.0)), 4),
                draw=decimal(float(ece_by_target.get("draw", 0.0)), 4),
                away_win=decimal(float(ece_by_target.get("away_win", 0.0)), 4),
            )
        )
    return "\n".join(rows)


def validation_anomaly_notes(models: list[dict[str, object]], aggregate_models: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    aggregate_frame = pd.DataFrame(aggregate_models)
    if not aggregate_frame.empty:
        for row in aggregate_frame.itertuples(index=False):
            label = str(getattr(row, "model_label"))
            scope = str(getattr(row, "training_scope"))
            log_std = float(getattr(row, "multiclass_log_loss_std", 0.0))
            brier_std = float(getattr(row, "multiclass_brier_score_std", 0.0))
            draw_gap = abs(float(getattr(row, "draw_rate_predicted_mean", 0.0)) - float(getattr(row, "draw_rate_actual_mean", 0.0)))
            if log_std > 0.02 or brier_std > 0.02:
                notes.append(f"{label} ({scope}) has elevated fold dispersion: log-loss std {log_std:.4f}, Brier std {brier_std:.4f}.")
            if draw_gap > 3.0:
                notes.append(f"{label} ({scope}) has mean draw prediction {draw_gap:.1f} percentage points from actual.")

    model_frame = pd.DataFrame(models)
    if not model_frame.empty:
        for (fold_year, scope), group in model_frame.groupby(["fold_year", "training_scope"], sort=True):
            v4_rows = group[group["model_id"].astype(str).str.startswith("v4_")]
            if v4_rows.empty:
                continue
            v4 = v4_rows.sort_values("model_id", kind="stable").iloc[0]
            simpler = group[~group["model_id"].astype(str).str.startswith("v4_")]
            winners = simpler[
                (pd.to_numeric(simpler["multiclass_log_loss"], errors="coerce") < float(v4["multiclass_log_loss"]))
                & (pd.to_numeric(simpler["multiclass_brier_score"], errors="coerce") < float(v4["multiclass_brier_score"]))
            ]
            for winner in winners.itertuples(index=False):
                notes.append(
                    f"{int(fold_year)} {scope}: {winner.model_label} beat {v4['model_label']} on both log loss and Brier."
                )
    return notes


def build_model_card_markdown(payload: dict[str, object]) -> str:
    models = list(payload["models"])
    validation = dict(payload["validation_window"])
    aggregate_models = list(payload.get("aggregate_models", []))
    calibration_rows = list(payload.get("calibration", []))
    is_multi_fold = bool(aggregate_models)
    if is_multi_fold:
        holdout_label = str(validation.get("holdout", "2014/2018/2022 FIFA World Cup folds"))
        validation_artifact = "model_validation_folds.json"
        validation_table = f"""### Match-Level Metrics

#### Per-Fold

{markdown_per_fold_validation_table(models)}

#### Aggregate

{markdown_aggregate_metric_table(aggregate_models)}

### Calibration

{markdown_calibration_ece_table(calibration_rows)}

### Anomaly Flags

{chr(10).join(f"- {note}" for note in validation_anomaly_notes(models, aggregate_models)) or "- No anomaly flags crossed the configured thresholds."}"""
        limitation_holdout = "The rolling holdouts are useful sanity checks, not a complete validation of every tournament format."
    else:
        holdout_year = int(validation.get("holdout_year", 2022))
        holdout_label = str(validation.get("holdout", tournament_holdout_label(holdout_year)))
        validation_artifact = validation_artifact_filenames(holdout_year)["json"]
        validation_table = f"""{markdown_metric_table(models)}

### Calibration

{markdown_calibration_table(calibration_rows)}"""
        limitation_holdout = f"The {holdout_year} holdout is a useful sanity check, not a full multi-tournament validation suite."
    return f"""# World Cup Forecasting Model Card

## Purpose

This project estimates preseason FIFA Men's World Cup 2026 team and tournament probabilities. It is intended as a forecasting and portfolio dashboard, not as betting advice or a match-day injury-aware prediction service.

The current dashboard primary model is V4, an enhanced Poisson expected-goals model. V4 is documented as the production-facing model because it includes the richest match-generation logic, but the validation table should be read across multiple metrics rather than as a single winner-takes-all leaderboard.

## Validation Snapshot

The committed validation artifact is `data/processed/validation/{validation_artifact}`. The validation window is {holdout_label}. Each trained row uses a cutoff before the first match in its holdout World Cup.

- Match window: `{validation["match_window"]}`
- Monte Carlo simulations: `{validation["simulations"]:,}`
- Seed: `{validation["seed"]}`

{validation_table}

The Elo-only baseline is match-level only. Its tournament-stage fields are set to zero and flagged with `tournament_simulated=false` in the JSON artifact.

## How To Read The Metrics

- **Log loss** rewards calibrated probabilities assigned to the actual class; lower is better.
- **Brier score** measures multiclass probability error; lower is better.
- **Top-1 accuracy** measures whether the highest-probability match outcome occurred.
- **Draw Pred./Actual** checks whether the model's average draw probability is close to the observed draw rate.
- **R16, SF, and Champion hits** evaluate tournament simulation outputs against actual holdout advancement outcomes.

## Model Families

- **Baseline:** multinomial logistic regression using only pre-match Elo difference, trained on all international matches since the anchor date with tournament sample weights.
- **V2:** multinomial logistic regression using Elo, recent form, goal profile, and prior World Cup history differences. It is validated under both World-Cup-only and all-international training scopes.
- **V3:** Poisson expected-goals model using Elo, form, historical pedigree, host/neutral-site context, and competition importance. It is validated under both World-Cup-only and all-international training scopes.
- **V4:** enhanced Poisson expected-goals model using quadratic recent form, World Cup last-5 goal-difference features, Dixon-Coles low-score correction, stage multipliers, time-decayed training weights, and alpha selection. It is the current primary dashboard model and is validated under both World-Cup-only and all-international training scopes.

## Training Scopes And Weights

- `world_cup_only`: historical World Cup finals matches from the anchor World Cup onward.
- `all_international_since_anchor`: all international matches from the anchor World Cup kickoff onward.
- Sample-weight policy: `{SAMPLE_WEIGHT_POLICY}`.

## Leakage Controls

- All validation rows use a cutoff before the first holdout World Cup match.
- Team features for each holdout are built from pre-tournament data.
- Tournament probabilities are evaluated against actual holdout outcomes after simulation.
- 2026 forecasts use pre-tournament team metadata, fixtures, rankings, Elo snapshots, and lead-in results only.

## Limitations

- The model does not ingest player-level squad quality, injuries, lineups, market odds, or tactical matchups.
- The 2026 forecast is preseason-oriented and should not be interpreted as live match pricing.
- {limitation_holdout}
- Penalty shootouts and extra time are simplified relative to real match dynamics.
- V4 has more components than V2/V3, so it carries higher overfitting risk until more rolling holdout folds are implemented.
"""


def write_model_card(payload: dict[str, object], model_card_path: Path) -> None:
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(build_model_card_markdown(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_all_data()
    artifacts = build_validation_artifacts(
        match_window=args.match_window,
        simulations=args.simulations,
        seed=args.seed,
        holdout_year=args.holdout_year,
        data=data,
    )
    paths = write_validation_artifacts(artifacts, Path(args.output_dir))
    multi_fold_artifacts = build_multi_fold_validation_artifacts(
        match_window=args.match_window,
        simulations=args.simulations,
        seed=args.seed,
        data=data,
    )
    multi_paths = write_multi_fold_validation_artifacts(multi_fold_artifacts, Path(args.output_dir))
    payload = json.loads(multi_paths["json"].read_text(encoding="utf-8"))
    if not args.skip_docs:
        write_model_card(payload, Path(args.model_card_path))
    output_paths = {f"single_{key}": str(value) for key, value in paths.items()}
    output_paths.update({f"folds_{key}": str(value) for key, value in multi_paths.items()})
    print(json.dumps(output_paths, indent=2))


if __name__ == "__main__":
    main()
