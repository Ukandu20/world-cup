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
)
from world_cup_sim.shared import build_2022_backtest_data, outcome_label_from_scoreline, resolve_training_anchor_date
from world_cup_sim.v2 import build_v2_training_frame, run_v2_backtest_2022
from world_cup_sim.v3 import run_v3_2022_backtest


VALIDATION_DIR = ROOT / "data" / "processed" / "validation"
MODEL_CARD_PATH = ROOT / "docs" / "model_card.md"
DEFAULT_SIMULATIONS = 20000
DEFAULT_SEED = 20260403
DEFAULT_MATCH_WINDOW = RECENT_MATCH_WINDOW
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2022 holdout model-validation artifacts.")
    parser.add_argument("--match-window", type=int, default=DEFAULT_MATCH_WINDOW)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    dataset = build_2022_backtest_data()
    edition_start = pd.to_datetime(pd.DataFrame(dataset["results_df"])["date"], errors="coerce").min()
    training_end_date = str((pd.Timestamp(edition_start) - pd.Timedelta(days=1)).date())
    training_df = build_v2_training_frame(
        match_window=match_window,
        exclude_editions=(2022,),
        training_scope=TRAINING_SCOPE_ALL_INTERNATIONAL,
        reference_edition_year=2022,
        end_date=training_end_date,
    )
    if "date" in training_df.columns and (pd.to_datetime(training_df["date"], errors="coerce") >= pd.Timestamp(edition_start)).any():
        raise ValueError("Elo baseline training data leaked 2022 matches")

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
            "anchor_year": 1998,
            "anchor_date": resolve_training_anchor_date(2022).strftime("%Y-%m-%d"),
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


def build_summary_row(
    model_id: str,
    model_label: str,
    model_type: str,
    backtest: dict[str, object],
    feature_columns: list[str],
    tournament_simulated: bool,
    training_metadata: dict[str, object],
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
        "holdout": "2022 FIFA World Cup",
        "feature_columns": feature_columns,
        "tournament_simulated": bool(tournament_simulated),
        **training_metadata,
    }
    for field in METRIC_FIELDS:
        row[field] = metric_float(metrics.get(field))
    row["predicted_champion_team_id"] = metrics.get("predicted_champion_team_id", "")
    row["actual_champion_team_id"] = metrics.get("actual_champion_team_id", "ARG")
    return row


def build_validation_artifacts(
    match_window: int = DEFAULT_MATCH_WINDOW,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    baseline = run_elo_baseline_2022(match_window=match_window, seed=seed)
    v2_world_cup = run_v2_backtest_2022(
        match_window=match_window,
        simulations=simulations,
        seed=seed,
        training_scope=TRAINING_SCOPE_WORLD_CUP_ONLY,
    )
    v2_all_international = run_v2_backtest_2022(
        match_window=match_window,
        simulations=simulations,
        seed=seed,
        training_scope=TRAINING_SCOPE_ALL_INTERNATIONAL,
    )
    v3_world_cup = run_v3_2022_backtest(
        match_window=match_window,
        simulations=simulations,
        seed=seed,
        training_scope=TRAINING_SCOPE_WORLD_CUP_ONLY,
    )
    v3_all_international = run_v3_2022_backtest(
        match_window=match_window,
        simulations=simulations,
        seed=seed,
        training_scope=TRAINING_SCOPE_ALL_INTERNATIONAL,
    )

    model_rows = [
        build_summary_row(
            "baseline_elo",
            "Elo-only baseline",
            "Multinomial logistic regression",
            baseline,
            list(baseline["feature_columns"]),
            bool(baseline["tournament_simulated"]),
            dict(baseline["training_metadata"]),
        ),
        build_summary_row(
            "v2_world_cup_only",
            "V2 World Cup only",
            "Historical World Cup multinomial regression + Monte Carlo",
            v2_world_cup,
            list(V2_FEATURE_COLUMNS),
            True,
            dict(v2_world_cup["training_metadata"]),
        ),
        build_summary_row(
            "v2_all_international_since_anchor",
            "V2 all international since anchor",
            "Historical international multinomial regression + Monte Carlo",
            v2_all_international,
            list(V2_FEATURE_COLUMNS),
            True,
            dict(v2_all_international["training_metadata"]),
        ),
        build_summary_row(
            "v3_world_cup_only",
            "V3 World Cup only",
            "Historical World Cup Poisson regression + Monte Carlo",
            v3_world_cup,
            list(V3_FEATURE_COLUMNS),
            True,
            dict(v3_world_cup["training_metadata"]),
        ),
        build_summary_row(
            "v3_all_international_since_anchor",
            "V3 all international since anchor",
            "Historical international Poisson regression + Monte Carlo",
            v3_all_international,
            list(V3_FEATURE_COLUMNS),
            True,
            dict(v3_all_international["training_metadata"]),
        ),
    ]

    match_predictions = pd.concat(
        [
            pd.DataFrame(baseline["match_predictions"]),
            normalize_backtest_match_predictions(v2_world_cup, "v2_world_cup_only", "V2 World Cup only"),
            normalize_backtest_match_predictions(v2_all_international, "v2_all_international_since_anchor", "V2 all international since anchor"),
            normalize_backtest_match_predictions(v3_world_cup, "v3_world_cup_only", "V3 World Cup only"),
            normalize_backtest_match_predictions(v3_all_international, "v3_all_international_since_anchor", "V3 all international since anchor"),
        ],
        ignore_index=True,
    )
    team_backtest = pd.concat(
        [
            normalize_team_backtest(v2_world_cup, "v2_world_cup_only", "V2 World Cup only"),
            normalize_team_backtest(v2_all_international, "v2_all_international_since_anchor", "V2 all international since anchor"),
            normalize_team_backtest(v3_world_cup, "v3_world_cup_only", "V3 World Cup only"),
            normalize_team_backtest(v3_all_international, "v3_all_international_since_anchor", "V3 all international since anchor"),
        ],
        ignore_index=True,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "validation_window": {
            "holdout": "2022 FIFA World Cup",
            "match_window": int(match_window),
            "simulations": int(simulations),
            "seed": int(seed),
        },
        "models": model_rows,
        "match_predictions": match_predictions,
        "team_backtest": team_backtest,
    }


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_validation_artifacts(artifacts: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_validation_2022.json"
    match_predictions_path = output_dir / "match_predictions_2022.csv"
    team_backtest_path = output_dir / "team_backtest_2022.csv"

    payload = {
        "generated_at_utc": artifacts["generated_at_utc"],
        "validation_window": artifacts["validation_window"],
        "models": artifacts["models"],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(match_predictions_path, pd.DataFrame(artifacts["match_predictions"]))
    write_csv(team_backtest_path, pd.DataFrame(artifacts["team_backtest"]))
    return {
        "json": json_path,
        "match_predictions": match_predictions_path,
        "team_backtest": team_backtest_path,
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


def build_model_card_markdown(payload: dict[str, object]) -> str:
    models = list(payload["models"])
    validation = dict(payload["validation_window"])
    return f"""# World Cup Forecasting Model Card

## Purpose

This project estimates preseason FIFA World Cup 2026 team and tournament probabilities. It is intended as a forecasting and portfolio dashboard, not as betting advice or a match-day injury-aware prediction service.

## Validation Snapshot

The committed validation artifact is `data/processed/validation/model_validation_2022.json`. The holdout is the 2022 FIFA World Cup. Each trained row uses the same anchor policy: for the 2022 holdout, training starts at the 1998 World Cup kickoff and ends before the first 2022 World Cup match.

- Match window: `{validation["match_window"]}`
- Monte Carlo simulations: `{validation["simulations"]:,}`
- Seed: `{validation["seed"]}`

{markdown_metric_table(models)}

The Elo-only baseline is match-level only. Its tournament-stage fields are set to zero and flagged with `tournament_simulated=false` in the JSON artifact.

## Model Families

- **Baseline:** multinomial logistic regression using only pre-match Elo difference, trained on all international matches since the anchor date with tournament sample weights.
- **V2:** multinomial logistic regression using Elo, recent form, goal profile, and prior World Cup history differences. It is validated under both World-Cup-only and all-international training scopes.
- **V3:** Poisson expected-goals model using Elo, form, historical pedigree, host/neutral-site context, and competition importance. It is validated under both World-Cup-only and all-international training scopes.

## Training Scopes And Weights

- `world_cup_only`: historical World Cup finals matches from the anchor World Cup onward.
- `all_international_since_anchor`: all international matches from the anchor World Cup kickoff onward.
- Sample-weight policy: `{SAMPLE_WEIGHT_POLICY}`.

## Leakage Controls

- All validation rows use a cutoff before the first 2022 World Cup match.
- Team features for the 2022 holdout are built from pre-tournament data.
- Tournament probabilities are evaluated against actual 2022 outcomes after simulation.

## Limitations

- The model does not ingest player-level squad quality, injuries, lineups, market odds, or tactical matchups.
- The 2026 forecast is preseason-oriented and should not be interpreted as live match pricing.
- The 2022 holdout is a useful sanity check, not a full multi-tournament validation suite.
- Penalty shootouts and extra time are simplified relative to real match dynamics.
"""


def write_model_card(payload: dict[str, object], model_card_path: Path) -> None:
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(build_model_card_markdown(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifacts = build_validation_artifacts(
        match_window=args.match_window,
        simulations=args.simulations,
        seed=args.seed,
    )
    paths = write_validation_artifacts(artifacts, Path(args.output_dir))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if not args.skip_docs:
        write_model_card(payload, Path(args.model_card_path))
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
