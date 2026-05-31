from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_model_validation import (  # noqa: E402
    DEFAULT_MATCH_WINDOW,
    DEFAULT_SEED,
    build_model_card_markdown,
    build_model_runner_registry,
    artifacts_from_fold_results,
    markdown_aggregate_metric_table,
)
from world_cup_sim.validation import aggregate_across_folds, save_fold_result, validate_all_folds  # noqa: E402
from world_cup_sim.validation_data import load_all_data  # noqa: E402
from world_cup_sim.validation_folds import build_all_folds  # noqa: E402


SIMULATIONS = 20000
SEED = 20260403
HOLDOUT_YEARS = (2014, 2018, 2022)
VALIDATION_DIR = ROOT / "data" / "processed" / "validation"
MODEL_CARD_PATH = ROOT / "docs" / "model_card.md"
README_PATH = ROOT / "README.md"

T = TypeVar("T")


def run_step(name: str, fn: Callable[[], T]) -> T:
    print(f"\n=== {name} ===", flush=True)
    try:
        result = fn()
        print(f"CONFIRMED: {name}", flush=True)
        return result
    except Exception as exc:  # pragma: no cover - command-line failure path
        tb = traceback.extract_tb(exc.__traceback__)
        frame = tb[-1] if tb else None
        if frame is not None:
            print(f"FAILED: {name}: {type(exc).__name__}: {exc}", flush=True)
            print(f"ERROR_LOCATION: {frame.filename}:{frame.lineno}", flush=True)
        else:
            print(f"FAILED: {name}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        raise SystemExit(1) from exc


def _ordered_per_fold_table(fold_results_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fold_year",
        "model",
        "scope",
        "log_loss",
        "brier",
        "top1_acc_pct",
        "draw_pred_pct",
        "draw_actual_pct",
        "r16_hits",
        "sf_hits",
        "champion_hit",
    ]
    return fold_results_df.loc[:, columns].sort_values(["fold_year", "model", "scope"], kind="stable").reset_index(drop=True)


def _interpret_aggregate(aggregate_models: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(aggregate_models)
    best_log_loss = frame.loc[pd.to_numeric(frame["multiclass_log_loss_mean"], errors="coerce").idxmin()]
    best_brier = frame.loc[pd.to_numeric(frame["multiclass_brier_score_mean"], errors="coerce").idxmin()]
    best_top1 = frame.loc[pd.to_numeric(frame["top1_match_accuracy_mean"], errors="coerce").idxmax()]
    dominates = (
        best_log_loss["model_id"] == best_brier["model_id"] == best_top1["model_id"]
        and best_log_loss["training_scope"] == best_brier["training_scope"] == best_top1["training_scope"]
    )
    dominance_sentence = (
        f"{best_log_loss['model_label']} ({best_log_loss['training_scope']}) consistently dominates the headline match metrics."
        if dominates
        else "No model consistently dominates all headline match metrics across the folds."
    )
    return (
        f"{best_log_loss['model_label']} ({best_log_loss['training_scope']}) has the lowest mean log loss, "
        f"while {best_brier['model_label']} ({best_brier['training_scope']}) has the lowest mean Brier score. "
        f"{best_top1['model_label']} ({best_top1['training_scope']}) has the highest mean top-1 accuracy. "
        f"{dominance_sentence}"
    )


def _update_readme(aggregate_models: list[dict[str, object]]) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    start = readme.index("## Validation Summary")
    end = readme.index("## Project Tour")
    replacement = f"""## Validation Summary

The published validation now uses 2014, 2018, and 2022 World Cup holdout folds with `20,000` simulations, match window `{DEFAULT_MATCH_WINDOW}`, and seed `{SEED}`.

{markdown_aggregate_metric_table(aggregate_models)}

{_interpret_aggregate(aggregate_models)}

See [docs/model_card.md](docs/model_card.md) for per-fold metrics, calibration details, anomaly flags, and limitations.

"""
    README_PATH.write_text(readme[:start] + replacement + readme[end:], encoding="utf-8")


def main() -> None:
    state: dict[str, object] = {}

    def step_1_validate() -> dict[str, object]:
        data = load_all_data()
        folds = build_all_folds(data, holdout_years=HOLDOUT_YEARS)
        result = validate_all_folds(
            build_model_runner_registry(),
            [],
            data,
            folds=folds,
            n_simulations=SIMULATIONS,
            seed=SEED,
            match_window=DEFAULT_MATCH_WINDOW,
        )
        fold_results = list(result["fold_results"])
        if len(fold_results) != 21:
            raise AssertionError(f"Expected 21 fold results, got {len(fold_results)}")
        bad_matches = [fr.to_row() for fr in fold_results if int(fr.n_holdout_matches) != 64]
        if bad_matches:
            raise AssertionError(f"Expected every fold/model to have 64 holdout matches, got {bad_matches}")
        state["validation_result"] = result
        return result

    validation_result = run_step("Step 1 - validate_all_folds", step_1_validate)

    def step_2_aggregate() -> dict[str, pd.DataFrame]:
        fold_results_df = pd.DataFrame([fr.to_row() for fr in validation_result["fold_results"]])
        aggregate_df = aggregate_across_folds(fold_results_df)
        per_fold_table = _ordered_per_fold_table(fold_results_df)
        print("\nPER-FOLD TABLE")
        print(per_fold_table.to_string(index=False))
        print("\nAGGREGATE TABLE")
        print(aggregate_df.to_string(index=False))
        state["fold_results_df"] = fold_results_df
        state["aggregate_df"] = aggregate_df
        return {"fold_results_df": fold_results_df, "aggregate_df": aggregate_df}

    tables = run_step("Step 2 - aggregate_across_folds", step_2_aggregate)

    def step_3_write_artifacts() -> dict[str, object]:
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        fold_paths = [save_fold_result(fr, VALIDATION_DIR) for fr in validation_result["fold_results"]]
        artifact_payload = artifacts_from_fold_results(
            list(validation_result["fold_results"]),
            match_window=DEFAULT_MATCH_WINDOW,
            simulations=SIMULATIONS,
            seed=SEED,
            holdout_years=HOLDOUT_YEARS,
            include_holdout_year=True,
        )
        aggregate_payload = {
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "run_settings": {
                "holdout_years": list(HOLDOUT_YEARS),
                "match_window": DEFAULT_MATCH_WINDOW,
                "simulations": SIMULATIONS,
                "seed": SEED,
            },
            "per_fold_rows": tables["fold_results_df"].to_dict("records"),
            "aggregate_rows": tables["aggregate_df"].to_dict("records"),
            "aggregate_model_rows": artifact_payload["aggregate_models"],
            "calibration": artifact_payload["calibration"],
        }
        aggregate_path = VALIDATION_DIR / "aggregate_validation.json"
        aggregate_path.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
        if len(aggregate_payload["per_fold_rows"]) != 21:
            raise AssertionError("aggregate_validation.json payload does not contain 21 per-fold rows")
        if len(aggregate_payload["aggregate_rows"]) != 7:
            raise AssertionError("aggregate_validation.json payload does not contain 7 aggregate rows")
        state["artifact_payload"] = artifact_payload
        state["aggregate_path"] = aggregate_path
        return {"fold_paths": fold_paths, "aggregate_path": aggregate_path}

    run_step("Step 3 - write artifacts", step_3_write_artifacts)

    def step_4_model_card() -> Path:
        artifact_payload = dict(state["artifact_payload"])
        MODEL_CARD_PATH.write_text(build_model_card_markdown(artifact_payload), encoding="utf-8")
        text = MODEL_CARD_PATH.read_text(encoding="utf-8")
        required = ["#### Per-Fold", "#### Aggregate", "### Calibration", "### Anomaly Flags"]
        missing = [value for value in required if value not in text]
        if missing:
            raise AssertionError(f"model card is missing required sections: {missing}")
        return MODEL_CARD_PATH

    run_step("Step 4 - update model card", step_4_model_card)

    def step_5_readme() -> Path:
        artifact_payload = dict(state["artifact_payload"])
        _update_readme(list(artifact_payload["aggregate_models"]))
        text = README_PATH.read_text(encoding="utf-8")
        if "See [docs/model_card.md](docs/model_card.md)" not in text:
            raise AssertionError("README validation section does not link to docs/model_card.md")
        if "| fold_year |" in text:
            raise AssertionError("README contains a per-fold validation table")
        return README_PATH

    run_step("Step 5 - update README", step_5_readme)


if __name__ == "__main__":
    main()
