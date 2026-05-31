from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .validation_folds import build_all_folds


OUTCOME_CLASSES = ("home_win", "draw", "away_win")


@dataclass(frozen=True)
class MatchPrediction:
    match_id: str
    home_team_id: str
    away_team_id: str
    predicted_home_win: float
    predicted_draw: float
    predicted_away_win: float
    actual_outcome: str
    stage: str


@dataclass
class FoldResult:
    fold_year: int
    model_name: str
    training_scope: str
    n_training_matches: int
    n_holdout_matches: int
    match_predictions: list[MatchPrediction]
    simulated_r16_teams: set[str]
    simulated_sf_teams: set[str]
    simulated_champion: str
    actual_r16_teams: set[str]
    actual_sf_teams: set[str]
    actual_champion: str
    n_simulations: int
    metadata: dict = field(default_factory=dict)
    raw_backtest: dict[str, object] | None = None
    match_predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_backtest_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def log_loss_score(self) -> float:
        y_pred = np.array(
            [[p.predicted_home_win, p.predicted_draw, p.predicted_away_win] for p in self.match_predictions],
            dtype=float,
        )
        y_true_index = np.array([OUTCOME_CLASSES.index(p.actual_outcome) for p in self.match_predictions], dtype=int)
        return float(-np.mean(np.log(np.clip(y_pred[np.arange(len(y_pred)), y_true_index], 1e-15, 1.0))))

    @property
    def brier_score(self) -> float:
        y_pred = np.array(
            [[p.predicted_home_win, p.predicted_draw, p.predicted_away_win] for p in self.match_predictions],
            dtype=float,
        )
        y_true = np.array([[1.0 if p.actual_outcome == label else 0.0 for label in OUTCOME_CLASSES] for p in self.match_predictions])
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    @property
    def top1_accuracy(self) -> float:
        labels = np.array(OUTCOME_CLASSES)
        predicted = labels[
            np.argmax(
                np.array([[p.predicted_home_win, p.predicted_draw, p.predicted_away_win] for p in self.match_predictions]),
                axis=1,
            )
        ]
        actual = np.array([p.actual_outcome for p in self.match_predictions])
        return float(np.mean(predicted == actual))

    @property
    def mean_draw_prediction(self) -> float:
        return float(np.mean([p.predicted_draw for p in self.match_predictions]))

    @property
    def actual_draw_rate(self) -> float:
        return float(np.mean([p.actual_outcome == "draw" for p in self.match_predictions]))

    @property
    def r16_hits(self) -> int:
        return len(self.simulated_r16_teams.intersection(self.actual_r16_teams))

    @property
    def sf_hits(self) -> int:
        return len(self.simulated_sf_teams.intersection(self.actual_sf_teams))

    @property
    def champion_hit(self) -> bool:
        return self.simulated_champion == self.actual_champion

    def to_row(self) -> dict[str, object]:
        return {
            "fold_year": self.fold_year,
            "model": self.model_name,
            "scope": self.training_scope,
            "n_matches": self.n_holdout_matches,
            "log_loss": round(self.log_loss_score, 4),
            "brier": round(self.brier_score, 4),
            "top1_acc_pct": round(self.top1_accuracy * 100.0, 1),
            "draw_pred_pct": round(self.mean_draw_prediction * 100.0, 1),
            "draw_actual_pct": round(self.actual_draw_rate * 100.0, 1),
            "r16_hits": self.r16_hits,
            "sf_hits": self.sf_hits,
            "champion_hit": "Yes" if self.champion_hit else "No",
            "n_training_matches": self.n_training_matches,
            "n_simulations": self.n_simulations,
        }


def fold_result_from_backtest(
    fold,
    model_name: str,
    training_scope: str,
    backtest: dict[str, object],
    n_simulations: int,
) -> FoldResult:
    predictions = []
    match_predictions_df = pd.DataFrame(backtest["match_predictions"]).copy()
    for row in match_predictions_df.itertuples(index=False):
        predictions.append(
            MatchPrediction(
                match_id=str(getattr(row, "match_id", getattr(row, "match_number", ""))),
                home_team_id=str(getattr(row, "home_team_id", getattr(row, "home_team", ""))),
                away_team_id=str(getattr(row, "away_team_id", getattr(row, "away_team", ""))),
                predicted_home_win=float(getattr(row, "home_win_prob")),
                predicted_draw=float(getattr(row, "draw_prob")),
                predicted_away_win=float(getattr(row, "away_win_prob")),
                actual_outcome=str(getattr(row, "actual_outcome")),
                stage=str(getattr(row, "stage")),
            )
        )

    team_table = pd.DataFrame(backtest.get("team_backtest_table", pd.DataFrame())).copy()
    if team_table.empty or "team_id" not in team_table.columns:
        simulated_r16: set[str] = set()
        simulated_sf: set[str] = set()
        simulated_champion = ""
    else:
        simulated_r16 = set(
            team_table.sort_values(["r16_prob"], ascending=False, kind="stable").head(len(fold.tournament.actual_r16_teams))["team_id"].astype(str)
        )
        simulated_sf = set(team_table.sort_values(["sf_prob"], ascending=False, kind="stable").head(4)["team_id"].astype(str))
        simulated_champion = str(team_table.sort_values(["champion_prob"], ascending=False, kind="stable").iloc[0]["team_id"])
    metadata = dict(backtest.get("training_metadata", {}))
    return FoldResult(
        fold_year=fold.holdout_year,
        model_name=model_name,
        training_scope=training_scope,
        n_training_matches=int(metadata.get("training_match_count", 0) or 0),
        n_holdout_matches=len(predictions),
        match_predictions=predictions,
        simulated_r16_teams=simulated_r16,
        simulated_sf_teams=simulated_sf,
        simulated_champion=simulated_champion,
        actual_r16_teams=fold.tournament.actual_r16_teams,
        actual_sf_teams=fold.tournament.actual_sf_teams,
        actual_champion=str(fold.tournament.actual_champion),
        n_simulations=int(n_simulations),
        metadata=metadata,
        raw_backtest=backtest,
        match_predictions_df=match_predictions_df,
        team_backtest_df=team_table,
    )


def validate_fold(
    fold,
    model_runner: Callable[..., dict[str, object]],
    training_scope: str,
    n_simulations: int = 20000,
    seed: int = 20260403,
    match_window: int = 10,
    model_name: str | None = None,
    data: dict | None = None,
) -> FoldResult:
    runner_kwargs = {
        "holdout_year": fold.holdout_year,
        "match_window": match_window,
        "simulations": n_simulations,
        "seed": seed,
        "training_scope": training_scope,
    }
    if data is not None:
        runner_kwargs["data"] = data
    backtest = model_runner(
        **runner_kwargs,
    )
    return fold_result_from_backtest(
        fold,
        model_name or getattr(model_runner, "__name__", "model"),
        training_scope,
        backtest,
        n_simulations,
    )


def validate_all_folds(
    models: dict[str, Callable[..., dict[str, object]] | dict[str, object]],
    scopes: list[str],
    data: dict,
    folds: list | None = None,
    n_simulations: int = 20000,
    seed: int = 20260403,
    match_window: int = 10,
) -> dict[str, object]:
    if folds is None:
        folds = build_all_folds(data)
    fold_results: list[FoldResult] = []
    for fold in folds:
        for model_name, model_config in models.items():
            if isinstance(model_config, dict):
                model_runner = model_config["runner"]
                model_scopes = list(model_config.get("scopes", scopes))
            else:
                model_runner = model_config
                model_scopes = scopes
            for scope in model_scopes:
                fold_results.append(
                    validate_fold(
                        fold,
                        model_runner,
                        scope,
                        n_simulations=n_simulations,
                        seed=seed,
                        match_window=match_window,
                        model_name=model_name,
                        data=data,
                    )
                )
    results_df = pd.DataFrame([result.to_row() for result in fold_results])
    return {
        "fold_results": fold_results,
        "fold_results_df": results_df,
        "aggregate_df": aggregate_across_folds(results_df) if not results_df.empty else pd.DataFrame(),
    }


def aggregate_across_folds(results_df: pd.DataFrame) -> pd.DataFrame:
    numeric = ["log_loss", "brier", "top1_acc_pct", "r16_hits", "sf_hits"]
    grouped = results_df.groupby(["model", "scope"], dropna=False)
    aggregate = grouped[numeric].agg(["mean", "std"]).round(4)
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    champion_hits = grouped["champion_hit"].apply(lambda values: (values == "Yes").sum()).rename("champion_hits")
    return aggregate.join(champion_hits).reset_index()


def _match_predictions_frame(fr: FoldResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_year": fr.fold_year,
                "model": fr.model_name,
                "scope": fr.training_scope,
                "match_id": prediction.match_id,
                "home_team_id": prediction.home_team_id,
                "away_team_id": prediction.away_team_id,
                "home_win_prob": prediction.predicted_home_win,
                "draw_prob": prediction.predicted_draw,
                "away_win_prob": prediction.predicted_away_win,
                "actual_outcome": prediction.actual_outcome,
                "stage": prediction.stage,
            }
            for prediction in fr.match_predictions
        ]
    )


def save_fold_result(fr: FoldResult, out_dir: str | Path = "data/processed/validation") -> dict[str, Path]:
    """Write one fold result to JSON and match-prediction CSV files."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{fr.model_name}_{fr.training_scope}_{fr.fold_year}".replace("/", "_").replace(" ", "_")
    json_path = output_dir / f"{stem}_fold_result.json"
    predictions_path = output_dir / f"{stem}_match_predictions.csv"
    payload = {
        "summary": fr.to_row(),
        "actual": {
            "r16_teams": sorted(fr.actual_r16_teams),
            "sf_teams": sorted(fr.actual_sf_teams),
            "champion": fr.actual_champion,
        },
        "simulated": {
            "r16_teams": sorted(fr.simulated_r16_teams),
            "sf_teams": sorted(fr.simulated_sf_teams),
            "champion": fr.simulated_champion,
        },
        "metadata": fr.metadata,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _match_predictions_frame(fr).to_csv(predictions_path, index=False)
    return {"json": json_path, "match_predictions": predictions_path}
