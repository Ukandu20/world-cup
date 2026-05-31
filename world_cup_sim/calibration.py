from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    holdout_year: int
    target: str
    bin_count: int
    bins: list[dict[str, float | int]]
    brier_score: float
    ece: float
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _target_vectors(predictions: pd.DataFrame, target: str) -> tuple[pd.Series, pd.Series]:
    normalized = str(target).strip().lower()
    if normalized in {"home_win", "team_win"}:
        return predictions["home_win_prob"].astype(float), predictions["actual_outcome"].astype(str).eq("home_win").astype(float)
    if normalized == "draw":
        return predictions["draw_prob"].astype(float), predictions["actual_outcome"].astype(str).eq("draw").astype(float)
    if normalized == "away_win":
        return predictions["away_win_prob"].astype(float), predictions["actual_outcome"].astype(str).eq("away_win").astype(float)
    if normalized == "champion":
        actual = predictions["actual_champion"]
        if actual.dtype == bool:
            actual_values = actual.astype(float)
        else:
            actual_values = actual.astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)
        return predictions["champion_prob"].astype(float) / 100.0, actual_values
    raise ValueError(f"Unsupported calibration target: {target}")


def compute_calibration(
    predictions: pd.DataFrame,
    target: str,
    *,
    model: str,
    holdout_year: int,
    bin_count: int = 10,
) -> CalibrationResult:
    """Compute fixed-width calibration bins, Brier score, and expected calibration error."""
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    if predictions.empty:
        return CalibrationResult(
            model=str(model),
            holdout_year=int(holdout_year),
            target=str(target),
            bin_count=int(bin_count),
            bins=[],
            brier_score=0.0,
            ece=0.0,
            sample_count=0,
        )

    probabilities, actuals = _target_vectors(predictions, target)
    frame = pd.DataFrame(
        {
            "probability": pd.to_numeric(probabilities, errors="coerce").clip(0.0, 1.0),
            "actual": pd.to_numeric(actuals, errors="coerce").clip(0.0, 1.0),
        }
    ).dropna()
    if frame.empty:
        sample_count = 0
        brier_score = 0.0
    else:
        sample_count = int(len(frame))
        brier_score = float(np.mean((frame["probability"] - frame["actual"]) ** 2))

    edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(int(bin_count)):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == int(bin_count) - 1:
            mask = frame["probability"].ge(lower) & frame["probability"].le(upper)
        else:
            mask = frame["probability"].ge(lower) & frame["probability"].lt(upper)
        subset = frame.loc[mask]
        count = int(len(subset))
        predicted_mean = float(subset["probability"].mean()) if count else 0.0
        observed_rate = float(subset["actual"].mean()) if count else 0.0
        absolute_error = abs(predicted_mean - observed_rate) if count else 0.0
        if sample_count:
            ece += (count / sample_count) * absolute_error
        bins.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "predicted_mean": predicted_mean,
                "observed_rate": observed_rate,
                "absolute_error": absolute_error,
            }
        )

    return CalibrationResult(
        model=str(model),
        holdout_year=int(holdout_year),
        target=str(target),
        bin_count=int(bin_count),
        bins=bins,
        brier_score=brier_score,
        ece=float(ece),
        sample_count=sample_count,
    )


__all__ = ["CalibrationResult", "compute_calibration"]
