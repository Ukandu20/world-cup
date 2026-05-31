from __future__ import annotations

from datetime import date
import inspect

import pandas as pd
import pytest

from world_cup_sim.calibration import compute_calibration
from world_cup_sim.feature_snapshots import get_lead_in_form
from world_cup_sim.historical_tournaments import verify_loaders
from world_cup_sim.shared import build_actual_group_standings, build_historical_world_cup_backtest_data, build_2022_actual_group_standings
from world_cup_sim.validation import FoldResult, MatchPrediction, save_fold_result
from world_cup_sim.validation_data import load_all_data, validate_schema_integrity
from world_cup_sim.v2 import run_v2_historical_backtest
from world_cup_sim.v3 import run_v3_historical_backtest
from world_cup_sim.v4 import run_v4_historical_backtest


def test_load_all_data_returns_lead_in_and_placement_with_harmonized_columns():
    data = load_all_data()

    assert {"results", "schedule", "teams", "lead_in", "placement"}.issubset(data)
    assert {"tournament_id", "match_id", "stage", "status"}.issubset(data["lead_in"].columns)
    assert {"next_edition", "next_placement", "next_position"}.issubset(data["placement"].columns)


def test_schema_integrity_allows_future_columns_only_on_placement():
    data = load_all_data()
    validate_schema_integrity(data)

    invalid = {name: frame.copy() for name, frame in data.items()}
    invalid["lead_in"]["next_position"] = 1

    with pytest.raises(ValueError, match="future-looking"):
        validate_schema_integrity(invalid)


def test_lead_in_form_uses_only_rows_before_cutoff():
    lead_in = pd.DataFrame(
        [
            {
                "qualified_team_id": "AAA",
                "date": "2022-01-01",
                "team_score": 2,
                "opponent_score": 1,
                "goal_difference": 1,
                "team_elo_start": 1500,
                "opponent_elo_start": 1450,
                "result": "win",
                "tournament": "Friendly",
            },
            {
                "qualified_team_id": "AAA",
                "date": "2022-02-01",
                "team_score": 0,
                "opponent_score": 3,
                "goal_difference": -3,
                "team_elo_start": 1510,
                "opponent_elo_start": 1600,
                "result": "loss",
                "tournament": "Friendly",
            },
        ]
    )

    form = get_lead_in_form(["AAA"], date(2022, 1, 15), lead_in, k=10, time_decay_halflife=None)

    assert int(form.loc["AAA", "lead_in_match_count"]) == 1
    assert form.loc["AAA", "goals_for"] == 2.0
    assert form.loc["AAA", "goals_against"] == 1.0
    assert form.loc["AAA", "pre_tournament_elo"] == 1500.0


def test_historical_tournament_loaders_verify_recent_folds():
    data = load_all_data()

    verify_loaders(data)


def test_historical_backtest_data_uses_preloaded_core_tables(monkeypatch):
    data = load_all_data()

    def fail_read_csv(*args, **kwargs):
        raise AssertionError("unexpected per-fold read_csv")

    import world_cup_sim.shared as shared

    monkeypatch.setattr(shared.pd, "read_csv", fail_read_csv)
    dataset = build_historical_world_cup_backtest_data(2018, data=data)

    assert dataset["results_df"].shape[0] == 64
    assert dataset["base_df"].shape[0] == 32
    assert dataset["snapshot_features_df"].shape[0] == 32


def test_historical_backtest_wrappers_accept_preloaded_data_argument():
    assert "data" in inspect.signature(run_v2_historical_backtest).parameters
    assert "data" in inspect.signature(run_v3_historical_backtest).parameters
    assert "data" in inspect.signature(run_v4_historical_backtest).parameters


def test_validation_training_paths_do_not_use_legacy_results_reads(monkeypatch):
    import world_cup_sim.v2 as v2_module
    import world_cup_sim.v3 as v3_module
    import world_cup_sim.v4 as v4_module
    from world_cup_sim.constants import V2_FEATURE_COLUMNS, V3_FEATURE_COLUMNS, V4_FEATURE_COLUMNS

    def fail_read_csv(*args, **kwargs):
        raise AssertionError("validation training path used pd.read_csv")

    def tiny_training_frame(feature_columns):
        rows = []
        for index in range(4):
            row = {
                "date": pd.Timestamp(f"2017-01-{index + 1:02d}"),
                "edition": 2017,
                "stage": "Friendly",
                "stage_bucket": "group",
                "home_team": f"H{index}",
                "away_team": f"A{index}",
                "tournament": "Friendly",
                "home_score": index % 3,
                "away_score": (index + 1) % 3,
                "outcome_label": "home_win" if index % 3 > (index + 1) % 3 else "away_win",
                "sample_weight": 1.0,
                "training_scope": "all_international_since_anchor",
                "anchor_year": 1998,
                "anchor_date": "1998-06-10",
            }
            for column in feature_columns:
                row[column] = float(index + 1)
            rows.append(row)
        return pd.DataFrame(rows)

    monkeypatch.setattr(v2_module.pd, "read_csv", fail_read_csv)
    monkeypatch.setattr(v2_module, "lead_in_to_match_level_results", lambda frame: pd.DataFrame())
    monkeypatch.setattr(v2_module, "build_snapshot_training_frame", lambda *args, **kwargs: tiny_training_frame(V2_FEATURE_COLUMNS))
    v2_frame = v2_module.build_v2_training_frame(
        training_scope="all_international_since_anchor",
        data={"lead_in": pd.DataFrame(), "results": pd.DataFrame(), "teams": pd.DataFrame()},
    )
    assert not v2_frame.empty

    monkeypatch.setattr(v3_module.pd, "read_csv", fail_read_csv)
    monkeypatch.setattr(v3_module, "build_v3_training_frame", lambda *args, **kwargs: tiny_training_frame(V3_FEATURE_COLUMNS))
    v3_bundle = v3_module.fit_v3_poisson_models(data={"lead_in": pd.DataFrame(), "results": pd.DataFrame(), "teams": pd.DataFrame()})
    assert len(v3_bundle["training_frame"]) == 4

    monkeypatch.setattr(v4_module.pd, "read_csv", fail_read_csv)
    monkeypatch.setattr(v4_module, "build_v4_training_frame", lambda *args, **kwargs: tiny_training_frame(V4_FEATURE_COLUMNS))
    monkeypatch.setattr(v4_module, "compute_v4_stage_multipliers", lambda cutoff_year=2026: {})
    v4_bundle = v4_module.fit_v4_poisson_models(data={"lead_in": pd.DataFrame(), "results": pd.DataFrame(), "teams": pd.DataFrame()})
    assert len(v4_bundle["training_frame"]) == 4


def test_generalized_actual_group_standings_keeps_2022_wrapper_compatible():
    results = pd.DataFrame(
        [
            {"stage": "Group Stage", "match_number": 1, "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0},
            {"stage": "Group Stage", "match_number": 2, "home_team": "C", "away_team": "D", "home_score": 1, "away_score": 1},
            {"stage": "Group Stage", "match_number": 3, "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0},
            {"stage": "Group Stage", "match_number": 4, "home_team": "B", "away_team": "D", "home_score": 3, "away_score": 0},
            {"stage": "Group Stage", "match_number": 5, "home_team": "A", "away_team": "D", "home_score": 2, "away_score": 2},
            {"stage": "Group Stage", "match_number": 6, "home_team": "B", "away_team": "C", "home_score": 1, "away_score": 0},
        ]
    )
    lookup = {"A": "A", "B": "A", "C": "A", "D": "A"}
    features = pd.DataFrame(
        [
            {"team_id": "A", "display_name": "A", "team_strength": 4.0},
            {"team_id": "B", "display_name": "B", "team_strength": 3.0},
            {"team_id": "C", "display_name": "C", "team_strength": 2.0},
            {"team_id": "D", "display_name": "D", "team_strength": 1.0},
        ]
    )

    generalized = build_actual_group_standings(results, lookup, features, group_order=["A"])
    wrapped = build_2022_actual_group_standings(results, lookup, features)

    assert generalized[["team_id", "actual_group_rank"]].to_dict("records") == wrapped[
        ["team_id", "actual_group_rank"]
    ].to_dict("records")
    assert generalized.iloc[0]["team_id"] == "A"


def test_build_validation_artifacts_accepts_non_2022_holdout_with_stubbed_runners(monkeypatch):
    import scripts.build_model_validation as validation_script

    def fake_model_backtest(*args, **kwargs):
        return {
            "summary_metrics": {
                "multiclass_log_loss": 1.0,
                "multiclass_brier_score": 0.5,
                "top1_match_accuracy": 50.0,
                "draw_rate_actual": 25.0,
                "draw_rate_predicted": 25.0,
                "round_of_16_hit_count": 8,
                "semifinal_hit_count": 2,
                "exact_champion_hit": 0,
                "predicted_champion_team_id": "AAA",
                "actual_champion_team_id": "BBB",
            },
            "match_predictions": pd.DataFrame(
                [
                    {
                        "match_number": 1,
                        "stage": "Group Stage",
                        "home_team": "A",
                        "away_team": "B",
                        "home_score": 1,
                        "away_score": 0,
                        "home_win_prob": 0.5,
                        "draw_prob": 0.25,
                        "away_win_prob": 0.25,
                        "predicted_outcome": "home_win",
                        "actual_outcome": "home_win",
                        "top1_correct": True,
                    }
                ]
            ),
            "team_backtest_table": pd.DataFrame(
                [
                    {
                        "team_id": "AAA",
                        "display_name": "A",
                        "group_code": "A",
                        "r16_prob": 50.0,
                        "qf_prob": 25.0,
                        "sf_prob": 10.0,
                        "final_prob": 5.0,
                        "champion_prob": 1.0,
                    }
                ]
            ),
            "training_metadata": {
                "training_scope": "world_cup_only",
                "anchor_year": 1998,
                "anchor_date": "1998-06-10",
                "training_start_date": "1998-06-10",
                "training_end_date": "2018-06-13",
                "training_match_count": 10,
                "sample_weight_policy": "test",
            },
        }

    def fake_baseline(*args, **kwargs):
        backtest = fake_model_backtest(*args, **kwargs)
        backtest["feature_columns"] = ["elo_diff"]
        backtest["tournament_simulated"] = False
        return backtest

    monkeypatch.setattr(validation_script, "run_elo_baseline_historical", fake_baseline)
    monkeypatch.setattr(validation_script, "run_v2_historical_backtest", fake_model_backtest)
    monkeypatch.setattr(validation_script, "run_v3_historical_backtest", fake_model_backtest)
    monkeypatch.setattr(validation_script, "run_v4_historical_backtest", fake_model_backtest)

    artifacts = validation_script.build_validation_artifacts(holdout_year=2018, simulations=1)

    assert artifacts["validation_window"]["holdout_year"] == 2018
    assert artifacts["validation_window"]["holdout"] == "2018 FIFA World Cup"


def test_calibration_bins_and_scores_are_deterministic():
    predictions = pd.DataFrame(
        {
            "home_win_prob": [0.8, 0.2],
            "draw_prob": [0.1, 0.3],
            "away_win_prob": [0.1, 0.5],
            "actual_outcome": ["home_win", "away_win"],
        }
    )

    result = compute_calibration(predictions, "home_win", model="test", holdout_year=2022)

    assert result.sample_count == 2
    assert len(result.bins) == 10
    assert result.brier_score == pytest.approx(0.04)
    assert result.ece >= 0.0


def test_save_fold_result_writes_json_and_predictions(tmp_path):
    fold_result = FoldResult(
        fold_year=2022,
        model_name="test_model",
        training_scope="test_scope",
        n_training_matches=10,
        n_holdout_matches=1,
        match_predictions=[
            MatchPrediction(
                match_id="1",
                home_team_id="A",
                away_team_id="B",
                predicted_home_win=0.5,
                predicted_draw=0.25,
                predicted_away_win=0.25,
                actual_outcome="home_win",
                stage="Group Stage",
            )
        ],
        simulated_r16_teams={"A"},
        simulated_sf_teams={"A"},
        simulated_champion="A",
        actual_r16_teams={"A"},
        actual_sf_teams={"A"},
        actual_champion="A",
        n_simulations=1,
    )

    paths = save_fold_result(fold_result, tmp_path)

    assert paths["json"].exists()
    assert paths["match_predictions"].exists()
