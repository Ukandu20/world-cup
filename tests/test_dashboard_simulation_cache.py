from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.dashboard import pages
from apps.dashboard import data as dashboard_data
from apps.dashboard.artifact_formatting import format_artifact_timestamp
from apps.dashboard.config import TRAINING_SCOPE_ALL_INTERNATIONAL, TRAINING_SCOPE_WORLD_CUP_ONLY
from apps.dashboard.model_registry import MODEL_REGISTRY, PRIMARY_MODEL_ID
from apps.dashboard.simulation_store import (
    ArtifactSettings,
    artifact_dir,
    artifact_key,
    load_artifact,
    load_official_artifact,
    save_artifact,
)
from scripts import prewarm_dashboard_simulations
from apps import home
from apps import team_report_card


def make_settings(**overrides) -> ArtifactSettings:
    values = {
        "model_id": "v3",
        "model_version": "v3",
        "data_build_date": "2026-05-26",
        "simulations": 20_000,
        "match_window": 10,
        "training_scope": "all_international_since_anchor",
        "seed": 20260403,
        "bracket_head_to_head_simulations": 10_000,
    }
    values.update(overrides)
    return ArtifactSettings(**values)


def test_artifact_key_is_stable_and_sensitive() -> None:
    settings = make_settings()
    assert artifact_key(settings) == artifact_key(make_settings())

    changed_keys = {
        artifact_key(make_settings(model_id="v4")),
        artifact_key(make_settings(model_version="v4")),
        artifact_key(make_settings(data_build_date="2026-05-27")),
        artifact_key(make_settings(simulations=100_000)),
        artifact_key(make_settings(match_window=12)),
        artifact_key(make_settings(training_scope="world_cup_only")),
        artifact_key(make_settings(seed=17)),
    }
    assert artifact_key(settings) not in changed_keys
    assert len(changed_keys) == 7


def test_model_registry_marks_v4_primary_and_legacy_models_secondary() -> None:
    assert PRIMARY_MODEL_ID == "v4"
    assert MODEL_REGISTRY["v4"].is_primary is True
    assert MODEL_REGISTRY["v4"].default_training_scope == TRAINING_SCOPE_WORLD_CUP_ONLY
    assert MODEL_REGISTRY["v4"].artifact_builder_name == "build_v4_probability_artifact"
    assert MODEL_REGISTRY["v1"].supports_official_artifact is False
    assert MODEL_REGISTRY["v2"].artifact_builder_name == "build_v2_probability_artifact"
    assert MODEL_REGISTRY["v3"].default_training_scope == TRAINING_SCOPE_WORLD_CUP_ONLY
    assert MODEL_REGISTRY["v3"].is_primary is False


def test_home_dashboard_uses_world_cup_only_primary_projection_copy() -> None:
    source = inspect.getsource(pages.render_home_page)

    assert "training_scope = PRIMARY_MODEL.default_training_scope" in source
    assert "V4 Primary | {artifact_scope_display}" in source
    assert "historical World Cup finals only" in source
    assert "artifact_scope_display" in source
    assert "Training scope:" in source
    assert "V4 multi-fold validation" in source


def test_probability_pages_migrate_stale_all_international_state() -> None:
    source = inspect.getsource(pages.render_v4_probabilities_dashboard)
    helper_source = inspect.getsource(pages.migrate_default_training_scope)

    assert "migrate_default_training_scope" in source
    assert "st.session_state.pop(widget_key, None)" in helper_source
    assert "TRAINING_SCOPE_ALL_INTERNATIONAL" in helper_source
    assert TRAINING_SCOPE_ALL_INTERNATIONAL == "all_international_since_anchor"
    assert TRAINING_SCOPE_WORLD_CUP_ONLY == MODEL_REGISTRY["v4"].default_training_scope


def test_official_v4_world_cup_only_artifact_metadata_is_committed() -> None:
    metadata_path = (
        ROOT
        / "data"
        / "processed"
        / "dashboard_simulations"
        / "official"
        / "v4"
        / "f48403bd87c0f74265ee0e46"
        / "metadata.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["model_id"] == "v4"
    assert metadata["model_version"] == "v4"
    assert metadata["simulations"] == 20_000
    assert metadata["match_window"] == 10
    assert metadata["training_scope"] == TRAINING_SCOPE_WORLD_CUP_ONLY
    assert metadata["seed"] == 20260403


def test_save_and_load_artifact_round_trip(tmp_path, monkeypatch) -> None:
    import apps.dashboard.simulation_store as store

    monkeypatch.setattr(store, "OFFICIAL_ARTIFACT_ROOT", tmp_path / "official")
    monkeypatch.setattr(store, "RUNTIME_ARTIFACT_ROOT", tmp_path / "runtime")
    settings = make_settings(model_id="v4", model_version="v4")
    dashboard_df = pd.DataFrame(
        [
            {"team_id": "ARG", "prob_1": 0.6, "champion_prob": 0.2},
            {"team_id": "BRA", "prob_1": 0.4, "champion_prob": 0.1},
        ]
    )
    bracket_data = {"rounds": [{"round_code": "F", "matches": [{"winner_team_id": "ARG"}]}]}
    metadata = {
        "alpha": 0.1,
        "rho": -0.05,
        "time_decay_halflife_days": 1095,
        "stage_multipliers": {"F": 1.1},
        "alpha_source": "cv",
        "rho_source": "fit",
    }

    saved = save_artifact(settings, dashboard_df, bracket_data, metadata, tier="official")
    loaded = load_artifact(settings)

    assert saved.source == "official"
    assert loaded.artifact is not None
    assert loaded.artifact.source == "official"
    assert loaded.artifact.dashboard_df.loc[:, ["team_id", "prob_1"]].to_dict("records") == [
        {"team_id": "ARG", "prob_1": 0.6},
        {"team_id": "BRA", "prob_1": 0.4},
    ]
    assert loaded.artifact.bracket_data == bracket_data
    assert loaded.artifact.metadata["rho"] == -0.05
    assert loaded.artifact.metadata["stage_multipliers"] == {"F": 1.1}


def test_save_artifact_uses_unique_temp_directory_and_cleans_up(tmp_path, monkeypatch) -> None:
    import apps.dashboard.simulation_store as store

    monkeypatch.setattr(store, "OFFICIAL_ARTIFACT_ROOT", tmp_path / "official")
    monkeypatch.setattr(store, "RUNTIME_ARTIFACT_ROOT", tmp_path / "runtime")
    settings = make_settings()
    dashboard_df = pd.DataFrame([{"team_id": "ARG", "prob_1": 1.0}])

    first = save_artifact(settings, dashboard_df, {"rounds": []}, {"run": 1}, tier="runtime")
    second = save_artifact(settings, dashboard_df, {"rounds": []}, {"run": 2}, tier="runtime")

    assert first.artifact_dir == second.artifact_dir
    assert second.metadata["run"] == 2
    leftovers = list(second.artifact_dir.parent.glob(f".{second.artifact_dir.name}*.tmp"))
    assert leftovers == []


def test_load_artifact_prefers_runtime_and_ignores_corrupt_official(tmp_path, monkeypatch) -> None:
    import apps.dashboard.simulation_store as store

    monkeypatch.setattr(store, "OFFICIAL_ARTIFACT_ROOT", tmp_path / "official")
    monkeypatch.setattr(store, "RUNTIME_ARTIFACT_ROOT", tmp_path / "runtime")
    settings = make_settings()
    official_dir = artifact_dir(settings, "official")
    official_dir.mkdir(parents=True)
    (official_dir / "metadata.json").write_text(json.dumps({"created_at_utc": "bad"}), encoding="utf-8")

    runtime_df = pd.DataFrame([{"team_id": "ARG", "prob_1": 1.0}])
    save_artifact(settings, runtime_df, {"rounds": []}, {"created_by": "test"}, tier="runtime")
    loaded = load_artifact(settings)

    assert loaded.artifact is not None
    assert loaded.artifact.source == "runtime"
    assert loaded.warnings == ()
    assert loaded.artifact.dashboard_df.iloc[0]["team_id"] == "ARG"


def test_load_official_artifact_ignores_runtime_cache(tmp_path, monkeypatch) -> None:
    import apps.dashboard.simulation_store as store

    monkeypatch.setattr(store, "OFFICIAL_ARTIFACT_ROOT", tmp_path / "official")
    monkeypatch.setattr(store, "RUNTIME_ARTIFACT_ROOT", tmp_path / "runtime")
    settings = make_settings(model_id="v4", model_version="v4")
    save_artifact(
        settings,
        pd.DataFrame([{"team_id": "RUN", "prob_1": 1.0}]),
        {"rounds": []},
        {"created_by": "runtime"},
        tier="runtime",
    )
    save_artifact(
        settings,
        pd.DataFrame([{"team_id": "OFF", "prob_1": 1.0}]),
        {"rounds": []},
        {"created_by": "official"},
        tier="official",
    )

    loaded = load_official_artifact(settings)

    assert loaded.artifact is not None
    assert loaded.artifact.source == "official"
    assert loaded.artifact.dashboard_df.iloc[0]["team_id"] == "OFF"


def test_probability_pages_are_wired_to_artifact_loader() -> None:
    v2_page_text = inspect.getsource(pages.render_v2_probabilities_dashboard)
    page_text = inspect.getsource(pages.render_v3_probabilities_dashboard)
    v4_page_text = inspect.getsource(pages.render_v4_probabilities_dashboard)

    assert "load_or_run_probability_artifact" in v2_page_text
    assert "build_v2_probability_artifact" in v2_page_text
    assert "load_or_run_probability_artifact" in page_text
    assert "build_v3_probability_artifact" in page_text
    assert "load_or_run_probability_artifact" in v4_page_text
    assert "build_v4_probability_artifact" in v4_page_text
    assert "artifact_metadata" in pages.render_v4_probabilities_dashboard.__code__.co_varnames


def test_report_cards_use_primary_v4_artifact_path() -> None:
    settings_source = inspect.getsource(team_report_card.official_report_card_settings)
    dataset_source = inspect.getsource(team_report_card.build_report_card_dataset)
    page_source = inspect.getsource(team_report_card.render_team_report_card_page)

    assert "ArtifactSettings" in settings_source
    assert "default_simulation_settings" in settings_source
    assert "load_official_artifact" in dataset_source
    assert "load_or_create_artifact" not in dataset_source
    assert "build_v4_probability_artifact" not in dataset_source
    assert "PRIMARY_MODEL" in settings_source
    assert "official Enhanced Poisson Model projections" in page_source
    assert "render_filter_bar" not in page_source
    assert "home.V3_MODEL_VERSION" not in page_source


def test_report_card_artifact_updated_at_formatter() -> None:
    assert (
        team_report_card.format_artifact_updated_at("2026-05-26T06:10:37Z")
        == "last updated on 2026-05-26 @ 06:10am"
    )
    assert (
        team_report_card.format_artifact_updated_at("2026-05-26T18:45:02Z")
        == "last updated on 2026-05-26 @ 06:45pm"
    )
    assert team_report_card.format_artifact_updated_at(None) == "last updated time unavailable"
    assert team_report_card.format_artifact_updated_at("not-a-date") == "last updated time unavailable"


def test_shared_artifact_timestamp_formatter() -> None:
    assert format_artifact_timestamp("2026-05-26T06:10:37Z") == "2026-05-26 @ 06:10am"
    assert format_artifact_timestamp("2026-05-26T18:45:02Z") == "2026-05-26 @ 06:45pm"
    assert format_artifact_timestamp(None) == "time unavailable"
    assert format_artifact_timestamp("not-a-date") == "time unavailable"


def test_display_artifact_status_uses_shared_timestamp_format(monkeypatch) -> None:
    captions: list[str] = []
    monkeypatch.setattr(pages.st, "caption", captions.append)

    cached_result = SimpleNamespace(
        warnings=(),
        created=False,
        artifact=SimpleNamespace(source="official", created_at_utc="2026-05-26T18:45:02Z"),
    )
    pages.display_artifact_status(cached_result, "V4")

    fresh_result = SimpleNamespace(
        warnings=(),
        created=True,
        artifact=SimpleNamespace(source="runtime", created_at_utc="2026-05-26T06:10:37Z"),
    )
    pages.display_artifact_status(fresh_result, "V4")

    assert captions == [
        "Using official cached V4 simulation run from 2026-05-26 @ 06:45pm.",
        "Fresh runtime V4 simulation run saved at 2026-05-26 @ 06:10am.",
    ]
    assert all("T" not in caption and "Z" not in caption for caption in captions)


def test_prewarm_model_writes_official_artifact_without_real_simulation(tmp_path, monkeypatch) -> None:
    import apps.dashboard.simulation_store as store

    monkeypatch.setattr(store, "OFFICIAL_ARTIFACT_ROOT", tmp_path / "official")
    monkeypatch.setattr(store, "RUNTIME_ARTIFACT_ROOT", tmp_path / "runtime")

    def fake_build(*args, **kwargs):
        return {
            "dashboard_df": pd.DataFrame([{"team_id": "ARG", "prob_1": 1.0}]),
            "bracket_data": {"rounds": []},
            "metadata": {"model_id": "v3", "model_version": "v3"},
        }

    monkeypatch.setattr(prewarm_dashboard_simulations, "build_v3_probability_artifact", fake_build)
    path = prewarm_dashboard_simulations.prewarm_model(
        "v3",
        base_df=pd.DataFrame(),
        fixtures_df=pd.DataFrame(),
        lead_in_df=pd.DataFrame(),
        data_build_date="2026-05-26",
        simulations=12,
        match_window=10,
        training_scope="all_international_since_anchor",
        force=True,
    )

    assert (path / "probabilities.csv.gz").exists()
    assert (path / "bracket.json").exists()
    assert (path / "metadata.json").exists()


def test_prewarm_default_selection_is_primary_only(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prewarm_dashboard_simulations.py"])
    args = prewarm_dashboard_simulations.parse_args()
    assert PRIMARY_MODEL_ID == "v4"
    assert args.model is None
    assert args.include_legacy is False


def test_validation_artifact_loader_exposes_committed_multi_fold_rows() -> None:
    artifacts = dashboard_data.load_validation_artifacts()

    assert artifacts["available"] is True
    assert len(artifacts["per_fold_rows"]) == 21
    assert len(artifacts["aggregate_rows"]) == 7
    assert len(artifacts["aggregate_model_rows"]) == 7
    assert not artifacts["aggregate_display"].empty
    assert not artifacts["per_fold_display"].empty
    assert not artifacts["calibration_display"].empty


def test_multi_fold_backtest_page_uses_validation_artifacts_not_live_rolling_runner() -> None:
    page_source = inspect.getsource(pages.render_v4_rolling_backtest_dashboard)

    assert "load_validation_artifacts" in page_source
    assert "run_v4_rolling_backtest_dashboard" not in page_source
    assert "Multi-Fold Backtest Findings" in page_source


def test_navigation_exposes_multi_fold_backtests_label() -> None:
    navigation_source = inspect.getsource(home.build_navigation_pages)

    assert "Multi-Fold Backtests" in navigation_source
    assert "V2 2022 Drilldown" in navigation_source
    assert "V3 2022 Drilldown" in navigation_source
    assert "V4 2022 Drilldown" in navigation_source
    assert "V4 Rolling Backtest" not in navigation_source


def test_validation_headline_findings_match_model_card_summary() -> None:
    artifacts = dashboard_data.load_validation_artifacts()
    findings = artifacts["headline_findings"]

    assert findings["best_log_loss"] == "V4 World Cup only"
    assert findings["best_brier"] == "Elo-only baseline"
    assert findings["best_top1"] == "V3 all international since anchor"
    assert findings["dominance"] == "No model consistently dominates all headline metrics."


def test_missing_validation_artifact_returns_warning_state(tmp_path) -> None:
    missing_path = tmp_path / "aggregate_validation.json"
    artifacts = dashboard_data.load_validation_artifacts(str(missing_path))

    assert artifacts["available"] is False
    assert str(missing_path) in artifacts["warning"]
    assert "python scripts/run_multi_fold_validation.py" in artifacts["warning"]
    assert artifacts["aggregate_rows"].empty
    assert artifacts["per_fold_rows"].empty
