from __future__ import annotations

import json
import inspect

import pandas as pd

from apps.dashboard import pages
from apps.dashboard.simulation_store import (
    ArtifactSettings,
    artifact_dir,
    artifact_key,
    load_artifact,
    save_artifact,
)
from scripts import prewarm_dashboard_simulations


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
        artifact_key(make_settings(model_version="v4")),
        artifact_key(make_settings(data_build_date="2026-05-27")),
        artifact_key(make_settings(simulations=100_000)),
        artifact_key(make_settings(match_window=12)),
        artifact_key(make_settings(training_scope="world_cup_only")),
        artifact_key(make_settings(seed=17)),
    }
    assert artifact_key(settings) not in changed_keys
    assert len(changed_keys) == 6


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


def test_probability_pages_are_wired_to_artifact_loader() -> None:
    page_text = inspect.getsource(pages.render_v3_probabilities_dashboard)
    v4_page_text = inspect.getsource(pages.render_v4_probabilities_dashboard)

    assert "load_or_run_probability_artifact" in page_text
    assert "build_v3_probability_artifact" in page_text
    assert "load_or_run_probability_artifact" in v4_page_text
    assert "build_v4_probability_artifact" in v4_page_text
    assert "artifact_metadata" in pages.render_v4_probabilities_dashboard.__code__.co_varnames


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
