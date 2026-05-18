from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_cup_sim.analysis import (  # noqa: E402
    build_2026_implication_tables,
    build_correlation_metrics,
    build_data_quality_summary,
    build_goal_metrics,
    build_host_effect_metrics,
    build_participation_metrics,
    build_winner_followup_metrics,
    load_historical_world_cup_data,
)


def load_page_module(page_name: str):
    spec = importlib.util.spec_from_file_location(page_name, ROOT / "apps" / "pages" / page_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_historical_loader_and_quality_summary_return_expected_datasets():
    datasets = load_historical_world_cup_data()

    assert {"history", "teams", "placement", "results", "globe"}.issubset(datasets)
    assert {"edition", "country", "era"}.issubset(datasets["teams"].columns)
    assert {"edition", "country", "host_country", "era"}.issubset(datasets["results"].columns)

    quality = build_data_quality_summary(datasets)

    assert {"dataset", "rows", "columns", "duplicate_rows", "missing_cells", "missing_pct"}.issubset(
        quality.columns
    )
    assert set(quality["dataset"]) == set(datasets)


def test_participation_metrics_include_2026_and_stable_debutants():
    datasets = load_historical_world_cup_data()
    metrics = build_participation_metrics(datasets)

    participating = metrics["participating_teams"]
    assert {"edition", "team_counts", "tournament_count", "global_participation_pct"}.issubset(
        participating.columns
    )
    assert int(participating.loc[participating["edition"].eq(2026), "team_counts"].iloc[0]) == 48

    debutants = metrics["debutants_by_edition"]
    assert {"edition", "debutant_count"}.issubset(debutants.columns)
    assert debutants["edition"].is_unique
    assert pd.to_numeric(debutants["debutant_count"], errors="coerce").ge(0).all()


def test_goal_and_winner_followup_metrics_have_expected_columns():
    datasets = load_historical_world_cup_data()
    goal_metrics = build_goal_metrics(datasets)
    winners = build_winner_followup_metrics(datasets)

    assert {"edition", "total_goals", "total_matches", "goals_per_match"}.issubset(
        goal_metrics["tournament_goals"].columns
    )
    assert {"edition", "country", "gf", "ga", "goals_per_game"}.issubset(
        goal_metrics["team_goals"].columns
    )
    assert {"edition", "country", "next_edition", "next_placement"}.issubset(winners.columns)
    assert winners["country"].notna().all()


def test_host_effect_metrics_handle_multi_host_2026():
    datasets = load_historical_world_cup_data()
    metrics = build_host_effect_metrics(datasets)
    hosts = metrics["hosts"]

    assert {"edition", "country", "is_host", "next_placement", "gf", "ga"}.issubset(hosts.columns)
    hosts_2026 = set(hosts.loc[hosts["edition"].eq(2026), "team_id"])
    assert {"CAN", "MEX", "USA"}.issubset(hosts_2026)
    assert int(metrics["host_summary"].loc[0, "host_teams"]) >= 23


def test_correlation_metrics_default_lookback_returns_numeric_fields():
    datasets = load_historical_world_cup_data()
    metrics = build_correlation_metrics(datasets)

    summary = metrics["correlation_summary"]
    last_k_summary = metrics["last_k_summary"]

    assert {"feature", "correlation_with_finish_score", "rows"}.issubset(summary.columns)
    assert pd.api.types.is_numeric_dtype(summary["correlation_with_finish_score"])
    assert int(last_k_summary.loc[0, "lookback"]) == 5
    assert pd.api.types.is_numeric_dtype(last_k_summary["correlation_with_finish_score"])


def test_2026_implication_tables_include_distribution_and_qualified_teams():
    datasets = load_historical_world_cup_data()
    tables = build_2026_implication_tables(datasets)

    distribution = tables["confederation_distribution"]
    qualified = tables["qualified_teams"]

    assert {"confederation", "team_count", "edition"}.issubset(distribution.columns)
    assert int(distribution["team_count"].sum()) == 48
    assert {"team_id", "country", "confederation", "group_code"}.issubset(qualified.columns)
    assert len(qualified) == 48


def test_historical_eda_modules_import_and_notebook_references_shared_analysis():
    import apps.historical_eda as historical_eda

    assert callable(historical_eda.render_historical_eda_page)
    page = load_page_module("1_Analysis.py")
    assert callable(page.render_historical_eda_page)

    notebook = nbformat.read(ROOT / "main.ipynb", as_version=4)
    source = "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)
    assert "world_cup_sim.analysis" in source
