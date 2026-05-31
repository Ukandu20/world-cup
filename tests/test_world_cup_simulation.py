from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import world_cup_sim.v4 as v4_module  # noqa: E402
import apps.dashboard.export as dashboard_export  # noqa: E402

from world_cup_simulation import (  # noqa: E402
    THIRD_PLACE_ROUTING_MAP,
    SAMPLE_WEIGHT_POLICY,
    TRAINING_SCOPE_ALL_INTERNATIONAL,
    TRAINING_SCOPE_WORLD_CUP_ONLY,
    V2_PREVIOUS_EDITION_LOOKBACK,
    V3_MATCH_START_YEAR,
    WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT,
    build_2022_backtest_data,
    build_deterministic_bracket,
    build_deterministic_bracket_v2,
    build_deterministic_bracket_v2_32team,
    build_deterministic_bracket_v3,
    build_v4_score_matrix,
    build_v2_match_feature_table,
    build_v2_team_strengths,
    build_v2_training_frame,
    build_v3_team_feature_table,
    build_v3_training_frame,
    compute_quadratic_form_snapshot,
    dixon_coles_tau,
    build_weighted_form_table,
    build_team_strengths,
    build_recent_form_metrics,
    classify_competition_importance,
    compute_elo_expected_score,
    extract_group_stage_fixtures,
    fit_v2_match_multinomial_model,
    fit_v3_poisson_models,
    get_modal_group_rankings,
    resolve_training_anchor_date,
    resolve_training_anchor_year,
    normalize_weight_pair,
    predict_match_probabilities_v2,
    predict_knockout_matchup,
    predict_knockout_matchup_v2,
    predict_match_lambdas_v3,
    predict_knockout_matchup_v3,
    quadratic_recency_weights,
    rank_best_third_place_teams,
    rank_group_standings,
    run_v2_backtest_2022,
    run_v3_2022_backtest,
    simulate_group_probabilities,
    simulate_group_probabilities_v2_32team,
    simulate_group_probabilities_v2,
    simulate_group_probabilities_v3,
    strength_weighted_penalty_probability,
)
from world_cup_sim.constants import WORLD_CUP_ROOT  # noqa: E402
from world_cup_sim.constants import INTERNATIONAL_RESULTS_PATH  # noqa: E402
from world_cup_sim.shared import build_historical_world_cup_backtest_data, validation_artifact_filenames  # noqa: E402
from world_cup_sim.v4 import compute_v4_stage_multipliers, fit_v4_dixon_coles_rho, v4_stage_key  # noqa: E402
from scripts.build_world_cup_2026_dataset import (  # noqa: E402
    QualifiedTeam,
    build_alias_maps,
    compute_world_cup_history_features,
    compute_world_cup_placement_score,
)
from scripts.build_model_validation import (  # noqa: E402
    METRIC_FIELDS,
    build_model_card_markdown,
    build_validation_artifacts,
    run_elo_baseline_2022,
)

DATA_DIR = WORLD_CUP_ROOT / "2026"
WORLD_CUP_EDITIONS = (
    1930,
    1934,
    1938,
    1950,
    1954,
    1958,
    1962,
    1966,
    1970,
    1974,
    1978,
    1982,
    1986,
    1990,
    1994,
    1998,
    2002,
    2006,
    2010,
    2014,
    2018,
    2022,
    2026,
)


def load_home_module():
    spec = importlib.util.spec_from_file_location("world_cup_home", ROOT / "apps" / "home.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_page_module(page_name: str):
    spec = importlib.util.spec_from_file_location(page_name, ROOT / "apps" / "pages" / page_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_team_report_card_module():
    spec = importlib.util.spec_from_file_location("team_report_card", ROOT / "apps" / "team_report_card.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_historical_eda_module():
    spec = importlib.util.spec_from_file_location("historical_eda", ROOT / "apps" / "historical_eda.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.query_params = {}
        self.session_state = {}

    def cache_data(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    cache_resource = cache_data

    def selectbox(self, label, options, index=0, key=None, format_func=None, on_change=None):
        selected = list(options)[index]
        if key is not None:
            self.session_state[key] = selected
        return selected

    def caption(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None

        return noop


def load_team_selection_module(monkeypatch, streamlit_stub: StreamlitStub):
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    spec = importlib.util.spec_from_file_location("team_selection_test", ROOT / "apps" / "team_selection.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.st = streamlit_stub
    return module


def test_global_team_state_prefers_query_param(monkeypatch):
    streamlit_stub = StreamlitStub()
    streamlit_stub.query_params["team"] = "BRA"
    streamlit_stub.session_state["global_team_id"] = "ARG"
    team_selection = load_team_selection_module(monkeypatch, streamlit_stub)

    selected = team_selection.get_global_team_id(["ARG", "BRA"], "ARG")

    assert selected == "BRA"
    assert streamlit_stub.session_state["global_team_id"] == "BRA"


def test_global_team_state_uses_session_then_default(monkeypatch):
    streamlit_stub = StreamlitStub()
    streamlit_stub.session_state["global_team_id"] = "ARG"
    team_selection = load_team_selection_module(monkeypatch, streamlit_stub)

    assert team_selection.get_global_team_id(["ARG", "BRA"], "BRA") == "ARG"
    streamlit_stub.session_state["global_team_id"] = "XXX"
    assert team_selection.get_global_team_id(["ARG", "BRA"], "BRA") == "BRA"
    assert streamlit_stub.session_state["global_team_id"] == "BRA"


def test_set_global_team_id_updates_session_and_query(monkeypatch):
    streamlit_stub = StreamlitStub()
    team_selection = load_team_selection_module(monkeypatch, streamlit_stub)

    team_selection.set_global_team_id("CZE")

    assert streamlit_stub.session_state["global_team_id"] == "CZE"
    assert streamlit_stub.query_params["team"] == "CZE"


def test_resolve_team_country_name_uses_aliases(monkeypatch):
    streamlit_stub = StreamlitStub()
    team_selection = load_team_selection_module(monkeypatch, streamlit_stub)
    team_row = pd.Series({"team_id": "CZE", "display_name": "Czechia", "canonical_name": "Czechia"})

    assert team_selection.resolve_team_country_name(
        team_row,
        ["Brazil", "Czechoslovakia"],
        aliases={"Czechia": "Czechoslovakia"},
    ) == "Czechoslovakia"


def test_historical_eda_team_country_lookup_and_missing_winner_default(monkeypatch):
    streamlit_stub = StreamlitStub()
    streamlit_stub.session_state["global_team_id"] = "CZE"
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    historical_eda = load_historical_eda_module()
    historical_eda.st = streamlit_stub
    historical_eda.team_selection.st = streamlit_stub
    base_df = pd.DataFrame(
        [
            {"team_id": "CZE", "display_name": "Czechia", "canonical_name": "Czechia", "tournament_name": "Czechia"},
            {"team_id": "BRA", "display_name": "Brazil", "canonical_name": "Brazil", "tournament_name": "Brazil"},
        ]
    )

    lookup = historical_eda.build_team_country_lookup(base_df, ["Brazil", "Czechoslovakia"])

    assert lookup["CZE"] == "Czechoslovakia"
    assert historical_eda.global_country_default(["Brazil"], base_df, "Brazil") == "Brazil"
    assert streamlit_stub.session_state["global_team_id"] == "CZE"


def test_processed_world_cup_dataset_has_normalized_ids_and_metadata():
    for year in WORLD_CUP_EDITIONS:
        edition_dir = WORLD_CUP_ROOT / str(year)
        assert (edition_dir / "teams.csv").exists()
        assert (edition_dir / "elo.csv").exists()

        teams_df = pd.read_csv(edition_dir / "teams.csv")
        elo_df = pd.read_csv(edition_dir / "elo.csv")
        assert {"tournament_id", "year", "team_id", "confederation"}.issubset(teams_df.columns)
        assert {"tournament_id", "year", "team_id", "confederation", "elo_start", "elo_end"}.issubset(elo_df.columns)
        assert teams_df["team_id"].astype(str).str.len().gt(0).all()
        assert teams_df["confederation"].astype(str).str.len().gt(0).all()

    for filename in ("teams.csv", "squads.csv", "placement.csv", "results.csv", "elo.csv"):
        assert (WORLD_CUP_ROOT / "all_editions" / filename).exists()

    for year in WORLD_CUP_EDITIONS[:-1]:
        results_df = pd.read_csv(WORLD_CUP_ROOT / str(year) / "results.csv")
        expected_prefix = f"WC-{year}_"
        assert {"tournament_id", "match_id", "team_id", "opponent_id", "team_confederation", "opponent_confederation"}.issubset(results_df.columns)
        assert results_df["match_id"].str.match(rf"^WC-{year}_\d{{3}}$").all()
        assert results_df["match_id"].str.startswith(expected_prefix).all()
        assert results_df.groupby("match_id").size().eq(2).all()
        assert results_df["team_id"].astype(str).str.len().gt(0).all()
        assert results_df["team_confederation"].astype(str).str.len().gt(0).all()

    fixtures_df = pd.read_csv(WORLD_CUP_ROOT / "2026" / "fixtures.csv")
    assert {"tournament_id", "match_id", "source_match_id", "home_team_confederation", "away_team_confederation"}.issubset(fixtures_df.columns)
    assert fixtures_df["match_id"].str.match(r"^WC-2026_\d{3}$").all()
    assert fixtures_df["source_match_id"].astype(str).str.len().gt(0).all()


def test_build_recent_form_metrics_uses_last_8_matches_only():
    lead_in_df = pd.DataFrame(
        [
            {
                "lead_in_id": f"lead_in_{index:03d}",
                "qualified_team_id": "AAA",
                "date": f"2026-01-{index + 1:02d}",
                "goal_difference": goal_difference,
                "result": result,
            }
            for index, (result, goal_difference) in enumerate(
                [
                    ("win", 4),
                    ("win", 3),
                    ("draw", 0),
                    ("draw", 0),
                    ("loss", -1),
                    ("loss", -1),
                    ("loss", -1),
                    ("loss", -1),
                    ("loss", -1),
                    ("loss", -1),
                ]
            )
        ]
        + [
            {
                "lead_in_id": f"lead_in_b_{index:03d}",
                "qualified_team_id": "BBB",
                "date": f"2026-01-{index + 1:02d}",
                "goal_difference": 1,
                "result": "win",
            }
            for index in range(10)
        ]
    )

    form_df = build_recent_form_metrics(lead_in_df, match_window=8).set_index("qualified_team_id")

    assert form_df.loc["AAA", "recent_matches"] == 8
    assert form_df.loc["AAA", "points_per_match"] == 0.25
    assert form_df.loc["AAA", "goal_diff_per_match"] == -0.75


def test_normalize_weight_pair_scales_to_one():
    normalized = normalize_weight_pair(65, 35)

    assert normalized == (0.65, 0.35)


def test_compute_world_cup_placement_score_respects_dnq_bounds_and_shape():
    assert compute_world_cup_placement_score(rank=None, n_teams=32, qualified=False) == 0.0
    assert compute_world_cup_placement_score(rank=1, n_teams=32, qualified=True) == 1.0
    assert compute_world_cup_placement_score(rank=32, n_teams=32, qualified=True) == 0.05
    assert compute_world_cup_placement_score(rank=2, n_teams=32, qualified=True) > compute_world_cup_placement_score(rank=4, n_teams=32, qualified=True)
    assert compute_world_cup_placement_score(rank=2, n_teams=48, qualified=True) > compute_world_cup_placement_score(rank=2, n_teams=16, qualified=True)


def test_compute_world_cup_history_features_maps_west_germany_and_uses_unique_editions():
    qualified_teams = {
        "GER": QualifiedTeam(team_id="GER", fifa_code="GER", tournament_name="Germany", canonical_name="Germany", group_code="A"),
        "USA": QualifiedTeam(team_id="USA", fifa_code="USA", tournament_name="United States", canonical_name="United States", group_code="B"),
    }
    alias_map, dated_former_aliases = build_alias_maps(qualified_teams, [])

    history_features = compute_world_cup_history_features(qualified_teams, alias_map, dated_former_aliases)

    placement_df = pd.read_csv(WORLD_CUP_ROOT / "all_editions" / "placement.csv")
    history_df = pd.read_csv(WORLD_CUP_ROOT / "fifa_world_cup_history.csv")
    editions = sorted(history_df["Year"].astype(int).tolist())
    edition_weight_map = {edition: (index + 1) ** 2 for index, edition in enumerate(editions)}
    total_edition_weight = float(sum(edition_weight_map.values()))

    germany_rows = placement_df[placement_df["country"].isin(["Germany", "West Germany"])]
    germany_positions = {
        int(row.edition): int(row.position)
        for row in germany_rows.drop_duplicates(subset=["edition"], keep="first").itertuples(index=False)
    }
    expected_weighted_participations = float(sum(edition_weight_map[edition] for edition in germany_positions))
    expected_weighted_placement = sum(
        edition_weight_map[edition] * compute_world_cup_placement_score(
            rank=germany_positions.get(edition),
            n_teams=max(
                int(history_df.loc[history_df["Year"] == edition, "Teams"].iloc[0]),
                germany_positions.get(edition, 0),
            ),
            qualified=edition in germany_positions,
        )
        for edition in editions
    ) / total_edition_weight

    assert history_features["Germany"]["world_cup_participations"] == len(germany_positions) + 1
    assert history_features["Germany"]["weighted_world_cup_participations"] == expected_weighted_participations
    assert abs(history_features["Germany"]["weighted_world_cup_placement_score"] - expected_weighted_placement) < 1e-12
    assert history_features["United States"]["world_cup_participations"] >= 1


def test_default_simulation_settings_include_form_window():
    home = load_home_module()

    defaults = home.default_simulation_settings()

    assert defaults == {
        "simulation_label": "20k",
        "form_match_window": 10,
        "v2_results_weight": 40,
        "v2_gd_weight": 25,
        "v2_perf_weight": 25,
        "v2_elo_delta_weight": 10,
    }


def test_load_data_preserves_weighted_world_cup_history_columns():
    home = load_home_module()

    base_df, _, _, _ = home.load_data()

    assert "weighted_world_cup_participations" in base_df.columns
    assert "weighted_world_cup_placement_score" in base_df.columns


def test_build_weighted_form_table_uses_linear_recency_weights_and_confederation():
    base_df = pd.DataFrame(
        [
            {
                "team_id": "AAA",
                "display_name": "Alpha",
                "flag_icon_code": "aa",
                "group_code": "A",
                "confederation": "UEFA",
                "elo_rating": 1900,
                "world_rank": 5,
            },
            {
                "team_id": "BBB",
                "display_name": "Beta",
                "flag_icon_code": "bb",
                "group_code": "B",
                "confederation": "CAF",
                "elo_rating": 1750,
                "world_rank": 18,
            },
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "a1", "qualified_team_id": "AAA", "date": "2026-01-01", "team_score": 0, "opponent_score": 1, "result": "L", "team_elo_start": 1680, "opponent_elo_start": 1690, "team_elo_delta": -8},
            {"lead_in_id": "a2", "qualified_team_id": "AAA", "date": "2026-01-02", "team_score": 1, "opponent_score": 1, "result": "D", "team_elo_start": 1700, "opponent_elo_start": 1800, "team_elo_delta": 4},
            {"lead_in_id": "a3", "qualified_team_id": "AAA", "date": "2026-01-03", "team_score": 2, "opponent_score": 0, "result": "W", "team_elo_start": 1750, "opponent_elo_start": 1900, "team_elo_delta": 10},
            {"lead_in_id": "b1", "qualified_team_id": "BBB", "date": "2026-01-01", "team_score": 1, "opponent_score": 0, "result": "win", "team_elo_start": 1650, "opponent_elo_start": 1600, "team_elo_delta": 6},
            {"lead_in_id": "b2", "qualified_team_id": "BBB", "date": "2026-01-02", "team_score": 0, "opponent_score": 0, "result": "draw", "team_elo_start": 1660, "opponent_elo_start": 1610, "team_elo_delta": 1},
            {"lead_in_id": "b3", "qualified_team_id": "BBB", "date": "2026-01-03", "team_score": 0, "opponent_score": 2, "result": "loss", "team_elo_start": 1670, "opponent_elo_start": 1620, "team_elo_delta": -5},
        ]
    )

    form_df = build_weighted_form_table(base_df, lead_in_df, match_window=2).set_index("team_id")
    aaa_expected_perf = (
        (0.5 - compute_elo_expected_score(1700, 1800)) * 1
        + (1.0 - compute_elo_expected_score(1750, 1900)) * 2
    ) / 3
    bbb_expected_perf = (
        (0.5 - compute_elo_expected_score(1660, 1610)) * 1
        + (0.0 - compute_elo_expected_score(1670, 1620)) * 2
    ) / 3

    assert form_df.index.tolist() == ["AAA", "BBB"]
    assert form_df.loc["AAA", "confederation"] == "UEFA"
    assert form_df.loc["AAA", "wins"] == 1
    assert form_df.loc["AAA", "draws"] == 1
    assert form_df.loc["AAA", "losses"] == 0
    assert form_df.loc["AAA", "goals_for"] == 3
    assert form_df.loc["AAA", "goals_against"] == 1
    assert form_df.loc["AAA", "avg_opp_elo"] == 1866.7
    assert form_df.loc["AAA", "avg_elo_gap"] == 133.3
    assert abs(form_df.loc["AAA", "results_form"] - 0.833) < 1e-9
    assert abs(form_df.loc["AAA", "gd_form"] - 1.333) < 1e-9
    assert abs(form_df.loc["AAA", "expected_score"] - 0.318) < 1e-9
    assert abs(form_df.loc["AAA", "perf_vs_exp"] - round(float(aaa_expected_perf), 3)) < 1e-9
    assert form_df.loc["AAA", "elo_delta_form"] == 8.0
    assert form_df.loc["AAA", "difficulty"] == 133.333
    assert form_df.loc["AAA", "results_form_z"] == 1.0
    assert form_df.loc["AAA", "gd_form_z"] == 1.0
    assert form_df.loc["AAA", "perf_vs_exp_z"] == 1.0
    assert form_df.loc["AAA", "elo_delta_form_z"] == 1.0
    assert form_df.loc["AAA", "results_score"] == 0.8333
    assert form_df.loc["AAA", "gd_score"] == 0.6667
    assert form_df.loc["AAA", "perf_score"] == 1.0
    assert form_df.loc["AAA", "elo_score"] == 0.7667
    assert form_df.loc["AAA", "form_index_0to1"] == 0.8267
    assert form_df.loc["AAA", "form"] == 8.44
    assert form_df.loc["BBB", "perf_score"] == 0.0952
    assert form_df.loc["BBB", "form_index_0to1"] == 0.2138
    assert form_df.loc["BBB", "form"] == 2.9242
    assert form_df.loc["AAA", "schedule_difficulty"] == 5.0
    assert form_df.loc["BBB", "schedule_difficulty"] == 1.0
    assert abs(form_df.loc["BBB", "perf_vs_exp"] - round(float(bbb_expected_perf), 3)) < 1e-9
    assert form_df.loc["BBB", "elo_delta_form"] == -3.0


def test_report_card_grade_bands_and_scores_are_bounded():
    report_card = load_team_report_card_module()

    sample = pd.Series([10, 20, 30], index=["low", "mid", "high"], dtype=float)
    scores = report_card.series_to_report_scores(sample)

    assert scores.loc["low"] == 1.0
    assert scores.loc["mid"] == 5.5
    assert scores.loc["high"] == 10.0
    assert report_card.score_to_grade(9.5) == "A+"
    assert report_card.score_to_grade(8.8) == "A"
    assert report_card.score_to_grade(7.5) == "B"
    assert report_card.score_to_grade(6.0) == "C"
    assert report_card.score_to_grade(4.5) == "D"
    assert report_card.score_to_grade(4.4) == "F"
    assert report_card.chart_title("Qualification Goals", "Brazil") == "Brazil's FIFA Men's World Cup Qualification Goals"
    assert report_card.chart_title("Brazil's Goals Scored", "Brazil") == "Brazil's FIFA Men's World Cup Goals Scored"
    assert (
        report_card.chart_title("FIFA Men's World Cup Team Profile Radar", "Brazil")
        == "Brazil's FIFA Men's World Cup Team Profile Radar"
    )


def test_build_best_finish_lookup_maps_historical_aliases():
    home = load_home_module()
    report_card = load_team_report_card_module()

    base_df, _, _, _ = home.load_data()
    lookup = report_card.build_best_finish_lookup(base_df)

    assert lookup["GER"] == "Winner"
    assert lookup["USA"] == "Third Place"


def test_build_recent_matches_table_limits_to_latest_10_and_sorts_newest_first():
    report_card = load_team_report_card_module()
    lead_in_df = pd.DataFrame(
        [
            {
                "lead_in_id": f"lead_{index:02d}",
                "date": f"2026-03-{index + 1:02d}",
                "qualified_team_id": "AAA",
                "opponent_name": f"Opp {index:02d}",
                "team_score": 2 if index % 3 == 0 else 1,
                "opponent_score": 0 if index % 2 == 0 else 1,
                "team_elo_start": 1800 + index,
                "opponent_elo_start": 1750 + index,
                "team_elo_delta": 5 - index * 0.1,
                "result": "win" if index % 3 == 0 else "draw",
                "tournament": "Friendly",
            }
            for index in range(12)
        ]
    )

    recent = report_card.build_recent_matches_table(lead_in_df, "AAA", match_window=10)

    assert len(recent) == 10
    assert recent.iloc[0]["Date"] == "2026-03-12"
    assert recent.iloc[-1]["Date"] == "2026-03-03"
    assert {"Date", "Opponent", "Competition", "Result", "Score", "Elo Change", "Performance Score", "Grade"}.issubset(recent.columns)
    assert recent["Performance Score"].between(1.0, 10.0).all()
    assert recent["Grade"].isin({"A+", "A", "B", "C", "D", "F"}).all()


def test_build_qualification_path_table_filters_and_derives_cycle_metrics():
    report_card = load_team_report_card_module()
    lead_in_df = pd.DataFrame(
        [
            {
                "lead_in_id": "old",
                "date": "2021-10-01",
                "qualified_team_id": "AAA",
                "opponent_name": "Old Opp",
                "team_score": 4,
                "opponent_score": 0,
                "team_elo_start": 1700,
                "opponent_elo_start": 1600,
                "team_elo_delta": 5,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Old City",
                "country": "Old Country",
            },
            {
                "lead_in_id": "friendly",
                "date": "2024-01-01",
                "qualified_team_id": "AAA",
                "opponent_name": "Friendly Opp",
                "team_score": 1,
                "opponent_score": 0,
                "team_elo_start": 1710,
                "opponent_elo_start": 1660,
                "team_elo_delta": 2,
                "result": "win",
                "tournament": "Friendly",
                "city": "Town",
                "country": "Land",
            },
            {
                "lead_in_id": "q1",
                "date": "2024-03-01",
                "qualified_team_id": "AAA",
                "opponent_name": "Beta",
                "team_score": 2,
                "opponent_score": 1,
                "team_elo_start": 1720,
                "opponent_elo_start": 1680,
                "team_elo_delta": 7.5,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Alpha City",
                "country": "Alpha Land",
            },
            {
                "lead_in_id": "q2",
                "date": "2024-03-05",
                "qualified_team_id": "AAA",
                "opponent_name": "Gamma",
                "team_score": 0,
                "opponent_score": 0,
                "team_elo_start": 1727.5,
                "opponent_elo_start": 1750,
                "team_elo_delta": 1.0,
                "result": "draw",
                "tournament": "FIFA World Cup inter-confederation playoff",
                "city": "Neutral City",
                "country": "Neutral Land",
            },
            {
                "lead_in_id": "q3",
                "date": "2026-03-25",
                "qualified_team_id": "AAA",
                "opponent_name": "Delta",
                "team_score": 1,
                "opponent_score": 0,
                "team_elo_start": 1728.5,
                "opponent_elo_start": 1650,
                "team_elo_delta": 4.0,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Qualifier City",
                "country": "Qualifier Land",
            },
            {
                "lead_in_id": "q4",
                "date": "2026-03-31",
                "qualified_team_id": "AAA",
                "opponent_name": "Epsilon",
                "team_score": 2,
                "opponent_score": 1,
                "team_elo_start": 1732.5,
                "opponent_elo_start": 1690,
                "team_elo_delta": 5.0,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Playoff City",
                "country": "Playoff Land",
            },
            {
                "lead_in_id": "other_team",
                "date": "2024-03-08",
                "qualified_team_id": "BBB",
                "opponent_name": "Alpha",
                "team_score": 3,
                "opponent_score": 0,
                "team_elo_start": 1600,
                "opponent_elo_start": 1500,
                "team_elo_delta": 6,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Other",
                "country": "Other",
            },
        ]
    )

    path = report_card.build_qualification_path_table(lead_in_df, "AAA")

    assert path["lead_in_id"].tolist() == ["q1", "q2", "q3", "q4"]
    assert path["Opponent"].tolist() == ["Beta", "Gamma", "Delta", "Epsilon"]
    assert path["qualification_stage"].tolist() == ["Qualifiers", "Playoffs", "Qualifiers", "Playoffs"]
    assert path["Result"].tolist() == ["W", "D", "W", "W"]
    assert path["points"].tolist() == [3, 1, 3, 3]
    assert path["cumulative_points"].tolist() == [3, 4, 7, 10]
    assert path["goal_difference"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert path["cumulative_goal_difference"].tolist() == [1.0, 1.0, 2.0, 3.0]
    assert path["Score"].tolist() == ["2-1", "0-0", "1-0", "2-1"]
    assert path["team_elo_delta"].tolist() == [7.5, 1.0, 4.0, 5.0]
    assert path["post_match_elo"].tolist() == [1727.5, 1728.5, 1732.5, 1737.5]


def test_build_qualification_path_table_returns_empty_frame_without_qualifiers():
    report_card = load_team_report_card_module()
    lead_in_df = pd.DataFrame(
        [
            {
                "lead_in_id": "friendly",
                "date": "2024-01-01",
                "qualified_team_id": "AAA",
                "opponent_name": "Friendly Opp",
                "team_score": 1,
                "opponent_score": 0,
                "team_elo_start": 1710,
                "opponent_elo_start": 1660,
                "team_elo_delta": 2,
                "result": "win",
                "tournament": "Friendly",
                "city": "Town",
                "country": "Land",
            }
        ]
    )

    path = report_card.build_qualification_path_table(lead_in_df, "AAA")

    assert path.empty
    assert {"Date", "Opponent", "Competition", "points", "cumulative_points"}.issubset(path.columns)


def test_build_knockout_path_table_handles_projected_exit_and_progression():
    report_card = load_team_report_card_module()
    display_lookup = {"AAA": "Alpha", "BBB": "Beta", "CCC": "Gamma"}
    bracket_data = {
        "rounds": [
            {
                "round_code": "R32",
                "round_label": "Round of 32",
                "matches": [
                    {
                        "home_team_id": "AAA",
                        "away_team_id": "BBB",
                        "winner_team_id": "AAA",
                        "home_win_prob": 62.5,
                        "away_win_prob": 37.5,
                        "round_label": "Round of 32",
                    }
                ],
            },
            {
                "round_code": "R16",
                "round_label": "Round of 16",
                "matches": [
                    {
                        "home_team_id": "AAA",
                        "away_team_id": "CCC",
                        "winner_team_id": "CCC",
                        "home_win_prob": 41.0,
                        "away_win_prob": 59.0,
                        "round_label": "Round of 16",
                    }
                ],
            },
        ]
    }

    alpha_path = report_card.build_knockout_path_table(bracket_data, "AAA", display_lookup)
    beta_path = report_card.build_knockout_path_table(bracket_data, "BBB", display_lookup)

    assert alpha_path.to_dict("records") == [
        {
            "Stage": "Round of 32",
            "Opponent": "Beta",
            "Matchup Win %": 62.5,
            "Projected Winner": "Alpha",
        },
        {
            "Stage": "Round of 16",
            "Opponent": "Gamma",
            "Matchup Win %": 41.0,
            "Projected Winner": "Gamma",
        },
    ]
    assert beta_path.to_dict("records") == [
        {
            "Stage": "Round of 32",
            "Opponent": "Alpha",
            "Matchup Win %": 37.5,
            "Projected Winner": "Alpha",
        }
    ]


def test_build_identity_rows_marks_pending_fields_cleanly():
    report_card = load_team_report_card_module()
    team_row = pd.Series(
        {
            "confederation": "UEFA",
            "group_code": "B",
            "world_rank": 7,
            "elo_rating": 1892,
            "world_cup_participations": 12,
        }
    )
    history_summary = {
        "appearances": "12",
        "best_finish": "Winner",
        "best_finish_years": [1998, 2018],
        "latest_world_cup": "2022",
        "latest_finish": "Runner-up",
        "goals_scored_per_game": "2.10",
        "goals_conceded_per_game": "0.90",
    }

    rows = report_card.build_identity_rows(team_row, history_summary)
    by_label = {row["label"]: row["value"] for row in rows}

    assert list(by_label) == [
        "FIFA Ranking",
        "Elo Rating",
        "World Cup Appearances",
        "Best Finish",
        "Latest World Cup",
        "Latest Finish",
        "Goals Scored/Game",
        "Goals Conceded/Game",
    ]
    assert "Confederation" not in by_label
    assert "Group" not in by_label
    assert "Coach" not in by_label
    assert "Captain" not in by_label
    assert by_label["FIFA Ranking"] == "7"
    assert by_label["Elo Rating"] == "1892"
    assert by_label["Best Finish"] == "Winner"
    assert rows[3]["value_html"] == 'Winner<span class="trc-fact-subscript">[1998, 2018]</span>'

    debut_row = pd.Series(
        {
            "confederation": "CAF",
            "group_code": "A",
            "world_rank": 61,
            "elo_rating": 1540,
            "world_cup_participations": 1,
        }
    )
    debut_summary = report_card.build_header_history_summary(pd.DataFrame(), debut_row)
    debut_rows = report_card.build_identity_rows(debut_row, debut_summary)
    debut_by_label = {row["label"]: row["value"] for row in debut_rows}

    assert report_card.is_debut_tournament(debut_row)
    assert debut_by_label["World Cup Appearances"] == "1"
    assert debut_by_label["Best Finish"] == "Debut tournament"


def test_build_header_history_summary_formats_best_finish_and_goal_rates():
    report_card = load_team_report_card_module()
    history = pd.DataFrame(
        [
            {
                "edition": 1998,
                "position": 1,
                "placement_label": "Winner",
                "goals_for": 15,
                "goals_against": 2,
                "matches_played": 7,
            },
            {
                "edition": 2018,
                "position": 1,
                "placement_label": "Winner",
                "goals_for": 14,
                "goals_against": 6,
                "matches_played": 7,
            },
            {
                "edition": 2022,
                "position": 2,
                "placement_label": "Runner-up",
                "goals_for": 16,
                "goals_against": 8,
                "matches_played": 7,
            },
        ]
    )
    summary = report_card.build_header_history_summary(history, pd.Series({"world_cup_participations": 12}))

    assert summary["appearances"] == "12"
    assert summary["best_finish"] == "Winner"
    assert summary["best_finish_years"] == [1998, 2018]
    assert summary["latest_world_cup"] == "2022"
    assert summary["latest_finish"] == "Runner-up"
    assert summary["goals_scored_per_game"] == "2.14"
    assert summary["goals_conceded_per_game"] == "0.76"
    assert (
        report_card.format_best_finish_value_html(summary["best_finish"], summary["best_finish_years"])
        == 'Winner<span class="trc-fact-subscript">[1998, 2018]</span>'
    )


def test_build_qualifier_performance_tables_filters_hosts_and_scores_metrics():
    historical_eda = load_historical_eda_module()
    teams_df = pd.DataFrame(
        [
            {
                "team_id": "AAA",
                "team": "Alpha",
                "confederation": "UEFA",
                "qualification_path": "Group winner",
                "is_host": False,
            },
            {
                "team_id": "BBB",
                "team": "Beta",
                "confederation": "CAF",
                "qualification_path": "Group winner",
                "is_host": False,
            },
            {
                "team_id": "HST",
                "team": "Host",
                "confederation": "CONCACAF",
                "qualification_path": "Host nation",
                "is_host": True,
            },
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {
                "lead_in_id": "old_qualifier",
                "date": "2022-12-18",
                "qualified_team_id": "AAA",
                "opponent_name": "Old Opp",
                "team_score": 9,
                "opponent_score": 0,
                "team_elo_delta": 90,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Old City",
                "country": "Old Land",
            },
            {
                "lead_in_id": "friendly",
                "date": "2024-01-01",
                "qualified_team_id": "AAA",
                "opponent_name": "Friendly Opp",
                "team_score": 5,
                "opponent_score": 0,
                "team_elo_delta": 50,
                "result": "win",
                "tournament": "Friendly",
                "city": "Friendly City",
                "country": "Friendly Land",
            },
            {
                "lead_in_id": "aaa_1",
                "date": "2024-01-02",
                "qualified_team_id": "AAA",
                "opponent_name": "Gamma",
                "team_score": 3,
                "opponent_score": 1,
                "team_elo_delta": 8,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Alpha City",
                "country": "Alpha Land",
            },
            {
                "lead_in_id": "aaa_2",
                "date": "2026-03-31",
                "qualified_team_id": "AAA",
                "opponent_name": "Delta",
                "team_score": 2,
                "opponent_score": 1,
                "team_elo_delta": 2,
                "result": "draw",
                "tournament": "FIFA World Cup qualification",
                "city": "Playoff City",
                "country": "Playoff Land",
            },
            {
                "lead_in_id": "bbb_1",
                "date": "2024-01-03",
                "qualified_team_id": "BBB",
                "opponent_name": "Epsilon",
                "team_score": 1,
                "opponent_score": 0,
                "team_elo_delta": 1,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Beta City",
                "country": "Beta Land",
            },
            {
                "lead_in_id": "bbb_2",
                "date": "2024-01-04",
                "qualified_team_id": "BBB",
                "opponent_name": "Zeta",
                "team_score": 1,
                "opponent_score": 2,
                "team_elo_delta": -5,
                "result": "loss",
                "tournament": "FIFA World Cup qualification",
                "city": "Beta City",
                "country": "Beta Land",
            },
            {
                "lead_in_id": "host_qualifier",
                "date": "2024-01-05",
                "qualified_team_id": "HST",
                "opponent_name": "Host Opp",
                "team_score": 4,
                "opponent_score": 0,
                "team_elo_delta": 40,
                "result": "win",
                "tournament": "FIFA World Cup qualification",
                "city": "Host City",
                "country": "Host Land",
            },
        ]
    )

    outputs = historical_eda.build_qualifier_performance_tables(lead_in_df, teams_df)
    summary = outputs["summary"].set_index("team_id")
    matches = outputs["matches"]

    assert summary.index.tolist() == ["AAA", "BBB"]
    assert summary.loc["AAA", "matches"] == 2
    assert summary.loc["AAA", "wins"] == 1
    assert summary.loc["AAA", "draws"] == 1
    assert summary.loc["AAA", "losses"] == 0
    assert summary.loc["AAA", "points"] == 4
    assert summary.loc["AAA", "points_per_match"] == 2.0
    assert summary.loc["AAA", "goals_for"] == 5
    assert summary.loc["AAA", "goals_against"] == 2
    assert summary.loc["AAA", "goal_difference"] == 3
    assert summary.loc["AAA", "goal_difference_per_match"] == 1.5
    assert summary.loc["AAA", "elo_change"] == 10
    assert summary.loc["AAA", "elo_change_per_match"] == 5.0
    assert summary.loc["AAA", "performance_score"] == 92.5
    assert summary.loc["BBB", "performance_score"] == 7.5
    assert len(matches) == 4
    assert set(matches["team_id"]) == {"AAA", "BBB"}
    assert pd.to_datetime(matches["date"]).min() >= pd.Timestamp("2022-12-19")
    assert matches.loc[matches["Date"].eq("2026-03-31"), "Stage"].iloc[0] == "Qualifier playoffs"


def test_build_model_reason_bullets_uses_team_id_index_not_range_index():
    report_card = load_team_report_card_module()
    full_df = pd.DataFrame(
        [
            {
                "team_id": "CZE",
                "elo_rating": 1800,
                "results_form": 0.7,
                "gd_form": 1.2,
                "history_metric": 0.2,
                "goals_for": 15,
                "host_flag": 0,
            },
            {
                "team_id": "BRA",
                "elo_rating": 2000,
                "results_form": 0.9,
                "gd_form": 1.8,
                "history_metric": 0.9,
                "goals_for": 20,
                "host_flag": 0,
            },
            {
                "team_id": "USA",
                "elo_rating": 1900,
                "results_form": 0.6,
                "gd_form": 0.8,
                "history_metric": 0.5,
                "goals_for": 12,
                "host_flag": 1,
            },
        ]
    )
    team_row = pd.Series({"team_id": "CZE", "host_flag": 0})

    bullets = report_card.build_model_reason_bullets(team_row, full_df)

    assert len(bullets) == 3
    assert all(isinstance(item, str) and item for item in bullets)


def test_build_weighted_form_table_uses_neutral_schedule_difficulty_when_constant():
    base_df = pd.DataFrame(
        [
            {"team_id": "AAA", "display_name": "Alpha", "flag_icon_code": "", "group_code": "A", "confederation": "UEFA", "elo_rating": 1800, "world_rank": 8},
            {"team_id": "BBB", "display_name": "Beta", "flag_icon_code": "", "group_code": "B", "confederation": "CAF", "elo_rating": 1700, "world_rank": 12},
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "1", "qualified_team_id": "AAA", "date": "2026-01-01", "team_score": 1, "opponent_score": 0, "result": "win", "team_elo_start": 1700, "opponent_elo_start": 1750, "team_elo_delta": 8},
            {"lead_in_id": "2", "qualified_team_id": "BBB", "date": "2026-01-01", "team_score": 0, "opponent_score": 1, "result": "loss", "team_elo_start": 1650, "opponent_elo_start": 1700, "team_elo_delta": -8},
        ]
    )

    form_df = build_weighted_form_table(base_df, lead_in_df, match_window=1)

    assert form_df["schedule_difficulty"].tolist() == [3.0, 3.0]


def test_build_weighted_form_table_caps_goal_difference_and_uses_scoreline_when_result_missing():
    base_df = pd.DataFrame(
        [
            {"team_id": "AAA", "display_name": "Alpha", "flag_icon_code": "", "group_code": "A", "confederation": "UEFA", "elo_rating": 1800, "world_rank": 8},
            {"team_id": "BBB", "display_name": "Beta", "flag_icon_code": "", "group_code": "B", "confederation": "CAF", "elo_rating": 1700, "world_rank": 12},
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "1", "qualified_team_id": "AAA", "date": "2026-01-01", "team_score": 6, "opponent_score": 0, "result": None, "team_elo_start": 1700, "opponent_elo_start": 1600, "team_elo_delta": 12},
            {"lead_in_id": "2", "qualified_team_id": "BBB", "date": "2026-01-01", "team_score": 0, "opponent_score": 5, "result": None, "team_elo_start": 1650, "opponent_elo_start": 1700, "team_elo_delta": -11},
        ]
    )

    form_df = build_weighted_form_table(base_df, lead_in_df, match_window=1).set_index("team_id")

    assert form_df.loc["AAA", "wins"] == 1
    assert form_df.loc["AAA", "gd_form"] == 4.0
    assert form_df.loc["AAA", "elo_delta_form"] == 12.0
    assert form_df.loc["AAA", "gd_score"] == 1.0
    assert form_df.loc["AAA", "elo_score"] == 0.9
    assert form_df.loc["BBB", "losses"] == 1
    assert form_df.loc["BBB", "gd_form"] == -4.0
    assert form_df.loc["BBB", "elo_delta_form"] == -11.0
    assert form_df.loc["BBB", "gd_score"] == 0.0
    assert form_df.loc["BBB", "elo_score"] == 0.1333
    assert form_df["form"].between(1.0, 10.0).all()


def test_build_weighted_form_table_accepts_custom_composite_weights():
    base_df = pd.DataFrame(
        [
            {"team_id": "AAA", "display_name": "Alpha", "flag_icon_code": "", "group_code": "A", "confederation": "UEFA", "elo_rating": 1800, "world_rank": 8},
            {"team_id": "BBB", "display_name": "Beta", "flag_icon_code": "", "group_code": "B", "confederation": "CAF", "elo_rating": 1700, "world_rank": 12},
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "a1", "qualified_team_id": "AAA", "date": "2026-01-01", "team_score": 1, "opponent_score": 0, "result": "win", "team_elo_start": 1700, "opponent_elo_start": 1650, "team_elo_delta": 2},
            {"lead_in_id": "a2", "qualified_team_id": "AAA", "date": "2026-01-02", "team_score": 1, "opponent_score": 1, "result": "draw", "team_elo_start": 1710, "opponent_elo_start": 1670, "team_elo_delta": 1},
            {"lead_in_id": "b1", "qualified_team_id": "BBB", "date": "2026-01-01", "team_score": 0, "opponent_score": 1, "result": "loss", "team_elo_start": 1600, "opponent_elo_start": 1750, "team_elo_delta": 10},
            {"lead_in_id": "b2", "qualified_team_id": "BBB", "date": "2026-01-02", "team_score": 0, "opponent_score": 0, "result": "draw", "team_elo_start": 1610, "opponent_elo_start": 1760, "team_elo_delta": 9},
        ]
    )

    default_form_df = build_weighted_form_table(base_df, lead_in_df, match_window=2).set_index("team_id")
    elo_heavy_form_df = build_weighted_form_table(
        base_df,
        lead_in_df,
        match_window=2,
        composite_weights=(0, 0, 0, 100),
    ).set_index("team_id")

    assert default_form_df.loc["AAA", "form"] > default_form_df.loc["BBB", "form"]
    assert elo_heavy_form_df.loc["BBB", "form"] > elo_heavy_form_df.loc["AAA", "form"]


def test_build_v2_team_strengths_blends_rating_form_and_history():
    base_df = pd.DataFrame(
        [
            {
                "team_id": "AAA",
                "display_name": "Alpha",
                "flag_icon_code": "aa",
                "group_code": "A",
                "confederation": "UEFA",
                "elo_rating": 1900,
                "world_rank": 5,
                "fifa_points": 1800,
                "world_cup_participations": 10,
                "weighted_world_cup_participations": WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT / 2.0,
                "weighted_world_cup_placement_score": 0.9,
            },
            {
                "team_id": "BBB",
                "display_name": "Beta",
                "flag_icon_code": "bb",
                "group_code": "B",
                "confederation": "CAF",
                "elo_rating": 1700,
                "world_rank": 18,
                "fifa_points": 1500,
                "world_cup_participations": 2,
                "weighted_world_cup_participations": 0.0,
                "weighted_world_cup_placement_score": 0.1,
            },
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "a1", "qualified_team_id": "AAA", "date": "2026-01-01", "team_score": 2, "opponent_score": 0, "result": "win", "team_elo_start": 1800, "opponent_elo_start": 1700, "team_elo_delta": 10},
            {"lead_in_id": "b1", "qualified_team_id": "BBB", "date": "2026-01-01", "team_score": 0, "opponent_score": 1, "result": "loss", "team_elo_start": 1600, "opponent_elo_start": 1700, "team_elo_delta": -8},
        ]
    )

    v2_df = build_v2_team_strengths(
        base_df,
        lead_in_df,
        match_window=1,
        form_composite_weights=(100, 0, 0, 0),
    ).set_index("team_id")

    aaa_history_score = 0.7 * 0.9 + 0.3 * 0.5
    bbb_history_score = 0.7 * 0.1 + 0.3 * 0.0
    aaa_expected_index = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * aaa_history_score
    bbb_expected_index = 0.4 * 0.0 + 0.4 * 0.0 + 0.2 * bbb_history_score

    assert v2_df.loc["AAA", "rating_index_0to1"] == 1.0
    assert v2_df.loc["BBB", "rating_index_0to1"] == 0.0
    assert v2_df.loc["AAA", "form_index_0to1"] == 1.0
    assert v2_df.loc["BBB", "form_index_0to1"] == 0.0
    assert v2_df.loc["AAA", "weighted_world_cup_participation_ratio"] == 0.5
    assert v2_df.loc["AAA", "history_score"] == round(aaa_history_score, 4)
    assert v2_df.loc["BBB", "history_score"] == round(bbb_history_score, 4)
    assert v2_df.loc["AAA", "v2_strength_index_0to1"] == round(aaa_expected_index, 4)
    assert v2_df.loc["BBB", "v2_strength_index_0to1"] == round(bbb_expected_index, 4)
    assert v2_df.loc["AAA", "v2_strength"] == round(1.0 + 9.0 * aaa_expected_index, 4)
    assert v2_df.loc["AAA", "v2_strength"] > v2_df.loc["BBB", "v2_strength"]


def test_build_team_strengths_respects_custom_weight_pairs():
    base_df = pd.DataFrame(
        [
            {"team_id": "A", "group_code": "A", "elo_rating": 1900, "fifa_points": 1800},
            {"team_id": "B", "group_code": "A", "elo_rating": 1700, "fifa_points": 1500},
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "1", "qualified_team_id": "A", "date": "2026-01-01", "goal_difference": 3, "result": "win"},
            {"lead_in_id": "2", "qualified_team_id": "A", "date": "2026-01-02", "goal_difference": 1, "result": "draw"},
            {"lead_in_id": "3", "qualified_team_id": "B", "date": "2026-01-01", "goal_difference": -2, "result": "loss"},
            {"lead_in_id": "4", "qualified_team_id": "B", "date": "2026-01-02", "goal_difference": 0, "result": "draw"},
        ]
    )

    strengths_df = build_team_strengths(
        base_df,
        lead_in_df,
        baseline_rating_weights=(1.0, 0.0),
        form_component_weights=(1.0, 0.0),
        strength_blend_weights=(0.0, 1.0),
    ).set_index("team_id")

    assert strengths_df.loc["A", "rating_score"] > strengths_df.loc["B", "rating_score"]
    assert strengths_df.loc["A", "form_score"] > strengths_df.loc["B", "form_score"]
    assert strengths_df.loc["A", "team_strength"] == strengths_df.loc["A", "form_score"]
    assert strengths_df.loc["B", "team_strength"] == strengths_df.loc["B", "form_score"]


def test_build_team_strengths_respects_custom_recent_match_window():
    base_df = pd.DataFrame(
        [
            {"team_id": "A", "group_code": "A", "elo_rating": 1800, "fifa_points": 1700},
            {"team_id": "B", "group_code": "A", "elo_rating": 1800, "fifa_points": 1700},
        ]
    )
    lead_in_df = pd.DataFrame(
        [
            {"lead_in_id": "1", "qualified_team_id": "A", "date": "2026-01-01", "goal_difference": -3, "result": "loss"},
            {"lead_in_id": "2", "qualified_team_id": "A", "date": "2026-01-02", "goal_difference": 4, "result": "win"},
            {"lead_in_id": "3", "qualified_team_id": "B", "date": "2026-01-01", "goal_difference": 0, "result": "draw"},
            {"lead_in_id": "4", "qualified_team_id": "B", "date": "2026-01-02", "goal_difference": 0, "result": "draw"},
        ]
    )

    short_window = build_team_strengths(
        base_df,
        lead_in_df,
        match_window=1,
        baseline_rating_weights=(1.0, 0.0),
        form_component_weights=(1.0, 0.0),
        strength_blend_weights=(0.0, 1.0),
    ).set_index("team_id")
    long_window = build_team_strengths(
        base_df,
        lead_in_df,
        match_window=2,
        baseline_rating_weights=(1.0, 0.0),
        form_component_weights=(1.0, 0.0),
        strength_blend_weights=(0.0, 1.0),
    ).set_index("team_id")

    assert short_window.loc["A", "points_per_match"] == 3.0
    assert long_window.loc["A", "points_per_match"] == 1.5
    assert short_window.loc["A", "goal_diff_per_match"] == 4.0
    assert long_window.loc["A", "goal_diff_per_match"] == 0.5


def test_extract_group_stage_fixtures_has_six_matches_and_three_per_team():
    fixtures_df = pd.read_csv(DATA_DIR / "fixtures.csv")

    group_fixtures = extract_group_stage_fixtures(fixtures_df)

    counts_by_group = group_fixtures.groupby("group_code").size()
    assert counts_by_group.eq(6).all()

    group_a = group_fixtures[group_fixtures["group_code"] == "A"]
    appearance_counts = pd.concat([group_a["home_team_id"], group_a["away_team_id"]]).value_counts()
    assert len(appearance_counts) == 4
    assert appearance_counts.eq(3).all()


def test_rank_group_standings_uses_head_to_head_after_overall_tie():
    table_df = pd.DataFrame(
        [
            {"team_id": "A", "points": 6, "goals_for": 3, "goals_against": 1, "team_strength": 0.9},
            {"team_id": "B", "points": 6, "goals_for": 3, "goals_against": 1, "team_strength": 0.1},
            {"team_id": "C", "points": 3, "goals_for": 1, "goals_against": 3, "team_strength": -0.1},
            {"team_id": "D", "points": 3, "goals_for": 1, "goals_against": 4, "team_strength": -0.2},
        ]
    )
    fixture_results_df = pd.DataFrame(
        [
            {"home_team_id": "A", "away_team_id": "B", "home_goals": 0, "away_goals": 1},
            {"home_team_id": "A", "away_team_id": "C", "home_goals": 1, "away_goals": 0},
            {"home_team_id": "A", "away_team_id": "D", "home_goals": 2, "away_goals": 0},
            {"home_team_id": "B", "away_team_id": "C", "home_goals": 2, "away_goals": 0},
            {"home_team_id": "B", "away_team_id": "D", "home_goals": 0, "away_goals": 1},
            {"home_team_id": "C", "away_team_id": "D", "home_goals": 0, "away_goals": 1},
        ]
    )

    ranked = rank_group_standings(table_df, fixture_results_df)

    assert ranked["team_id"].tolist()[:2] == ["B", "A"]


def test_simulate_group_probabilities_preserves_probability_invariants():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()

    dashboard_df = simulate_group_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=120,
    )

    required_columns = {
        "top8_third_prob",
        "ko_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "champion_prob",
    }
    assert required_columns.issubset(dashboard_df.columns)

    for _, row in dashboard_df.iterrows():
        total_probability = row["prob_1"] + row["prob_2"] + row["prob_3"] + row["prob_4"]
        assert abs(total_probability - 100.0) < 1e-9
        assert abs(row["ko_prob"] - (row["prob_1"] + row["prob_2"] + row["top8_third_prob"])) < 1e-9
        assert row["champion_prob"] <= row["final_prob"] + 1e-9
        assert row["final_prob"] <= row["sf_prob"] + 1e-9
        assert row["sf_prob"] <= row["qf_prob"] + 1e-9
        assert row["qf_prob"] <= row["r16_prob"] + 1e-9
        assert row["r16_prob"] <= row["ko_prob"] + 1e-9
        for column_name in required_columns:
            assert 0.0 <= row[column_name] <= 100.0

    place_totals = (
        dashboard_df.groupby("group_code")[["prob_1", "prob_2", "prob_3", "prob_4"]]
        .sum()
        .round(10)
    )
    assert (place_totals == 100.0).all().all()


def test_simulate_group_probabilities_tracks_modal_group_rankings():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()

    dashboard_df = simulate_group_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=40,
    )

    modal_rankings = get_modal_group_rankings(dashboard_df)

    assert set(modal_rankings) == set(home.GROUP_ORDER)
    for group_code, ranked_team_ids in modal_rankings.items():
        assert len(ranked_team_ids) == 4
        assert len(set(ranked_team_ids)) == 4
        expected_group_team_ids = set(dashboard_df[dashboard_df["group_code"] == group_code]["team_id"])
        assert set(ranked_team_ids) == expected_group_team_ids


def test_rank_best_third_place_teams_uses_points_goal_difference_goals_for_then_strength():
    third_place_df = pd.DataFrame(
        [
            {"team_id": "A", "group_code": "A", "points": 4, "goal_difference": 1, "goals_for": 4, "team_strength": 0.1},
            {"team_id": "B", "group_code": "B", "points": 4, "goal_difference": 1, "goals_for": 4, "team_strength": 0.8},
            {"team_id": "C", "group_code": "C", "points": 4, "goal_difference": 1, "goals_for": 3, "team_strength": 0.9},
            {"team_id": "D", "group_code": "D", "points": 4, "goal_difference": 0, "goals_for": 5, "team_strength": 1.0},
        ]
    )

    ranked = rank_best_third_place_teams(third_place_df, qualification_slots=2)

    assert ranked["team_id"].tolist() == ["B", "A", "C", "D"]
    assert ranked["qualifies_as_best_third"].tolist() == [True, True, False, False]


def test_rank_best_third_place_teams_marks_exactly_eight_qualifiers():
    third_place_df = pd.DataFrame(
        [
            {
                "team_id": f"T{index}",
                "group_code": chr(65 + index),
                "points": 6 - (index // 3),
                "goal_difference": 5 - index,
                "goals_for": 12 - index,
                "team_strength": float(12 - index),
            }
            for index in range(12)
        ]
    )

    ranked = rank_best_third_place_teams(third_place_df)

    assert int(ranked["qualifies_as_best_third"].sum()) == 8


def test_third_place_routing_map_includes_known_knockout_combination():
    assert THIRD_PLACE_ROUTING_MAP["EFGHIJKL"] == {
        79: "E",
        85: "J",
        81: "I",
        74: "F",
        82: "H",
        77: "G",
        87: "L",
        80: "K",
    }


def test_predict_knockout_matchup_returns_valid_winner_and_probability():
    prediction = predict_knockout_matchup(
        "AAA",
        "BBB",
        {"AAA": 1.2, "BBB": 0.3},
        simulations=200,
        seed=7,
    )

    assert prediction["winner_team_id"] in {"AAA", "BBB"}
    assert 50.0 <= prediction["winner_win_prob"] <= 100.0
    assert abs(prediction["home_win_prob"] + prediction["away_win_prob"] - 100.0) < 1e-9


def test_build_deterministic_bracket_produces_consistent_field():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()
    dashboard_df = simulate_group_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=60,
    )

    bracket = build_deterministic_bracket(dashboard_df, fixtures_df, head_to_head_simulations=200, seed=11)

    modal_rankings = bracket["modal_group_rankings"]
    assert len(modal_rankings) == 12
    assert len({rankings[0] for rankings in modal_rankings.values()}) == 12
    assert len({rankings[1] for rankings in modal_rankings.values()}) == 12
    assert len(bracket["qualifying_third_place_team_ids"]) == 8
    all_qualifiers = (
        [rankings[0] for rankings in modal_rankings.values()]
        + [rankings[1] for rankings in modal_rankings.values()]
        + bracket["qualifying_third_place_team_ids"]
    )
    assert len(all_qualifiers) == len(set(all_qualifiers))
    assert bracket["qualifying_third_place_groups"] in THIRD_PLACE_ROUTING_MAP
    assert [round_data["round_code"] for round_data in bracket["rounds"]] == ["R32", "R16", "QF", "SF", "F"]
    assert sum(len(round_data["matches"]) for round_data in bracket["rounds"]) == 31


def test_projected_group_table_frame_uses_modal_group_rankings():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {"team_id": "BBB", "group_code": "A", "display_name": "B", "flag_icon_code": "bb", "world_rank": 20, "elo_rating": 1800, "prob_1": 40.0, "prob_2": 30.0, "prob_3": 20.0, "prob_4": 10.0, "top8_third_prob": 5.0},
            {"team_id": "CCC", "group_code": "A", "display_name": "C", "flag_icon_code": "cc", "world_rank": 30, "elo_rating": 1750, "prob_1": 25.0, "prob_2": 35.0, "prob_3": 25.0, "prob_4": 15.0, "top8_third_prob": 10.0},
            {"team_id": "AAA", "group_code": "A", "display_name": "A", "flag_icon_code": "aa", "world_rank": 10, "elo_rating": 1900, "prob_1": 35.0, "prob_2": 25.0, "prob_3": 25.0, "prob_4": 15.0, "top8_third_prob": 8.0},
            {"team_id": "DDD", "group_code": "A", "display_name": "D", "flag_icon_code": "dd", "world_rank": 40, "elo_rating": 1600, "prob_1": 0.0, "prob_2": 10.0, "prob_3": 30.0, "prob_4": 60.0, "top8_third_prob": 2.0},
        ]
    )
    sample_df.attrs["modal_group_rankings"] = {"A": ["CCC", "AAA", "BBB", "DDD"]}

    projected = home.projected_group_table_frame(sample_df, "A")

    assert list(projected["team_id"]) == ["CCC", "AAA", "BBB", "DDD"]


def test_build_table_html_smoke_contains_expected_probability_columns():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "group_code": "J",
                "flag_icon_code": "ar",
                "display_name": "Argentina",
                "world_rank": 1,
                "elo_rating": 2140,
                "prob_1": 61.5,
                "prob_2": 24.5,
                "prob_3": 10.0,
                "prob_4": 4.0,
                "top8_third_prob": 1.0,
                "ko_prob": 86.0,
            }
        ]
    )

    html = home.build_table_html(sample_df, "Group J", include_group_column=False, include_ko_column=False)

    assert "Country" in html
    assert "World Rank" in html
    assert "1st %" in html
    assert "2nd %" in html
    assert "3rd %" in html
    assert "4th %" in html
    assert "KO %" not in html


def test_current_view_tables_adds_projected_order_for_group_views():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {"team_id": "AAA", "group_code": "A", "display_name": "A", "flag_icon_code": "aa", "world_rank": 10, "elo_rating": 1900, "prob_1": 50.0, "prob_2": 20.0, "prob_3": 20.0, "prob_4": 10.0, "top8_third_prob": 5.0},
            {"team_id": "BBB", "group_code": "A", "display_name": "B", "flag_icon_code": "bb", "world_rank": 20, "elo_rating": 1800, "prob_1": 30.0, "prob_2": 30.0, "prob_3": 20.0, "prob_4": 20.0, "top8_third_prob": 6.0},
            {"team_id": "CCC", "group_code": "A", "display_name": "C", "flag_icon_code": "cc", "world_rank": 30, "elo_rating": 1700, "prob_1": 20.0, "prob_2": 30.0, "prob_3": 30.0, "prob_4": 20.0, "top8_third_prob": 7.0},
            {"team_id": "DDD", "group_code": "A", "display_name": "D", "flag_icon_code": "dd", "world_rank": 40, "elo_rating": 1600, "prob_1": 0.0, "prob_2": 20.0, "prob_3": 30.0, "prob_4": 50.0, "top8_third_prob": 8.0},
        ]
    )
    sample_df.attrs["modal_group_rankings"] = {"A": ["BBB", "AAA", "CCC", "DDD"]}

    tables = home.current_view_tables(sample_df, "Single group", "A", simulation_count=100000)

    assert len(tables) == 1
    assert tables[0]["title"] == "Group A"
    assert tables[0]["card_subtitle"] == home.chart_subtitle("Bracket-Aligned Projected Order", 100000)
    assert list(tables[0]["frame"]["team_id"]) == ["BBB", "AAA", "CCC", "DDD"]


def test_build_table_html_group_views_include_qualification_marker():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "group_code": "J",
                "flag_icon_code": "ar",
                "display_name": "Argentina",
                "world_rank": 1,
                "elo_rating": 2140,
                "prob_1": 61.5,
                "prob_2": 24.5,
                "prob_3": 10.0,
                "prob_4": 4.0,
                "top8_third_prob": 1.0,
                "ko_prob": 87.0,
            }
        ]
    )

    html = home.build_table_html(sample_df, "Group J", include_group_column=False, include_ko_column=False)

    assert "wc-qual-marker" in html
    assert "wc-qual-segment-top2" in html
    assert "wc-qual-segment-third" in html


def test_build_form_table_html_includes_confederation_column():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "display_name": "Argentina",
                "flag_icon_code": "ar",
                "confederation": "CONMEBOL",
                "wins": 7,
                "draws": 2,
                "losses": 1,
                "goals_for": 18,
                "goals_against": 7,
                "elo_rating": 2140,
                "avg_opp_elo": 1888.4,
                "avg_elo_gap": 42.7,
                "schedule_difficulty": 4.3,
                "results_form": 0.85,
                "gd_form": 1.7,
                "expected_score": 0.63,
                "perf_vs_exp": 0.22,
                "elo_delta_form": 7.4,
                "form": 9.1,
            }
        ]
    )

    html = home.build_table_html(sample_df, "Form", table_kind="form")

    assert "Rank" in html
    assert '>1<' in html
    assert "Confederation" in html
    assert "CONMEBOL" in html
    assert "Results Form" in html
    assert "GD Form" in html
    assert "Perf vs Exp" in html
    assert "Elo Delta Form" in html
    assert "Sched Diff" in html
    assert "#173404" in html
    assert "#633806" in html
    assert "background-color: #" in html
    assert ">42.7</td>" in html


def test_build_form_table_html_includes_v2_history_columns_when_available():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "display_name": "Argentina",
                "flag_icon_code": "ar",
                "confederation": "CONMEBOL",
                "wins": 7,
                "draws": 2,
                "losses": 1,
                "goals_for": 18,
                "goals_against": 7,
                "elo_rating": 2140,
                "avg_opp_elo": 1888.4,
                "avg_elo_gap": 42.7,
                "schedule_difficulty": 4.3,
                "results_form": 0.85,
                "gd_form": 1.7,
                "expected_score": 0.63,
                "perf_vs_exp": 0.22,
                "elo_delta_form": 7.4,
                "form": 9.1,
                "weighted_world_cup_participations": 2500.0,
                "weighted_world_cup_placement_score": 0.8123,
                "history_score": 0.7186,
                "v2_strength": 8.8,
            }
        ]
    )

    html = home.build_table_html(sample_df, "V2", table_kind="form")

    assert "Wtd WC Apps" in html
    assert "Wtd WC Place" in html
    assert "History" in html
    assert "V2 Strength" in html
    assert ">2500.0</td>" in html
    assert ">0.8123</td>" in html
    assert ">8.8</td>" in html


def test_form_color_helpers_use_gradients_within_each_tier():
    home = load_home_module()

    bad_low = home.sequential_form_cell_style(0.05, 0.0, 1.0)
    bad_high = home.sequential_form_cell_style(0.25, 0.0, 1.0)
    assert bad_low != bad_high
    assert "color: #791F1F;" in bad_low
    assert "color: #791F1F;" in bad_high

    mid_low = home.sequential_form_cell_style(0.40, 0.0, 1.0)
    mid_high = home.sequential_form_cell_style(0.60, 0.0, 1.0)
    assert mid_low != mid_high
    assert "color: #633806;" in mid_low
    assert "color: #633806;" in mid_high

    good_low = home.sequential_form_cell_style(0.75, 0.0, 1.0)
    good_high = home.sequential_form_cell_style(0.95, 0.0, 1.0)
    assert good_low != good_high
    assert "color: #173404;" in good_low
    assert "color: #173404;" in good_high


def test_build_form_table_html_reverses_schedule_difficulty_colors():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "AAA",
                "display_name": "Alpha",
                "flag_icon_code": "aa",
                "confederation": "UEFA",
                "wins": 5,
                "draws": 1,
                "losses": 0,
                "goals_for": 12,
                "goals_against": 3,
                "elo_rating": 1900,
                "avg_opp_elo": 1800.0,
                "avg_elo_gap": -20.0,
                "schedule_difficulty": 1.0,
                "results_form": 0.9,
                "gd_form": 1.5,
                "expected_score": 0.7,
                "perf_vs_exp": 0.2,
                "elo_delta_form": 8.0,
                "form": 1.1,
            },
            {
                "team_id": "BBB",
                "display_name": "Beta",
                "flag_icon_code": "bb",
                "confederation": "CAF",
                "wins": 1,
                "draws": 1,
                "losses": 4,
                "goals_for": 4,
                "goals_against": 11,
                "elo_rating": 1700,
                "avg_opp_elo": 1900.0,
                "avg_elo_gap": 40.0,
                "schedule_difficulty": 5.0,
                "results_form": 0.2,
                "gd_form": -1.7,
                "expected_score": 0.3,
                "perf_vs_exp": -0.4,
                "elo_delta_form": -6.0,
                "form": -1.0,
            },
        ]
    )

    html = home.build_table_html(sample_df, "Form", table_kind="form")

    assert "background-color: #3B6D11; color: #173404;\">1.0</td>" in html
    assert "background-color: #A32D2D; color: #791F1F;\">5.0</td>" in html


def test_build_form_view_tables_adds_confederation_tables():
    home = load_home_module()
    form_df = pd.DataFrame(
        [
            {
                "team_id": "A1",
                "display_name": "Alpha",
                "flag_icon_code": "aa",
                "confederation": "UEFA",
                "wins": 5,
                "draws": 2,
                "losses": 1,
                "goals_for": 10,
                "goals_against": 4,
                "elo_rating": 1900,
                "world_rank": 4,
                "avg_opp_elo": 1820.0,
                "avg_elo_gap": 40.0,
                "schedule_difficulty": 4.2,
                "results_form": 0.82,
                "gd_form": 1.3,
                "expected_score": 0.58,
                "perf_vs_exp": 0.24,
                "elo_delta_form": 6.8,
                "form": 8.0,
            },
            {
                "team_id": "B1",
                "display_name": "Beta",
                "flag_icon_code": "bb",
                "confederation": "CAF",
                "wins": 4,
                "draws": 3,
                "losses": 1,
                "goals_for": 8,
                "goals_against": 3,
                "elo_rating": 1800,
                "world_rank": 11,
                "avg_opp_elo": 1750.0,
                "avg_elo_gap": 5.0,
                "schedule_difficulty": 2.5,
                "results_form": 0.63,
                "gd_form": 0.7,
                "expected_score": 0.51,
                "perf_vs_exp": 0.12,
                "elo_delta_form": 2.1,
                "form": 5.0,
            },
        ]
    )

    tables = home.build_form_view_tables(form_df, form_match_window=10)

    assert [table["title"] for table in tables] == ["All Countries", "CAF", "UEFA"]
    assert tables[1]["frame"]["confederation"].unique().tolist() == ["CAF"]
    assert tables[2]["frame"]["confederation"].unique().tolist() == ["UEFA"]
    assert all(table["table_kind"] == "form" for table in tables)


def test_current_form_view_tables_separates_all_countries_and_confederations():
    home = load_home_module()
    form_df = pd.DataFrame(
        [
            {
                "team_id": "A1",
                "display_name": "Alpha",
                "flag_icon_code": "aa",
                "confederation": "UEFA",
                "wins": 5,
                "draws": 2,
                "losses": 1,
                "goals_for": 10,
                "goals_against": 4,
                "elo_rating": 1900,
                "world_rank": 4,
                "avg_opp_elo": 1820.0,
                "avg_elo_gap": 40.0,
                "schedule_difficulty": 4.2,
                "results_form": 0.82,
                "gd_form": 1.3,
                "expected_score": 0.58,
                "perf_vs_exp": 0.24,
                "elo_delta_form": 6.8,
                "form": 8.0,
            },
            {
                "team_id": "B1",
                "display_name": "Beta",
                "flag_icon_code": "bb",
                "confederation": "CAF",
                "wins": 4,
                "draws": 3,
                "losses": 1,
                "goals_for": 8,
                "goals_against": 3,
                "elo_rating": 1800,
                "world_rank": 11,
                "avg_opp_elo": 1750.0,
                "avg_elo_gap": 5.0,
                "schedule_difficulty": 2.5,
                "results_form": 0.63,
                "gd_form": 0.7,
                "expected_score": 0.51,
                "perf_vs_exp": 0.12,
                "elo_delta_form": 2.1,
                "form": 5.0,
            },
        ]
    )

    all_countries = home.current_form_view_tables(form_df, "All Countries", "", form_match_window=10)
    single_confederation = home.current_form_view_tables(form_df, "Single confederation", "CAF", form_match_window=10)
    all_confederations = home.current_form_view_tables(form_df, "All confederations", "", form_match_window=10)

    assert [table["title"] for table in all_countries] == ["All Countries"]
    assert [table["title"] for table in single_confederation] == ["CAF"]
    assert single_confederation[0]["frame"]["confederation"].unique().tolist() == ["CAF"]
    assert [table["title"] for table in all_confederations] == ["CAF", "UEFA"]


def test_v2_view_options_include_confederation_views():
    home = load_home_module()

    assert home.V2_VIEW_OPTIONS == ("All Countries", "Single confederation", "All confederations")


def test_form_all_tables_download_frame_includes_confederation_sections():
    home = load_home_module()
    form_df = pd.DataFrame(
        [
            {
                "team_id": "A1",
                "display_name": "Alpha",
                "flag_icon_code": "aa",
                "confederation": "UEFA",
                "wins": 5,
                "draws": 2,
                "losses": 1,
                "goals_for": 10,
                "goals_against": 4,
                "elo_rating": 1900,
                "world_rank": 4,
                "avg_opp_elo": 1820.0,
                "avg_elo_gap": 40.0,
                "schedule_difficulty": 4.2,
                "results_form": 0.82,
                "gd_form": 1.3,
                "expected_score": 0.58,
                "perf_vs_exp": 0.24,
                "elo_delta_form": 6.8,
                "form": 8.0,
            },
            {
                "team_id": "B1",
                "display_name": "Beta",
                "flag_icon_code": "bb",
                "confederation": "CAF",
                "wins": 4,
                "draws": 3,
                "losses": 1,
                "goals_for": 8,
                "goals_against": 3,
                "elo_rating": 1800,
                "world_rank": 11,
                "avg_opp_elo": 1750.0,
                "avg_elo_gap": 5.0,
                "schedule_difficulty": 2.5,
                "results_form": 0.63,
                "gd_form": 0.7,
                "expected_score": 0.51,
                "perf_vs_exp": 0.12,
                "elo_delta_form": 2.1,
                "form": 5.0,
            },
        ]
    )

    download_frame = home.form_all_tables_download_frame(form_df, form_match_window=10)

    assert "section" in download_frame.columns
    assert set(download_frame["section"]) == {"All Countries", "CAF", "UEFA"}
    assert set(download_frame["confederation"]) == {"CAF", "UEFA"}


def test_render_tables_uses_single_column_wrapper_for_stacked_sections(monkeypatch):
    home = load_home_module()
    captured = {}

    def fake_markdown(content, unsafe_allow_html=False):
        captured["content"] = content
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(home.st, "markdown", fake_markdown)

    home.render_tables(
        [
            {
                "title": "CAF",
                "frame": pd.DataFrame(
                    [
                        {
                            "team_id": "B1",
                            "display_name": "Beta",
                            "flag_icon_code": "bb",
                            "confederation": "CAF",
                            "wins": 4,
                            "draws": 3,
                            "losses": 1,
                            "goals_for": 8,
                            "goals_against": 3,
                            "elo_rating": 1800,
                            "world_rank": 11,
                            "avg_opp_elo": 1750.0,
                            "avg_elo_gap": 5.0,
                            "schedule_difficulty": 2.5,
                            "results_form": 0.63,
                            "gd_form": 0.7,
                            "expected_score": 0.51,
                            "perf_vs_exp": 0.12,
                            "elo_delta_form": 2.1,
                            "form": 5.0,
                        }
                    ]
                ),
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": "Weighted Recent Form | Last 10 lead-in matches with Elo",
                "group_pill_label": None,
                "table_kind": "form",
            }
        ],
        multi_column=False,
    )

    assert 'class="wc-grid-single"' in captured["content"]
    assert captured["unsafe_allow_html"] is True


def test_render_tables_can_render_separate_single_column_sections(monkeypatch):
    home = load_home_module()
    captured = {}

    def fake_markdown(content, unsafe_allow_html=False):
        captured["content"] = content
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(home.st, "markdown", fake_markdown)

    home.render_tables(
        [
            {
                "title": "CAF",
                "frame": pd.DataFrame(
                    [
                        {
                            "team_id": "B1",
                            "display_name": "Beta",
                            "flag_icon_code": "bb",
                            "confederation": "CAF",
                            "wins": 4,
                            "draws": 3,
                            "losses": 1,
                            "goals_for": 8,
                            "goals_against": 3,
                            "elo_rating": 1800,
                            "world_rank": 11,
                            "avg_opp_elo": 1750.0,
                            "avg_elo_gap": 5.0,
                            "schedule_difficulty": 2.5,
                            "results_form": 0.63,
                            "gd_form": 0.7,
                            "expected_score": 0.51,
                            "perf_vs_exp": 0.12,
                            "elo_delta_form": 2.1,
                            "form": 5.0,
                        }
                    ]
                ),
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": "Weighted Recent Form | Last 10 lead-in matches with Elo",
                "group_pill_label": None,
                "table_kind": "form",
            },
            {
                "title": "UEFA",
                "frame": pd.DataFrame(
                    [
                        {
                            "team_id": "A1",
                            "display_name": "Alpha",
                            "flag_icon_code": "aa",
                            "confederation": "UEFA",
                            "wins": 5,
                            "draws": 2,
                            "losses": 1,
                            "goals_for": 10,
                            "goals_against": 4,
                            "elo_rating": 1900,
                            "world_rank": 4,
                            "avg_opp_elo": 1820.0,
                            "avg_elo_gap": 40.0,
                            "schedule_difficulty": 4.2,
                            "results_form": 0.82,
                            "gd_form": 1.3,
                            "expected_score": 0.58,
                            "perf_vs_exp": 0.24,
                            "elo_delta_form": 6.8,
                            "form": 8.0,
                        }
                    ]
                ),
                "include_group_column": False,
                "include_ko_column": False,
                "card_subtitle": "Weighted Recent Form | Last 10 lead-in matches with Elo",
                "group_pill_label": None,
                "table_kind": "form",
            },
        ],
        multi_column=False,
        separate_sections=True,
    )

    assert captured["content"].count('class="wc-grid-single"') == 2
    assert captured["unsafe_allow_html"] is True


def test_view_options_include_form_and_bracket():
    home = load_home_module()

    assert "Form" in home.VIEW_OPTIONS
    assert "Bracket" in home.VIEW_OPTIONS


def test_build_bracket_html_renders_rounds_and_winner_probabilities():
    home = load_home_module()
    metadata_lookup = {
        "ARG": {"display_name": "Argentina", "flag_icon_code": "ar"},
        "FRA": {"display_name": "France", "flag_icon_code": "fr"},
    }
    bracket_data = {
        "qualifying_third_place_groups": "EFGHIJKL",
        "rounds": [
            {
                "round_code": "R32",
                "round_label": "Round of 32",
                "matches": [
                    {
                        "match_number": 73,
                        "home_team_id": "ARG",
                        "away_team_id": "FRA",
                        "winner_team_id": "ARG",
                        "winner_win_prob": 61.5,
                    }
                ],
            },
            {"round_code": "R16", "round_label": "Round of 16", "matches": []},
            {"round_code": "QF", "round_label": "Quarter-finals", "matches": []},
            {"round_code": "SF", "round_label": "Semi-finals", "matches": []},
            {"round_code": "F", "round_label": "Final", "matches": []},
        ],
    }

    html = home.build_bracket_html(
        bracket_data,
        metadata_lookup,
        card_subtitle="Predicted Knockout Bracket | 100,000 simulations",
    )

    assert "Predicted Knockout Bracket | 100,000 simulations" in html
    assert "wc-bracket-side-left" in html
    assert "wc-bracket-final-column" in html
    assert "wc-bracket-side-right" in html
    assert "Round of 32" in html
    assert "Quarter-finals" in html
    assert "61.5%" in html
    assert "wc-bracket-team-win" in html
    assert "Argentina" in html
    assert "France" in html
    assert "Play-off for third place" not in html


def test_single_group_download_frame_preserves_probability_columns():
    home = load_home_module()
    probability_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "group_code": "J",
                "flag_icon_code": "ar",
                "display_name": "Argentina",
                "confederation": "CONMEBOL",
                "world_rank": 1,
                "elo_rating": 2140,
                "prob_1": 80.0,
                "prob_2": 10.0,
                "prob_3": 5.0,
                "prob_4": 5.0,
                "top8_third_prob": 4.0,
                "ko_prob": 94.0,
                "r16_prob": 70.0,
                "qf_prob": 50.0,
                "sf_prob": 30.0,
                "final_prob": 20.0,
                "champion_prob": 10.0,
            }
        ]
    )

    tables = home.current_view_tables(
        probability_df,
        "Single group",
        "J",
        simulation_count=1000,
    )
    download_frame = home.tables_to_download_frame(tables, section_column="group")

    assert "display_name" in download_frame.columns
    assert "prob_1" in download_frame.columns
    assert download_frame.loc[0, "display_name"] == "Argentina"
    assert download_frame.loc[0, "prob_1"] == 80.0


def test_bracket_download_frame_flattens_matches_without_png_renderer():
    home = load_home_module()
    metadata_lookup = {
        "ARG": {"display_name": "Argentina", "flag_icon_code": "ar"},
        "FRA": {"display_name": "France", "flag_icon_code": "fr"},
    }
    bracket_data = {
        "rounds": [
            {
                "round_code": "R32",
                "round_label": "Round of 32",
                "matches": [
                    {
                        "match_number": 73,
                        "home_team_id": "ARG",
                        "away_team_id": "FRA",
                        "winner_team_id": "ARG",
                        "winner_win_prob": 61.5,
                    }
                ],
            }
        ]
    }

    download_frame = home.bracket_to_download_frame(bracket_data, metadata_lookup)

    assert download_frame.loc[0, "round_code"] == "R32"
    assert download_frame.loc[0, "slot"] == 1
    assert download_frame.loc[0, "home_team"] == "Argentina"
    assert download_frame.loc[0, "away_team"] == "France"
    assert download_frame.loc[0, "winner_win_prob"] == 61.5


def test_export_document_css_omits_flag_icons_cdn():
    home = load_home_module()

    document = home.render_export_document("Export", [], multi_column=False)

    assert "cdn.jsdelivr.net/npm/flag-icons" not in document
    assert "wc-export-mode" in document


def test_export_document_includes_standalone_compat_css():
    home = load_home_module()

    document = home.render_export_document("Export", [], multi_column=False)

    assert ".wc-export-mode .wc-table thead th" in document
    assert ".wc-export-mode .wc-flag-fallback" in document
    assert ".wc-export-mode .wc-grid .wc-card" in document
    assert "display: inline-block;" in document
    assert "width: 31.6%;" in document
    assert "#5A4632" in document
    assert "#F6EBD8" in document
    assert "var(--wc-muted)" in document
    assert document.index("var(--wc-muted)") < document.index(".wc-export-mode .wc-table thead th")


def test_shared_css_hides_country_names_on_narrow_screens():
    home = load_home_module()

    css = home.shared_css()
    media_start = css.index("@media (max-width: 1100px)")
    media_end = css.index("@media (max-width: 860px)")
    responsive_css = css[media_start:media_end]

    assert ".wc-name-cell" in responsive_css
    assert ".wc-name-main" in responsive_css
    assert ".wc-name-text" in responsive_css
    assert "display: none;" in responsive_css
    assert ".wc-qual-marker" in responsive_css


def test_render_name_cell_includes_image_flag_fallback():
    home = load_home_module()

    cell_html = home.render_name_cell("ar", "Argentina")

    assert 'class="fi fi-ar"' in cell_html
    assert '<img class="wc-flag-fallback"' in cell_html
    assert "https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/flags/4x3/ar.svg" in cell_html


def test_home_reexports_csv_export_helpers():
    home = load_home_module()

    assert home.tables_to_download_frame is dashboard_export.tables_to_download_frame
    assert home.bracket_to_download_frame is dashboard_export.bracket_to_download_frame


def test_build_table_html_all_countries_includes_ko_column_only_when_requested():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "group_code": "J",
                "flag_icon_code": "ar",
                "display_name": "Argentina",
                "confederation": "CONMEBOL",
                "world_rank": 1,
                "elo_rating": 2140,
                "prob_1": 61.5,
                "prob_2": 24.5,
                "prob_3": 10.0,
                "prob_4": 4.0,
                "top8_third_prob": 1.0,
                "ko_prob": 86.0,
                "r16_prob": 61.0,
                "qf_prob": 39.0,
                "sf_prob": 22.0,
                "final_prob": 12.0,
                "champion_prob": 7.0,
            }
        ]
    )

    html = home.build_table_html(sample_df, "All Countries", include_group_column=True, include_ko_column=True)

    assert "Confederation" in html
    assert "KO %" in html
    assert "R16 %" in html
    assert "QF %" in html
    assert "SF %" in html
    assert "Final %" in html
    assert "Champion %" in html
    assert "Rank" in html
    assert "1st %" not in html
    assert "2nd %" not in html
    assert "3rd %" not in html
    assert "4th %" not in html
    assert "Top 8 3rd %" not in html
    assert "CONMEBOL" in html
    assert "86.0%" in html
    assert "wc-qual-marker" not in html


def test_build_table_html_all_countries_embeds_champion_trophy_icon():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "group_code": "J",
                "flag_icon_code": "ar",
                "display_name": "Argentina",
                "confederation": "CONMEBOL",
                "world_rank": 1,
                "elo_rating": 2140,
                "prob_1": 61.5,
                "prob_2": 24.5,
                "prob_3": 10.0,
                "prob_4": 4.0,
                "top8_third_prob": 1.0,
                "ko_prob": 86.0,
                "r16_prob": 61.0,
                "qf_prob": 39.0,
                "sf_prob": 22.0,
                "final_prob": 12.0,
                "champion_prob": 7.0,
            }
        ]
    )

    html = home.build_table_html(sample_df, "All Countries", include_group_column=True, include_ko_column=True)

    assert "data:image/svg+xml;base64," in html
    assert "Champion trophy" in html


def test_build_table_html_renders_simulation_count_when_provided():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "ARG",
                "group_code": "J",
                "flag_icon_code": "ar",
                "display_name": "Argentina",
                "confederation": "CONMEBOL",
                "world_rank": 1,
                "elo_rating": 2140,
                "prob_1": 61.5,
                "prob_2": 24.5,
                "prob_3": 10.0,
                "prob_4": 4.0,
                "top8_third_prob": 1.0,
                "ko_prob": 86.0,
                "r16_prob": 61.0,
                "qf_prob": 39.0,
                "sf_prob": 22.0,
                "final_prob": 12.0,
                "champion_prob": 7.0,
            }
        ]
    )

    html = home.build_table_html(
        sample_df,
        "All Countries",
        include_group_column=True,
        include_ko_column=True,
        card_subtitle=home.chart_subtitle("Pre-Tournament Probability Table", 100000),
    )

    assert "100,000 simulations" in html
    assert "&lt;/div&gt;" not in html


def test_all_teams_table_frame_sorts_by_champion_then_deeper_rounds():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {
                "team_id": "A",
                "champion_prob": 8.0,
                "final_prob": 15.0,
                "sf_prob": 28.0,
                "qf_prob": 42.0,
                "r16_prob": 61.0,
                "ko_prob": 82.0,
                "top8_third_prob": 2.0,
                "prob_1": 40.0,
                "elo_rating": 1800,
                "world_rank": 5,
            },
            {
                "team_id": "B",
                "champion_prob": 10.0,
                "final_prob": 14.0,
                "sf_prob": 25.0,
                "qf_prob": 40.0,
                "r16_prob": 60.0,
                "ko_prob": 80.0,
                "top8_third_prob": 1.0,
                "prob_1": 38.0,
                "elo_rating": 1700,
                "world_rank": 8,
            },
            {
                "team_id": "C",
                "champion_prob": 8.0,
                "final_prob": 16.0,
                "sf_prob": 29.0,
                "qf_prob": 43.0,
                "r16_prob": 62.0,
                "ko_prob": 82.0,
                "top8_third_prob": 3.0,
                "prob_1": 41.0,
                "elo_rating": 1750,
                "world_rank": 7,
            },
        ]
    )

    sorted_df = home.all_teams_table_frame(sample_df)

    assert sorted_df["team_id"].tolist() == ["B", "C", "A"]


def test_ensure_dashboard_probability_columns_backfills_missing_ko_prob():
    home = load_home_module()
    sample_df = pd.DataFrame(
        [
            {"team_id": "A", "prob_1": 40.0, "prob_2": 35.0, "prob_3": 20.0, "prob_4": 5.0},
        ]
    )

    normalized = home.ensure_dashboard_probability_columns(sample_df)

    assert "top8_third_prob" in normalized.columns
    assert "ko_prob" in normalized.columns
    assert "r32_prob" in normalized.columns
    assert "r16_prob" in normalized.columns
    assert "qf_prob" in normalized.columns
    assert "sf_prob" in normalized.columns
    assert "final_prob" in normalized.columns
    assert "champion_prob" in normalized.columns
    assert normalized.loc[0, "top8_third_prob"] == 0.0
    assert normalized.loc[0, "ko_prob"] == 75.0
    assert normalized.loc[0, "r32_prob"] == 75.0


def test_simulate_probabilities_accepts_custom_weight_filters():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()

    dashboard_df = home.simulate_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=20,
        baseline_rating_weights=(1.0, 0.0),
        form_component_weights=(1.0, 0.0),
        strength_blend_weights=(1.0, 0.0),
    )

    assert {"rating_score", "form_score", "team_strength"}.issubset(dashboard_df.columns)


def test_simulate_probabilities_accepts_custom_recent_match_window():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()

    short_window_df = home.simulate_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=20,
        match_window=5,
    )
    long_window_df = home.simulate_probabilities(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=20,
        match_window=8,
    )

    assert not short_window_df["form_score"].equals(long_window_df["form_score"])


def test_simulate_probabilities_falls_back_when_simulator_lacks_match_window(monkeypatch):
    home = load_home_module()
    captured = {}

    def legacy_simulator(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "team_id": "A",
                    "group_code": "A",
                    "prob_1": 40.0,
                    "prob_2": 35.0,
                    "prob_3": 15.0,
                    "prob_4": 10.0,
                }
            ]
        )

    monkeypatch.setattr(home, "simulate_group_probabilities", legacy_simulator)

    result = home.simulate_probabilities(
        base_df=pd.DataFrame([{"team_id": "A", "group_code": "A"}]),
        fixtures_df=pd.DataFrame(),
        lead_in_df=pd.DataFrame(),
        simulations=10,
        match_window=5,
    )

    assert "match_window" not in captured
    assert result.loc[0, "prob_1"] == 40.0


def test_simulate_probabilities_filters_all_unknown_optional_kwargs(monkeypatch):
    home = load_home_module()
    captured = {}

    def legacy_simulator(base_df, fixtures_df, lead_in_df, simulations, group_order):
        captured.update(
            {
                "base_df": base_df,
                "fixtures_df": fixtures_df,
                "lead_in_df": lead_in_df,
                "simulations": simulations,
                "group_order": group_order,
            }
        )
        return pd.DataFrame(
            [
                {
                    "team_id": "A",
                    "group_code": "A",
                    "prob_1": 50.0,
                    "prob_2": 25.0,
                    "prob_3": 15.0,
                    "prob_4": 10.0,
                }
            ]
        )

    monkeypatch.setattr(home, "simulate_group_probabilities", legacy_simulator)

    result = home.simulate_probabilities(
        base_df=pd.DataFrame([{"team_id": "A", "group_code": "A"}]),
        fixtures_df=pd.DataFrame(),
        lead_in_df=pd.DataFrame(),
        simulations=10,
        match_window=5,
        baseline_rating_weights=(1.0, 0.0),
        form_component_weights=(1.0, 0.0),
        strength_blend_weights=(1.0, 0.0),
    )

    assert captured["simulations"] == 10
    assert captured["group_order"] == home.GROUP_ORDER
    assert result.loc[0, "prob_1"] == 50.0


def test_build_export_stem_appends_suffix_without_overwriting_base_name():
    home = load_home_module()

    assert home.build_export_stem("group_a_view") == "group_a_view"
    assert home.build_export_stem("group_a_view", "20260403_220500_123456") == "group_a_view_20260403_220500_123456"


def test_get_first_kickoff_details_uses_earliest_group_stage_fixture():
    home = load_home_module()
    fixtures_df = pd.read_csv(DATA_DIR / "fixtures.csv")

    kickoff = home.get_first_kickoff_details(fixtures_df)

    assert kickoff["match_label"] == "Mexico vs South Africa"
    assert kickoff["kickoff_iso_utc"] == "2026-06-11T19:00:00Z"
    assert kickoff["kickoff_date_label"] == "June-11-2026"
    assert kickoff["kickoff_local_time_label"] == "13:00"
    assert kickoff["kickoff_utc_time_label"] == "19:00"


def test_training_anchor_resolution_uses_five_prior_world_cup_start():
    assert resolve_training_anchor_year(2026) == 2002
    assert resolve_training_anchor_date(2026).strftime("%Y-%m-%d") == "2002-05-31"
    assert resolve_training_anchor_year(2022) == 1998
    assert resolve_training_anchor_date(2022).strftime("%Y-%m-%d") == "1998-06-10"


def test_build_v2_training_frame_uses_anchor_editions_and_includes_knockout_rows():
    training_df = build_v2_training_frame(match_window=4)

    assert set(training_df["edition"].astype(int)) == {2002, 2006, 2010, 2014, 2018, 2022}
    assert {"Group Stage", "Quarter-final", "Semi-final", "Final"}.issubset(set(training_df["stage"]))
    assert {"group", "knockout"} == set(training_df["stage_bucket"])
    assert set(training_df["outcome_label"]).issubset({"home_win", "draw", "away_win"})
    assert set(training_df["training_scope"]) == {TRAINING_SCOPE_WORLD_CUP_ONLY}
    assert set(training_df["sample_weight"].astype(float)) == {3.0}
    for column_name in (
        "elo_diff",
        "results_form_diff",
        "gd_form_diff",
        "perf_vs_exp_diff",
        "goals_for_diff",
        "goals_against_diff",
        "placement_diff",
        "appearance_diff",
    ):
        assert column_name in training_df.columns


def test_build_v2_training_frame_excludes_holdout_edition():
    training_df = build_v2_training_frame(
        match_window=4,
        exclude_editions=(2022,),
        reference_edition_year=2022,
    )

    assert 2022 not in set(training_df["edition"].astype(int))
    assert set(training_df["edition"].astype(int)) == {1998, 2002, 2006, 2010, 2014, 2018}


def test_build_v2_training_frame_all_international_scope_has_tournament_mix_and_weights():
    training_df = build_v2_training_frame(
        match_window=4,
        training_scope=TRAINING_SCOPE_ALL_INTERNATIONAL,
        reference_edition_year=2022,
        end_date="1998-06-30",
    )

    assert not training_df.empty
    assert set(training_df["training_scope"]) == {TRAINING_SCOPE_ALL_INTERNATIONAL}
    assert str(pd.to_datetime(training_df["date"]).min().date()) >= "1998-06-10"
    assert str(pd.to_datetime(training_df["date"]).max().date()) <= "1998-06-30"
    assert training_df["tournament"].nunique() > 1
    assert training_df["sample_weight"].astype(float).between(1.0, 3.0).all()


def test_classify_competition_importance_uses_v3_scale():
    assert classify_competition_importance("FIFA World Cup") == 3.0
    assert classify_competition_importance("UEFA Euro") == 2.5
    assert classify_competition_importance("FIFA World Cup qualification") == 2.0
    assert classify_competition_importance("Friendly") == 1.0
    assert classify_competition_importance("Nehru Cup") == 1.5


def test_build_v3_training_frame_respects_cutoff_and_columns():
    results_df = pd.read_csv(INTERNATIONAL_RESULTS_PATH)
    cutoff = "2002-06-30"

    training_df = build_v3_training_frame(
        results_df,
        match_window=4,
        start_year=1998,
        end_date=cutoff,
    )

    assert not training_df.empty
    assert str(training_df["date"].max().date()) <= cutoff
    assert str(training_df["date"].min().date()) >= "2002-05-31"
    assert set(training_df["training_scope"]) == {TRAINING_SCOPE_ALL_INTERNATIONAL}
    for column_name in (
        "elo_diff",
        "results_form_diff",
        "goals_for_diff",
        "goals_against_diff",
        "placement_diff",
        "appearance_diff",
        "gd_form_diff",
        "perf_vs_exp_diff",
        "competition_importance",
        "neutral_site_flag",
        "net_host_flag",
    ):
        assert column_name in training_df.columns
    assert training_df["sample_weight"].astype(float).between(1.0, 3.0).all()


def test_build_v3_training_frame_world_cup_scope_has_constant_world_cup_weights():
    training_df = build_v3_training_frame(
        pd.DataFrame(),
        match_window=4,
        training_scope=TRAINING_SCOPE_WORLD_CUP_ONLY,
        reference_edition_year=2022,
        end_date="1998-06-30",
    )

    assert not training_df.empty
    assert set(training_df["training_scope"]) == {TRAINING_SCOPE_WORLD_CUP_ONLY}
    assert set(training_df["tournament"]) == {"FIFA World Cup"}
    assert set(training_df["sample_weight"].astype(float)) == {3.0}
    assert str(pd.to_datetime(training_df["date"]).min().date()) == "1998-06-10"


def test_fit_v3_model_predicts_valid_lambdas_and_probability_triplet():
    home = load_home_module()
    base_df, _, lead_in_df, _ = home.load_data()
    feature_df = build_v3_team_feature_table(base_df, lead_in_df, reference_date_or_edition=2026, match_window=4)
    feature_lookup = feature_df.set_index("team_id").to_dict("index")
    model_bundle = fit_v3_poisson_models(match_window=4)

    first_team_id = str(feature_df.iloc[0]["team_id"])
    second_team_id = str(feature_df.iloc[1]["team_id"])
    prediction = predict_match_lambdas_v3(first_team_id, second_team_id, feature_lookup, model_bundle, neutral_site=True)

    total_probability = (
        float(prediction["home_win_prob"])
        + float(prediction["draw_prob"])
        + float(prediction["away_win_prob"])
    )
    assert model_bundle["start_year"] == V3_MATCH_START_YEAR
    assert float(prediction["lambda_home"]) > 0.0
    assert float(prediction["lambda_away"]) > 0.0
    assert abs(total_probability - 1.0) < 1e-9


def test_predict_knockout_matchup_v3_returns_valid_winner_and_probability():
    home = load_home_module()
    base_df, _, lead_in_df, _ = home.load_data()
    feature_df = build_v3_team_feature_table(base_df, lead_in_df, reference_date_or_edition=2026, match_window=4)
    feature_lookup = feature_df.set_index("team_id").to_dict("index")
    model_bundle = fit_v3_poisson_models(match_window=4)

    prediction = predict_knockout_matchup_v3(
        str(feature_df.iloc[0]["team_id"]),
        str(feature_df.iloc[1]["team_id"]),
        feature_lookup,
        model_bundle,
        simulations=80,
        seed=17,
    )

    assert prediction["winner_team_id"] in {str(feature_df.iloc[0]["team_id"]), str(feature_df.iloc[1]["team_id"])}
    assert 50.0 <= float(prediction["winner_win_prob"]) <= 100.0
    assert abs(float(prediction["home_win_prob"]) + float(prediction["away_win_prob"]) - 100.0) < 1e-9


def test_fit_v2_model_predicts_valid_probability_triplet():
    home = load_home_module()
    base_df, _, lead_in_df, _ = home.load_data()
    feature_df = build_v2_match_feature_table(base_df, lead_in_df, match_window=4)
    feature_lookup = feature_df.set_index("team_id").to_dict("index")
    model_bundle = fit_v2_match_multinomial_model(match_window=4)

    assert set(model_bundle["model"].classes_) == {"away_win", "draw", "home_win"}
    assert model_bundle["edition_lookback"] == V2_PREVIOUS_EDITION_LOOKBACK

    first_team_id = str(feature_df.iloc[0]["team_id"])
    second_team_id = str(feature_df.iloc[1]["team_id"])
    prediction = predict_match_probabilities_v2(first_team_id, second_team_id, feature_lookup, model_bundle)

    total_probability = (
        float(prediction["home_win_prob"])
        + float(prediction["draw_prob"])
        + float(prediction["away_win_prob"])
    )
    assert abs(total_probability - 1.0) < 1e-9
    assert 0.0 <= float(prediction["home_win_prob"]) <= 1.0
    assert 0.0 <= float(prediction["draw_prob"]) <= 1.0
    assert 0.0 <= float(prediction["away_win_prob"]) <= 1.0


def test_build_2022_backtest_data_constructs_expected_tournament_shape():
    backtest_data = build_2022_backtest_data()
    base_df = pd.DataFrame(backtest_data["base_df"])
    fixtures_df = pd.DataFrame(backtest_data["fixtures_df"])

    assert len(base_df) == 32
    assert set(base_df["group_code"]) == set("ABCDEFGH")
    assert (base_df.groupby("group_code").size() == 4).all()
    assert len(fixtures_df[fixtures_df["round_code"] == "GS"]) == 48
    assert len(fixtures_df[fixtures_df["round_code"].isin(["R16", "QF", "SF", "3P", "F"])]) == 16

    knockout_labels = fixtures_df.set_index("match_number")[["home_slot_label", "away_slot_label"]].to_dict("index")
    assert knockout_labels[49] == {"home_slot_label": "1A", "away_slot_label": "2B"}
    assert knockout_labels[57] == {"home_slot_label": "W53", "away_slot_label": "W54"}
    assert knockout_labels[61] == {"home_slot_label": "W57", "away_slot_label": "W58"}
    assert knockout_labels[64] == {"home_slot_label": "W61", "away_slot_label": "W62"}


def test_historical_backtest_builder_derives_non_2022_fixture_shape():
    backtest_data = build_historical_world_cup_backtest_data(2018)
    base_df = pd.DataFrame(backtest_data["base_df"])
    fixtures_df = pd.DataFrame(backtest_data["fixtures_df"])

    assert len(base_df) == 32
    assert set(base_df["group_code"]) == set("ABCDEFGH")
    assert len(fixtures_df[fixtures_df["round_code"] == "GS"]) == 48
    assert len(fixtures_df[fixtures_df["round_code"].isin(["R16", "QF", "SF", "3P", "F"])]) == 16
    assert fixtures_df.loc[fixtures_df["round_code"].eq("R16"), "home_slot_label"].str.match(r"^[12][A-H]$").all()
    assert fixtures_df.loc[fixtures_df["round_code"].eq("R16"), "away_slot_label"].str.match(r"^[12][A-H]$").all()


def test_v4_stage_multiplier_cutoff_excludes_holdout_rows(monkeypatch):
    historical_results = pd.DataFrame(
        [
            {"edition": 2018, "stage": "Group Stage", "home_score": 1, "away_score": 1},
            {"edition": 2022, "stage": "Group Stage", "home_score": 10, "away_score": 10},
        ]
    )
    monkeypatch.setattr(v4_module, "load_historical_world_cup_results", lambda exclude_editions=(): historical_results)

    multipliers = compute_v4_stage_multipliers(cutoff_year=2022)

    assert abs(multipliers["group"] - 1.0) < 1e-12


def test_v4_stage_key_distinguishes_round_of_32_and_round_of_16():
    assert v4_stage_key("R32") == "round_of_32"
    assert v4_stage_key("Round of 32") == "round_of_32"
    assert v4_stage_key("R16") == "round_of_16"


def test_predict_knockout_matchup_v2_returns_valid_winner_and_probability():
    home = load_home_module()
    base_df, _, lead_in_df, _ = home.load_data()
    feature_df = build_v2_match_feature_table(base_df, lead_in_df, match_window=4)
    feature_lookup = feature_df.set_index("team_id").to_dict("index")
    model_bundle = fit_v2_match_multinomial_model(match_window=4)

    prediction = predict_knockout_matchup_v2(
        str(feature_df.iloc[0]["team_id"]),
        str(feature_df.iloc[1]["team_id"]),
        feature_lookup,
        model_bundle,
        simulations=120,
        seed=17,
    )

    assert prediction["winner_team_id"] in {str(feature_df.iloc[0]["team_id"]), str(feature_df.iloc[1]["team_id"])}
    assert 50.0 <= float(prediction["winner_win_prob"]) <= 100.0
    assert abs(float(prediction["home_win_prob"]) + float(prediction["away_win_prob"]) - 100.0) < 1e-9


def test_simulate_group_probabilities_v2_preserves_probability_invariants():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()

    dashboard_df = simulate_group_probabilities_v2(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=24,
        match_window=4,
    )

    required_columns = {
        "top8_third_prob",
        "ko_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "champion_prob",
    }
    assert required_columns.issubset(dashboard_df.columns)
    assert "team_strength" in dashboard_df.columns

    for _, row in dashboard_df.iterrows():
        total_probability = row["prob_1"] + row["prob_2"] + row["prob_3"] + row["prob_4"]
        assert abs(total_probability - 100.0) < 1e-9
        assert abs(row["ko_prob"] - (row["prob_1"] + row["prob_2"] + row["top8_third_prob"])) < 1e-9
        assert row["champion_prob"] <= row["final_prob"] + 1e-9
        assert row["final_prob"] <= row["sf_prob"] + 1e-9
        assert row["sf_prob"] <= row["qf_prob"] + 1e-9
        assert row["qf_prob"] <= row["r16_prob"] + 1e-9
        assert row["r16_prob"] <= row["ko_prob"] + 1e-9
        for column_name in required_columns:
            assert 0.0 <= row[column_name] <= 100.0

    modal_rankings = get_modal_group_rankings(dashboard_df)
    assert set(modal_rankings) == set(home.GROUP_ORDER)


def test_simulate_group_probabilities_v3_preserves_probability_invariants():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()

    dashboard_df = simulate_group_probabilities_v3(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=12,
        match_window=4,
    )

    required_columns = {
        "top8_third_prob",
        "ko_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "champion_prob",
    }
    assert required_columns.issubset(dashboard_df.columns)
    assert "team_strength" in dashboard_df.columns

    for _, row in dashboard_df.iterrows():
        total_probability = row["prob_1"] + row["prob_2"] + row["prob_3"] + row["prob_4"]
        assert abs(total_probability - 100.0) < 1e-9
        assert abs(row["ko_prob"] - (row["prob_1"] + row["prob_2"] + row["top8_third_prob"])) < 1e-9
        assert row["champion_prob"] <= row["final_prob"] + 1e-9
        assert row["final_prob"] <= row["sf_prob"] + 1e-9
        assert row["sf_prob"] <= row["qf_prob"] + 1e-9
        assert row["qf_prob"] <= row["r16_prob"] + 1e-9
        assert row["r16_prob"] <= row["ko_prob"] + 1e-9

    modal_rankings = get_modal_group_rankings(dashboard_df)
    assert set(modal_rankings) == set(home.GROUP_ORDER)


def test_build_deterministic_bracket_v2_produces_consistent_field():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()
    dashboard_df = simulate_group_probabilities_v2(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=20,
        match_window=4,
    )
    model_bundle = fit_v2_match_multinomial_model(match_window=4)

    bracket = build_deterministic_bracket_v2(
        dashboard_df,
        fixtures_df,
        dashboard_df,
        model_bundle,
        head_to_head_simulations=60,
        seed=19,
    )

    assert [round_data["round_code"] for round_data in bracket["rounds"]] == ["R32", "R16", "QF", "SF", "F"]
    assert sum(len(round_data["matches"]) for round_data in bracket["rounds"]) == 31
    assert bracket["qualifying_third_place_groups"] in THIRD_PLACE_ROUTING_MAP


def test_build_deterministic_bracket_v3_produces_consistent_field():
    home = load_home_module()
    base_df, fixtures_df, lead_in_df, _ = home.load_data()
    dashboard_df = simulate_group_probabilities_v3(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=12,
        match_window=4,
    )
    model_bundle = fit_v3_poisson_models(match_window=4)

    bracket = build_deterministic_bracket_v3(
        dashboard_df,
        fixtures_df,
        dashboard_df,
        model_bundle,
        head_to_head_simulations=40,
        seed=19,
    )

    assert [round_data["round_code"] for round_data in bracket["rounds"]] == ["R32", "R16", "QF", "SF", "F"]
    assert sum(len(round_data["matches"]) for round_data in bracket["rounds"]) == 31
    assert bracket["qualifying_third_place_groups"] in THIRD_PLACE_ROUTING_MAP


def test_simulate_group_probabilities_v2_32team_preserves_probability_invariants():
    backtest_data = build_2022_backtest_data()
    dashboard_df = simulate_group_probabilities_v2_32team(
        base_df=pd.DataFrame(backtest_data["base_df"]),
        fixtures_df=pd.DataFrame(backtest_data["fixtures_df"]),
        lead_in_df=pd.DataFrame(backtest_data["lead_in_df"]),
        simulations=20,
        match_window=4,
        exclude_editions=(2022,),
    )

    for _, row in dashboard_df.iterrows():
        total_probability = row["prob_1"] + row["prob_2"] + row["prob_3"] + row["prob_4"]
        assert abs(total_probability - 100.0) < 1e-9
        assert row["champion_prob"] <= row["final_prob"] + 1e-9
        assert row["final_prob"] <= row["sf_prob"] + 1e-9
        assert row["sf_prob"] <= row["qf_prob"] + 1e-9
        assert row["qf_prob"] <= row["r16_prob"] + 1e-9


def test_build_deterministic_bracket_v2_32team_produces_consistent_field():
    backtest_data = build_2022_backtest_data()
    base_df = pd.DataFrame(backtest_data["base_df"])
    fixtures_df = pd.DataFrame(backtest_data["fixtures_df"])
    lead_in_df = pd.DataFrame(backtest_data["lead_in_df"])
    simulation_df = simulate_group_probabilities_v2_32team(
        base_df=base_df,
        fixtures_df=fixtures_df,
        lead_in_df=lead_in_df,
        simulations=16,
        match_window=4,
        exclude_editions=(2022,),
    )
    feature_df = build_v2_match_feature_table(base_df, lead_in_df, match_window=4)
    model_bundle = fit_v2_match_multinomial_model(match_window=4, exclude_editions=(2022,))

    bracket = build_deterministic_bracket_v2_32team(
        simulation_df,
        fixtures_df,
        feature_df,
        model_bundle,
        head_to_head_simulations=40,
        seed=19,
    )

    assert [round_data["round_code"] for round_data in bracket["rounds"]] == ["R16", "QF", "SF", "F"]
    assert sum(len(round_data["matches"]) for round_data in bracket["rounds"]) == 15


def test_run_v2_backtest_2022_returns_valid_metrics_and_actual_champion():
    backtest = run_v2_backtest_2022(match_window=4, simulations=20, seed=17)
    summary_metrics = dict(backtest["summary_metrics"])
    team_backtest_table = pd.DataFrame(backtest["team_backtest_table"])
    match_predictions = pd.DataFrame(backtest["match_predictions"])

    assert 0.0 <= float(summary_metrics["multiclass_log_loss"])
    assert 0.0 <= float(summary_metrics["multiclass_brier_score"])
    assert 0.0 <= float(summary_metrics["top1_match_accuracy"]) <= 100.0
    assert int(summary_metrics["semifinal_hit_count"]) <= 4
    assert int(summary_metrics["round_of_16_hit_count"]) <= 16
    assert summary_metrics["actual_champion_team_id"] == "ARG"
    assert len(match_predictions) == 64

    argentina_row = team_backtest_table.loc[team_backtest_table["team_id"] == "ARG"].iloc[0]
    assert argentina_row["actual_stage"] == "Champion"
    assert argentina_row["champion_prob"] <= argentina_row["final_prob"] + 1e-9
    assert argentina_row["final_prob"] <= argentina_row["sf_prob"] + 1e-9
    assert argentina_row["sf_prob"] <= argentina_row["qf_prob"] + 1e-9
    assert argentina_row["qf_prob"] <= argentina_row["r16_prob"] + 1e-9


def test_run_v3_backtest_2022_returns_valid_metrics_and_actual_champion():
    backtest = run_v3_2022_backtest(match_window=4, simulations=12, seed=17)
    summary_metrics = dict(backtest["summary_metrics"])
    team_backtest_table = pd.DataFrame(backtest["team_backtest_table"])
    match_predictions = pd.DataFrame(backtest["match_predictions"])

    assert 0.0 <= float(summary_metrics["multiclass_log_loss"])
    assert 0.0 <= float(summary_metrics["multiclass_brier_score"])
    assert 0.0 <= float(summary_metrics["top1_match_accuracy"]) <= 100.0
    assert 0.0 <= float(summary_metrics["draw_rate_actual"]) <= 100.0
    assert 0.0 <= float(summary_metrics["draw_rate_predicted"]) <= 100.0
    assert int(summary_metrics["semifinal_hit_count"]) <= 4
    assert int(summary_metrics["round_of_16_hit_count"]) <= 16
    assert summary_metrics["actual_champion_team_id"] == "ARG"
    assert len(match_predictions) == 64

    argentina_row = team_backtest_table.loc[team_backtest_table["team_id"] == "ARG"].iloc[0]
    assert argentina_row["actual_stage"] == "Champion"
    assert argentina_row["champion_prob"] <= argentina_row["final_prob"] + 1e-9
    assert argentina_row["final_prob"] <= argentina_row["sf_prob"] + 1e-9
    assert argentina_row["sf_prob"] <= argentina_row["qf_prob"] + 1e-9
    assert argentina_row["qf_prob"] <= argentina_row["r16_prob"] + 1e-9


def test_model_validation_builder_returns_expected_models_and_numeric_metrics():
    artifacts = build_validation_artifacts(match_window=4, simulations=8, seed=17)
    model_rows = {row["model_id"]: row for row in artifacts["models"]}

    expected_models = {
        "baseline_elo",
        "v2_world_cup_only",
        "v2_all_international_since_anchor",
        "v3_world_cup_only",
        "v3_all_international_since_anchor",
        "v4_all_international_since_anchor",
    }
    assert set(model_rows) == expected_models
    for row in model_rows.values():
        for metric_name in METRIC_FIELDS:
            assert isinstance(float(row[metric_name]), float)
        assert row["holdout"] == "2022 FIFA World Cup"
        assert row["anchor_year"] == 1998
        assert row["anchor_date"] == "1998-06-10"
        assert int(row["training_match_count"]) > 0
        assert row["sample_weight_policy"] == SAMPLE_WEIGHT_POLICY

    match_predictions = pd.DataFrame(artifacts["match_predictions"])
    assert set(match_predictions["model_id"]) == expected_models
    assert match_predictions.groupby("model_id").size().eq(64).all()


def test_model_validation_training_excludes_2022_and_model_card_references_artifact():
    baseline = run_elo_baseline_2022(match_window=4, seed=17)

    assert baseline["training_metadata"]["training_scope"] == TRAINING_SCOPE_ALL_INTERNATIONAL
    assert baseline["training_metadata"]["training_end_date"] < "2022-11-20"

    artifacts = build_validation_artifacts(match_window=4, simulations=8, seed=17)
    assert artifacts["validation_window"]["holdout_year"] == 2022
    for row in artifacts["models"]:
        assert row["training_end_date"] < "2022-11-20"

    markdown = build_model_card_markdown(
        {
            "validation_window": artifacts["validation_window"],
            "models": artifacts["models"],
        }
    )
    assert "data/processed/validation/model_validation_2022.json" in markdown
    assert validation_artifact_filenames(2022)["json"] in markdown
    assert "all_international_since_anchor" in markdown
    assert "V4 all international since anchor" in markdown


def test_v2_probabilities_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "4_V2_Probabilities.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v2_probabilities_dashboard" in page_text


def test_v2_2022_backtest_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "5_V2_2022_Backtest.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v2_2022_backtest_dashboard" in page_text


def test_v3_probabilities_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "6_V3_Probabilities.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v3_probabilities_dashboard" in page_text


def test_v3_2022_backtest_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "7_V3_2022_Backtest.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v3_2022_backtest_dashboard" in page_text


def test_v4_quadratic_recency_weights_and_snapshot():
    weights = quadratic_recency_weights(4)

    assert weights.tolist() == [1.0, 4.0, 9.0, 16.0]

    results_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4),
            "result": ["loss", "draw", "win", "win"],
            "goal_difference": [-2, 0, 1, 3],
            "team_score": [0, 1, 2, 4],
            "opponent_score": [2, 1, 1, 1],
            "team_elo_start": [1500, 1510, 1520, 1530],
            "opponent_elo_start": [1500, 1500, 1500, 1500],
        }
    )
    snapshot = compute_quadratic_form_snapshot(results_df, match_window=4)

    assert abs(snapshot["results_form"] - ((0 * 1 + 0.5 * 4 + 1 * 9 + 1 * 16) / 30)) < 1e-9
    assert abs(snapshot["gd_form"] - ((-2 * 1 + 0 * 4 + 1 * 9 + 3 * 16) / 30)) < 1e-9
    assert snapshot["pre_tournament_elo"] == 1530


def test_v4_dixon_coles_score_matrix_and_penalties():
    lambda_home = 1.2
    lambda_away = 0.9
    rho = 0.1

    assert dixon_coles_tau(0, 0, lambda_home, lambda_away, rho) == 1 - lambda_home * lambda_away * rho
    assert dixon_coles_tau(0, 1, lambda_home, lambda_away, rho) == 1 + lambda_home * rho
    assert dixon_coles_tau(1, 0, lambda_home, lambda_away, rho) == 1 + lambda_away * rho
    assert dixon_coles_tau(1, 1, lambda_home, lambda_away, rho) == 1 - rho
    assert dixon_coles_tau(2, 2, lambda_home, lambda_away, rho) == 1.0

    corrected = build_v4_score_matrix(lambda_home, lambda_away, rho=rho)
    independent = build_v4_score_matrix(lambda_home, lambda_away, rho=0.0)

    assert abs(float(corrected.sum()) - 1.0) < 1e-12
    assert abs(float(independent.sum()) - 1.0) < 1e-12
    assert np.allclose(independent, build_v4_score_matrix(lambda_home, lambda_away, rho=0.0))
    assert strength_weighted_penalty_probability(1000, -1000) == 0.65
    assert strength_weighted_penalty_probability(-1000, 1000) == 0.35


def test_v4_dixon_coles_rho_fits_from_supplied_lambdas():
    training_df = pd.DataFrame(
        {
            "home_score": [0, 1, 2, 1],
            "away_score": [0, 1, 1, 0],
        }
    )
    lambda_home = np.array([1.0, 1.1, 1.3, 1.0])
    lambda_away = np.array([1.0, 1.1, 0.9, 0.8])

    rho, source = fit_v4_dixon_coles_rho(
        training_df,
        lambda_home,
        lambda_away,
        source="time_series_oof_grid_search",
    )

    assert source == "time_series_oof_grid_search"
    assert -0.2 <= rho <= 0.2


def test_v4_pages_exist_and_wire_dashboard_renderers():
    page_expectations = {
        "9_V4_Probabilities.py": "render_v4_probabilities_dashboard",
        "10_V4_2022_Backtest.py": "render_v4_2022_backtest_dashboard",
        "11_V4_Rolling_Backtest.py": "render_v4_rolling_backtest_dashboard",
    }
    for page_name, renderer_name in page_expectations.items():
        page_path = ROOT / "apps" / "pages" / page_name
        assert page_path.exists()
        assert renderer_name in page_path.read_text(encoding="utf-8")


def test_world_cup_simulation_facade_exports_representative_symbols():
    import world_cup_simulation as simulation

    assert simulation.MODEL_VERSION == "v1"
    assert simulation.V2_MODEL_VERSION == "v2"
    assert simulation.V3_MODEL_VERSION == "v3"
    assert simulation.V4_MODEL_VERSION == "v4"
    assert callable(simulation.simulate_group_probabilities)
    assert callable(simulation.simulate_group_probabilities_v2)
    assert callable(simulation.simulate_group_probabilities_v3)
    assert callable(simulation.simulate_group_probabilities_v4)
    assert callable(simulation.fit_v2_match_multinomial_model)
    assert callable(simulation.fit_v3_poisson_models)
    assert callable(simulation.fit_v4_poisson_models)
