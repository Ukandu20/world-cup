from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "INT-World Cup" / "world_cup" / "2026"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_cup_simulation import (  # noqa: E402
    THIRD_PLACE_ROUTING_MAP,
    build_team_strengths,
    build_recent_form_metrics,
    extract_group_stage_fixtures,
    normalize_weight_pair,
    rank_best_third_place_teams,
    rank_group_standings,
    simulate_group_probabilities,
)


def load_home_module():
    spec = importlib.util.spec_from_file_location("world_cup_home", ROOT / "apps" / "home.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_default_simulation_settings_keep_only_simulations():
    home = load_home_module()

    defaults = home.default_simulation_settings()

    assert defaults == {
        "simulation_label": "250",
    }


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


def test_build_table_html_all_countries_includes_ko_column_only_when_requested():
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
                "r16_prob": 61.0,
                "qf_prob": 39.0,
                "sf_prob": 22.0,
                "final_prob": 12.0,
                "champion_prob": 7.0,
            }
        ]
    )

    html = home.build_table_html(sample_df, "All Countries", include_group_column=True, include_ko_column=True)

    assert "Top 8 3rd %" in html
    assert "KO %" in html
    assert "R16 %" in html
    assert "QF %" in html
    assert "SF %" in html
    assert "Final %" in html
    assert "Champion %" in html
    assert "Rank" in html
    assert "86.0%" in html


def test_build_table_html_all_countries_embeds_champion_trophy_icon():
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


def test_build_table_html_does_not_render_simulation_count_in_header():
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
    )

    assert "Simulations" not in html
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
    assert "r16_prob" in normalized.columns
    assert "qf_prob" in normalized.columns
    assert "sf_prob" in normalized.columns
    assert "final_prob" in normalized.columns
    assert "champion_prob" in normalized.columns
    assert normalized.loc[0, "top8_third_prob"] == 0.0
    assert normalized.loc[0, "ko_prob"] == 75.0


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
