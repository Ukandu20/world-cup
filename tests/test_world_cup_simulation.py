from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_cup_simulation import (  # noqa: E402
    THIRD_PLACE_ROUTING_MAP,
    V2_PREVIOUS_EDITION_LOOKBACK,
    V3_MATCH_START_YEAR,
    WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT,
    build_2022_backtest_data,
    build_deterministic_bracket,
    build_deterministic_bracket_v2,
    build_deterministic_bracket_v2_32team,
    build_deterministic_bracket_v3,
    build_v2_match_feature_table,
    build_v2_team_strengths,
    build_v2_training_frame,
    build_v3_team_feature_table,
    build_v3_training_frame,
    build_weighted_form_table,
    build_team_strengths,
    build_recent_form_metrics,
    classify_competition_importance,
    compute_elo_expected_score,
    extract_group_stage_fixtures,
    fit_v2_match_multinomial_model,
    fit_v3_poisson_models,
    get_modal_group_rankings,
    normalize_weight_pair,
    predict_match_probabilities_v2,
    predict_knockout_matchup,
    predict_knockout_matchup_v2,
    predict_match_lambdas_v3,
    predict_knockout_matchup_v3,
    rank_best_third_place_teams,
    rank_group_standings,
    run_v2_backtest_2022,
    run_v3_2022_backtest,
    simulate_group_probabilities,
    simulate_group_probabilities_v2_32team,
    simulate_group_probabilities_v2,
    simulate_group_probabilities_v3,
)
from world_cup_sim.constants import WORLD_CUP_ROOT  # noqa: E402
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

    rows = report_card.build_identity_rows(team_row, "Winner")
    by_label = {row["label"]: row["value"] for row in rows}

    assert by_label["Best Finish"] == "Winner"
    assert by_label["Coach"] == "Pending data"
    assert by_label["Captain"] == "Pending data"


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


def test_export_all_tables_uses_single_column_all_confederations_export(monkeypatch):
    home = load_home_module()
    captured_calls = []

    def fake_export_document_png(
        filename_stem,
        page_title,
        tables,
        multi_column,
        separate_sections=False,
        export_suffix=None,
    ):
        captured_calls.append(
            {
                "filename_stem": filename_stem,
                "page_title": page_title,
                "tables": tables,
                "multi_column": multi_column,
                "separate_sections": separate_sections,
            }
        )
        return Path(f"{filename_stem}.png")

    monkeypatch.setattr(home, "export_document_png", fake_export_document_png)
    monkeypatch.setattr(home, "generate_export_suffix", lambda: "stamp")

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

    home.export_all_tables(form_df=form_df, form_match_window=10)

    all_confed_call = next(call for call in captured_calls if call["filename_stem"] == "form_all_confederations")
    assert all_confed_call["page_title"] == "All Confederations"
    assert all_confed_call["multi_column"] is False
    assert all_confed_call["separate_sections"] is False


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


def test_export_current_view_uses_bracket_export_when_selected(monkeypatch):
    home = load_home_module()
    captured = {}

    def fake_export_bracket_png(
        filename_stem,
        page_title,
        bracket_data,
        metadata_lookup,
        simulation_count=None,
        export_suffix=None,
    ):
        captured.update(
            {
                "filename_stem": filename_stem,
                "page_title": page_title,
                "bracket_data": bracket_data,
                "metadata_lookup": metadata_lookup,
                "simulation_count": simulation_count,
                "export_suffix": export_suffix,
            }
        )
        return Path("dummy.png")

    monkeypatch.setattr(home, "export_bracket_png", fake_export_bracket_png)

    result = home.export_current_view(
        "Bracket",
        "A",
        [],
        bracket_data={"rounds": []},
        metadata_lookup={},
        simulation_count=100000,
    )

    assert str(result) == "dummy.png"
    assert captured["filename_stem"] == "bracket_view"
    assert captured["page_title"] == "Bracket View"
    assert captured["simulation_count"] == 100000


def test_build_screenshot_command_supports_forced_viewport():
    home = load_home_module()

    command = home.build_screenshot_command(
        "file:///tmp/test.html",
        Path("test.png"),
        "chrome",
        viewport_size=home.BRACKET_EXPORT_VIEWPORT_SIZE,
    )

    assert "--viewport-size" in command
    viewport_index = command.index("--viewport-size")
    assert command[viewport_index + 1] == home.BRACKET_EXPORT_VIEWPORT_SIZE


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


def test_build_v2_training_frame_uses_previous_five_editions_and_includes_knockout_rows():
    training_df = build_v2_training_frame(match_window=4)

    assert set(training_df["edition"].astype(int)) == {2006, 2010, 2014, 2018, 2022}
    assert {"Group Stage", "Quarter-final", "Semi-final", "Final"}.issubset(set(training_df["stage"]))
    assert {"group", "knockout"} == set(training_df["stage_bucket"])
    assert set(training_df["outcome_label"]).issubset({"home_win", "draw", "away_win"})
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
    training_df = build_v2_training_frame(match_window=4, exclude_editions=(2022,))

    assert 2022 not in set(training_df["edition"].astype(int))
    assert set(training_df["edition"].astype(int)) == {2002, 2006, 2010, 2014, 2018}


def test_classify_competition_importance_uses_v3_scale():
    assert classify_competition_importance("FIFA World Cup") == 3.0
    assert classify_competition_importance("UEFA Euro") == 2.5
    assert classify_competition_importance("FIFA World Cup qualification") == 2.0
    assert classify_competition_importance("Friendly") == 1.0
    assert classify_competition_importance("Nehru Cup") == 1.5


def test_build_v3_training_frame_respects_cutoff_and_columns():
    results_df = pd.read_csv(ROOT / "data" / "results.csv")
    cutoff = "2002-06-30"

    training_df = build_v3_training_frame(
        results_df,
        match_window=4,
        start_year=1998,
        end_date=cutoff,
    )

    assert not training_df.empty
    assert str(training_df["date"].max().date()) <= cutoff
    assert training_df["date"].min().year >= 1998
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

    assert set(model_rows) == {"baseline_elo", "v2", "v3"}
    for row in model_rows.values():
        for metric_name in METRIC_FIELDS:
            assert isinstance(float(row[metric_name]), float)
        assert row["holdout"] == "2022 FIFA World Cup"

    match_predictions = pd.DataFrame(artifacts["match_predictions"])
    assert set(match_predictions["model_id"]) == {"baseline_elo", "v2", "v3"}
    assert match_predictions.groupby("model_id").size().eq(64).all()


def test_model_validation_training_excludes_2022_and_model_card_references_artifact():
    baseline = run_elo_baseline_2022(match_window=4, seed=17)

    assert 2022 not in set(baseline["training_editions"])

    artifacts = build_validation_artifacts(match_window=4, simulations=8, seed=17)
    v2_row = next(row for row in artifacts["models"] if row["model_id"] == "v2")
    assert 2022 not in set(v2_row["training_editions"])

    markdown = build_model_card_markdown(
        {
            "validation_window": artifacts["validation_window"],
            "models": artifacts["models"],
        }
    )
    assert "data/processed/validation/model_validation_2022.json" in markdown


def test_v2_probabilities_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "3_V2_Probabilities.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v2_probabilities_dashboard" in page_text


def test_v2_2022_backtest_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "4_V2_2022_Backtest.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v2_2022_backtest_dashboard" in page_text


def test_v3_probabilities_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "5_V3_Probabilities.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v3_probabilities_dashboard" in page_text


def test_v3_2022_backtest_page_exists_and_wires_home_renderer():
    page_path = ROOT / "apps" / "pages" / "6_V3_2022_Backtest.py"

    assert page_path.exists()
    page_text = page_path.read_text(encoding="utf-8")
    assert "render_v3_2022_backtest_dashboard" in page_text


def test_world_cup_simulation_facade_exports_representative_symbols():
    import world_cup_simulation as simulation

    assert simulation.MODEL_VERSION == "v1"
    assert simulation.V2_MODEL_VERSION == "v2"
    assert simulation.V3_MODEL_VERSION == "v3"
    assert callable(simulation.simulate_group_probabilities)
    assert callable(simulation.simulate_group_probabilities_v2)
    assert callable(simulation.simulate_group_probabilities_v3)
    assert callable(simulation.fit_v2_match_multinomial_model)
    assert callable(simulation.fit_v3_poisson_models)
