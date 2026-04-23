from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

DEFAULT_GROUP_ORDER = tuple("ABCDEFGHIJKL")

BACKTEST_2022_GROUP_ORDER = tuple("ABCDEFGH")

MODEL_VERSION = "v1"

MODEL_LABEL = "Team-strength Monte Carlo"

MODEL_SUMMARY = "Elo + recent form -> expected goals -> Poisson simulation"

RECENT_MATCH_WINDOW = 10

RESULT_POINTS = {"win": 3, "draw": 1, "loss": 0}

BASELINE_RATING_WEIGHTS = (1.0, 0.0)

FORM_COMPONENT_WEIGHTS = (0.7, 0.3)

STRENGTH_BLEND_WEIGHTS = (0.5, 0.5)

WEIGHTED_FORM_COMPOSITE_WEIGHTS = (0.4, 0.25, 0.25, 0.10)

WEIGHTED_FORM_GOAL_DIFFERENCE_CAP = 4

WEIGHTED_FORM_GD_BOUNDS = (-4.0, 4.0)

WEIGHTED_FORM_ELO_BOUNDS = (-15.0, 15.0)

WEIGHTED_FORM_PERF_BOUNDS = (-0.5, 0.5)

V2_HISTORY_COMPONENT_WEIGHTS = (0.7, 0.3)

V2_STRENGTH_BLEND_WEIGHTS = (0.4, 0.4, 0.2)

WORLD_CUP_HISTORY_EDITION_COUNT = 22

V2_PREVIOUS_EDITION_LOOKBACK = 5

WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT = float(
    sum((edition_index + 1) ** 2 for edition_index in range(WORLD_CUP_HISTORY_EDITION_COUNT))
)

EXPECTED_GOALS_BASE = 1.20

EXPECTED_GOALS_SCALE = 0.40

EXPECTED_GOALS_MIN = 0.20

EXPECTED_GOALS_MAX = 3.00

FORM_SCHEDULE_DIFFICULTY_MIN = 1.0

FORM_SCHEDULE_DIFFICULTY_MAX = 5.0

FORM_SCHEDULE_DIFFICULTY_NEUTRAL = 3.0

BEST_THIRD_QUALIFICATION_SLOTS = 8

EXTRA_TIME_FACTOR = 1.0 / 3.0

THIRD_PLACE_ROUTE_MATCHES = (79, 85, 81, 74, 82, 77, 87, 80)

MAIN_BRACKET_ROUND_CODES = ("R32", "R16", "QF", "SF", "F")

BACKTEST_2022_MAIN_BRACKET_ROUND_CODES = ("R16", "QF", "SF", "F")

ROUND_CODE_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
    "F": "Final",
}

WORLD_CUP_ROOT = Path(__file__).resolve().parents[1] / "INT-World Cup" / "world_cup"

HISTORICAL_RESULTS_START_YEAR = 1998

HISTORICAL_RESULTS_END_YEAR = 2022

V2_MODEL_VERSION = "v2"

V2_MODEL_LABEL = "Multinomial Match Model"

V2_MODEL_SUMMARY = "Historical World Cup multinomial regression -> Monte Carlo simulation"

V2_OUTCOME_LABELS = ("home_win", "draw", "away_win")

V2_FEATURE_COLUMNS = (
    "elo_diff",
    "results_form_diff",
    "gd_form_diff",
    "perf_vs_exp_diff",
    "goals_for_diff",
    "goals_against_diff",
    "placement_diff",
    "appearance_diff",
)

V2_STAGE_GROUP = "group"

V2_STAGE_KNOCKOUT = "knockout"

V3_MODEL_VERSION = "v3"

V3_MODEL_LABEL = "Poisson Expected Goals Model"

V3_MODEL_SUMMARY = "Historical international Poisson regression -> Monte Carlo simulation"

V3_MATCH_START_YEAR = 1998

V3_COMPETITION_IMPORTANCE = {
    "world_cup_finals": 3.0,
    "continental_finals": 2.5,
    "qualifier": 2.0,
    "other_competitive": 1.5,
    "friendly": 1.0,
}

V3_FEATURE_COLUMNS = (
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
)

V3_POISSON_GOAL_CAP = 10

V3_LAMBDA_MIN = 0.05

V3_LAMBDA_MAX = 4.5

V3_2022_HOST_TEAM_IDS = frozenset({"QAT"})

V3_2026_HOST_TEAM_IDS = frozenset({"CAN", "MEX", "USA"})

DEFAULT_SCORELINE_BY_OUTCOME = {
    "home_win": (1, 0),
    "draw": (0, 0),
    "away_win": (0, 1),
}

HISTORICAL_TEAM_NAME_ALIASES = {
    "congo dr": "dr congo",
    "dem rep of congo": "dr congo",
    "dr congo": "dr congo",
    "east germany": "german dr",
    "fr yugoslavia": "serbia",
    "ir iran": "iran",
    "korea republic": "south korea",
    "serbia and montenegro": "serbia",
    "turkiye": "turkey",
    "usa": "united states",
    "zaire": "dr congo",
}

THIRD_PLACE_ROUTING_COMPRESSED = (
    "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"
)


def load_third_place_routing_map() -> dict[str, dict[int, str]]:
    """Decode the static Round of 32 routing map for best third-place teams."""
    payload = zlib.decompress(base64.b64decode(THIRD_PLACE_ROUTING_COMPRESSED)).decode("utf-8")
    raw_mapping = json.loads(payload)
    return {
        combo_key: {int(match_number): group_code for match_number, group_code in match_mapping.items()}
        for combo_key, match_mapping in raw_mapping.items()
    }


THIRD_PLACE_ROUTING_MAP = load_third_place_routing_map()


__all__ = [
    'DEFAULT_GROUP_ORDER',
    'BACKTEST_2022_GROUP_ORDER',
    'MODEL_VERSION',
    'MODEL_LABEL',
    'MODEL_SUMMARY',
    'RECENT_MATCH_WINDOW',
    'RESULT_POINTS',
    'BASELINE_RATING_WEIGHTS',
    'FORM_COMPONENT_WEIGHTS',
    'STRENGTH_BLEND_WEIGHTS',
    'WEIGHTED_FORM_COMPOSITE_WEIGHTS',
    'WEIGHTED_FORM_GOAL_DIFFERENCE_CAP',
    'WEIGHTED_FORM_GD_BOUNDS',
    'WEIGHTED_FORM_ELO_BOUNDS',
    'WEIGHTED_FORM_PERF_BOUNDS',
    'V2_HISTORY_COMPONENT_WEIGHTS',
    'V2_STRENGTH_BLEND_WEIGHTS',
    'WORLD_CUP_HISTORY_EDITION_COUNT',
    'V2_PREVIOUS_EDITION_LOOKBACK',
    'WORLD_CUP_HISTORY_TOTAL_EDITION_WEIGHT',
    'EXPECTED_GOALS_BASE',
    'EXPECTED_GOALS_SCALE',
    'EXPECTED_GOALS_MIN',
    'EXPECTED_GOALS_MAX',
    'FORM_SCHEDULE_DIFFICULTY_MIN',
    'FORM_SCHEDULE_DIFFICULTY_MAX',
    'FORM_SCHEDULE_DIFFICULTY_NEUTRAL',
    'BEST_THIRD_QUALIFICATION_SLOTS',
    'EXTRA_TIME_FACTOR',
    'THIRD_PLACE_ROUTE_MATCHES',
    'MAIN_BRACKET_ROUND_CODES',
    'BACKTEST_2022_MAIN_BRACKET_ROUND_CODES',
    'ROUND_CODE_LABELS',
    'WORLD_CUP_ROOT',
    'HISTORICAL_RESULTS_START_YEAR',
    'HISTORICAL_RESULTS_END_YEAR',
    'V2_MODEL_VERSION',
    'V2_MODEL_LABEL',
    'V2_MODEL_SUMMARY',
    'V2_OUTCOME_LABELS',
    'V2_FEATURE_COLUMNS',
    'V2_STAGE_GROUP',
    'V2_STAGE_KNOCKOUT',
    'V3_MODEL_VERSION',
    'V3_MODEL_LABEL',
    'V3_MODEL_SUMMARY',
    'V3_MATCH_START_YEAR',
    'V3_COMPETITION_IMPORTANCE',
    'V3_FEATURE_COLUMNS',
    'V3_POISSON_GOAL_CAP',
    'V3_LAMBDA_MIN',
    'V3_LAMBDA_MAX',
    'V3_2022_HOST_TEAM_IDS',
    'V3_2026_HOST_TEAM_IDS',
    'DEFAULT_SCORELINE_BY_OUTCOME',
    'HISTORICAL_TEAM_NAME_ALIASES',
    'THIRD_PLACE_ROUTING_COMPRESSED',
    'THIRD_PLACE_ROUTING_MAP',
    'load_third_place_routing_map'
]
