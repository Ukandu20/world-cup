from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import world_cup_simulation as simulation


MODEL_LABEL = simulation.MODEL_LABEL
MODEL_SUMMARY = simulation.MODEL_SUMMARY
MODEL_VERSION = simulation.MODEL_VERSION
V2_MODEL_LABEL = simulation.V2_MODEL_LABEL
V2_MODEL_SUMMARY = simulation.V2_MODEL_SUMMARY
V2_MODEL_VERSION = simulation.V2_MODEL_VERSION
V3_MODEL_LABEL = simulation.V3_MODEL_LABEL
V3_MODEL_SUMMARY = simulation.V3_MODEL_SUMMARY
V3_MODEL_VERSION = simulation.V3_MODEL_VERSION
DEFAULT_V2_TRAINING_SCOPE = simulation.DEFAULT_V2_TRAINING_SCOPE
DEFAULT_V3_TRAINING_SCOPE = simulation.DEFAULT_V3_TRAINING_SCOPE
TRAINING_SCOPE_ALL_INTERNATIONAL = simulation.TRAINING_SCOPE_ALL_INTERNATIONAL
TRAINING_SCOPE_WORLD_CUP_ONLY = simulation.TRAINING_SCOPE_WORLD_CUP_ONLY
build_deterministic_bracket = simulation.build_deterministic_bracket
build_deterministic_bracket_v2 = simulation.build_deterministic_bracket_v2
build_deterministic_bracket_v3 = simulation.build_deterministic_bracket_v3
build_v2_team_strengths = simulation.build_v2_team_strengths
build_v2_match_feature_table = simulation.build_v2_match_feature_table
build_v3_team_feature_table = simulation.build_v3_team_feature_table
build_weighted_form_table = simulation.build_weighted_form_table
fit_v2_match_multinomial_model = simulation.fit_v2_match_multinomial_model
fit_v3_poisson_models = simulation.fit_v3_poisson_models
FORM_SCHEDULE_DIFFICULTY_NEUTRAL = simulation.FORM_SCHEDULE_DIFFICULTY_NEUTRAL
get_modal_group_rankings = simulation.get_modal_group_rankings
run_v2_backtest_2022 = simulation.run_v2_backtest_2022
run_v3_2022_backtest = simulation.run_v3_2022_backtest
simulate_group_probabilities = simulation.simulate_group_probabilities
simulate_group_probabilities_v2 = simulation.simulate_group_probabilities_v2
simulate_group_probabilities_v3 = simulation.simulate_group_probabilities_v3
WEIGHTED_FORM_COMPOSITE_WEIGHTS = simulation.WEIGHTED_FORM_COMPOSITE_WEIGHTS

DATA_DIR = simulation.WORLD_CUP_ROOT / "2026"
EXPORT_DIR = ROOT / "assets" / "charts" / "generated"
WORLD_CUP_LOGO_PATH = ROOT / "assets" / "logos" / "world-cup" / "fifa-world-cup-2026.football.cc.svg"
CHAMPION_TROPHY_PATH = ROOT / "assets" / "logos" / "world-cup" / "Coupe-du-monde.svg"
SIMULATION_COUNT = 20000
SIMULATION_OPTIONS = {
    "250": 250,
    "500": 500,
    "1k": 1000,
    "5k": 5000,
    "10k": 10000,
    "20k": 20000,
    "100k": 100000,
}
DEFAULT_RECENT_MATCH_WINDOW = 10
DEFAULT_SIMULATION_LABEL = "20k"
GROUP_ORDER = list("ABCDEFGHIJKL")
VIEW_OPTIONS = ("Single group", "All groups", "All Countries", "Form", "Bracket")
SCREENSHOT_CHANNELS = ("chrome", "msedge")
CURRENT_HOLDER_TEAM_ID = "ARG"
BRACKET_HEAD_TO_HEAD_SIMULATIONS = 10000
BRACKET_EXPORT_VIEWPORT_SIZE = "1800,1200"
EXPORT_VIEWPORT_HEIGHT = 1400
EXPORT_MIN_VIEWPORT_WIDTH = 1400
EXPORT_MAX_VIEWPORT_WIDTH = 3200
FORM_WINDOW_MIN = 3
FORM_WINDOW_MAX = 20
FORM_CONFEDERATION_ORDER = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")
V1_VIEW_OPTIONS = ("Single group", "All groups", "All Countries", "Bracket")
V2_VIEW_OPTIONS = ("All Countries", "Single confederation", "All confederations")
V2_PROB_VIEW_OPTIONS = ("Single group", "All groups", "All Countries", "Bracket")
TRAINING_SCOPE_LABELS = {
    "World Cup only": TRAINING_SCOPE_WORLD_CUP_ONLY,
    "All international since anchor": TRAINING_SCOPE_ALL_INTERNATIONAL,
}
TRAINING_SCOPE_LABEL_BY_VALUE = {value: label for label, value in TRAINING_SCOPE_LABELS.items()}
V1_STATE_KEY = "simulation_settings_v1"
V2_STATE_KEY = "simulation_settings_v2"
V2_PROB_STATE_KEY = "simulation_settings_v2_prob"
V2_BACKTEST_2022_STATE_KEY = "simulation_settings_v2_backtest_2022"
V3_PROB_STATE_KEY = "simulation_settings_v3_prob"
V3_BACKTEST_2022_STATE_KEY = "simulation_settings_v3_backtest_2022"
PROBABILITY_PALETTES = {
    "prob_1": ((220, 252, 231), (22, 163, 74)),
    "prob_2": ((219, 234, 254), (37, 99, 235)),
    "prob_3": ((254, 243, 199), (217, 119, 6)),
    "prob_4": ((254, 226, 226), (220, 38, 38)),
    "top8_third_prob": ((250, 245, 200), (202, 138, 4)),
    "ko_prob": ((224, 242, 254), (8, 145, 178)),
    "r16_prob": ((224, 231, 255), (79, 70, 229)),
    "qf_prob": ((233, 213, 255), (147, 51, 234)),
    "sf_prob": ((255, 228, 230), (225, 29, 72)),
    "final_prob": ((255, 237, 213), (234, 88, 12)),
    "champion_prob": ((254, 240, 138), (202, 138, 4)),
}
FORM_RED_TEXT = "#791F1F"
FORM_AMBER_TEXT = "#633806"
FORM_GREEN_TEXT = "#173404"
FORM_RED_GRADIENT = ("#FCEBEB", "#F7C1C1", "#F09595", "#E24B4A", "#A32D2D")
FORM_AMBER_GRADIENT = ("#FAEEDA", "#FAC775", "#EF9F27", "#BA7517", "#854F0B")
FORM_GREEN_GRADIENT = ("#EAF3DE", "#C0DD97", "#97C459", "#639922", "#3B6D11")
ALL_COUNTRIES_KNOCKOUT_COLUMNS = (
    ("ko_prob", "KO %"),
    ("r16_prob", "R16 %"),
    ("qf_prob", "QF %"),
    ("sf_prob", "SF %"),
    ("final_prob", "Final %"),
    ("champion_prob", "Champion %"),
)

