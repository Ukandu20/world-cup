from __future__ import annotations

from dataclasses import dataclass

from .config import (
    DEFAULT_V2_TRAINING_SCOPE,
    DEFAULT_V3_TRAINING_SCOPE,
    DEFAULT_V4_TRAINING_SCOPE,
    MODEL_LABEL,
    MODEL_VERSION,
    V2_MODEL_LABEL,
    V2_MODEL_VERSION,
    V3_MODEL_LABEL,
    V3_MODEL_VERSION,
    V4_MODEL_LABEL,
    V4_MODEL_VERSION,
)


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Dashboard model metadata used by cache prewarming and reporting."""

    model_id: str
    model_version: str
    model_label: str
    default_training_scope: str
    artifact_builder_name: str | None
    simulator_name: str
    bracket_builder_name: str
    is_primary: bool = False
    supports_official_artifact: bool = True


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "v1": ModelRegistryEntry(
        model_id="v1",
        model_version=MODEL_VERSION,
        model_label=MODEL_LABEL,
        default_training_scope="runtime_only",
        artifact_builder_name=None,
        simulator_name="simulate_probabilities",
        bracket_builder_name="build_deterministic_bracket",
        supports_official_artifact=False,
    ),
    "v2": ModelRegistryEntry(
        model_id="v2",
        model_version=V2_MODEL_VERSION,
        model_label=V2_MODEL_LABEL,
        default_training_scope=DEFAULT_V2_TRAINING_SCOPE,
        artifact_builder_name="build_v2_probability_artifact",
        simulator_name="simulate_probabilities_v2_dashboard",
        bracket_builder_name="build_deterministic_bracket_v2",
    ),
    "v3": ModelRegistryEntry(
        model_id="v3",
        model_version=V3_MODEL_VERSION,
        model_label=V3_MODEL_LABEL,
        default_training_scope=DEFAULT_V3_TRAINING_SCOPE,
        artifact_builder_name="build_v3_probability_artifact",
        simulator_name="simulate_probabilities_v3_dashboard",
        bracket_builder_name="build_deterministic_bracket_v3",
    ),
    "v4": ModelRegistryEntry(
        model_id="v4",
        model_version=V4_MODEL_VERSION,
        model_label=V4_MODEL_LABEL,
        default_training_scope=DEFAULT_V4_TRAINING_SCOPE,
        artifact_builder_name="build_v4_probability_artifact",
        simulator_name="simulate_probabilities_v4_dashboard",
        bracket_builder_name="build_deterministic_bracket_v4",
        is_primary=True,
    ),
}

PRIMARY_MODEL_ID = next(model_id for model_id, entry in MODEL_REGISTRY.items() if entry.is_primary)
PRIMARY_MODEL = MODEL_REGISTRY[PRIMARY_MODEL_ID]
LEGACY_MODEL_IDS = tuple(model_id for model_id, entry in MODEL_REGISTRY.items() if not entry.is_primary)
