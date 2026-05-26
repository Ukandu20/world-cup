from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.dashboard.config import (  # noqa: E402
    BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    DEFAULT_RECENT_MATCH_WINDOW,
    DEFAULT_V3_TRAINING_SCOPE,
    DEFAULT_V4_TRAINING_SCOPE,
    SIMULATION_COUNT,
    V3_MODEL_VERSION,
    V4_MODEL_VERSION,
)
from apps.dashboard.data import load_data  # noqa: E402
from apps.dashboard.projection_jobs import (  # noqa: E402
    build_v3_probability_artifact,
    build_v4_probability_artifact,
)
from apps.dashboard.simulation_store import (  # noqa: E402
    DEFAULT_SIMULATION_SEED,
    ArtifactSettings,
    artifact_dir,
    save_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prewarm official dashboard simulation artifacts.")
    parser.add_argument("--model", choices=("v3", "v4", "all"), default="all")
    parser.add_argument("--simulations", type=int, default=SIMULATION_COUNT)
    parser.add_argument("--match-window", type=int, default=DEFAULT_RECENT_MATCH_WINDOW)
    parser.add_argument(
        "--training-scope",
        default=None,
        help="Override training scope for all selected models. Defaults to each model's configured scope.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing official artifacts.")
    return parser.parse_args()


def prewarm_model(
    model_id: str,
    *,
    base_df,
    fixtures_df,
    lead_in_df,
    data_build_date: str,
    simulations: int,
    match_window: int,
    training_scope: str,
    force: bool,
) -> Path:
    model_version = V3_MODEL_VERSION if model_id == "v3" else V4_MODEL_VERSION
    settings = ArtifactSettings(
        model_id=model_id,
        model_version=model_version,
        data_build_date=data_build_date,
        simulations=simulations,
        match_window=match_window,
        training_scope=training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    destination = artifact_dir(settings, "official")
    if destination.exists() and not force:
        print(f"Skipping existing {model_id.upper()} artifact: {destination}")
        return destination

    if model_id == "v3":
        payload = build_v3_probability_artifact(
            base_df,
            fixtures_df,
            lead_in_df,
            simulations=simulations,
            match_window=match_window,
            training_scope=training_scope,
            seed=DEFAULT_SIMULATION_SEED,
            bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        )
    else:
        payload = build_v4_probability_artifact(
            base_df,
            fixtures_df,
            lead_in_df,
            simulations=simulations,
            match_window=match_window,
            training_scope=training_scope,
            seed=DEFAULT_SIMULATION_SEED,
            bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
        )

    artifact = save_artifact(
        settings,
        dashboard_df=payload["dashboard_df"],
        bracket_data=payload["bracket_data"],
        metadata=payload.get("metadata", {}),
        tier="official",
    )
    print(f"Wrote {model_id.upper()} official artifact: {artifact.artifact_dir}")
    return artifact.artifact_dir


def main() -> None:
    args = parse_args()
    base_df, fixtures_df, lead_in_df, metadata = load_data()
    selected_models = ("v3", "v4") if args.model == "all" else (args.model,)
    for model_id in selected_models:
        default_scope = DEFAULT_V3_TRAINING_SCOPE if model_id == "v3" else DEFAULT_V4_TRAINING_SCOPE
        prewarm_model(
            model_id,
            base_df=base_df,
            fixtures_df=fixtures_df,
            lead_in_df=lead_in_df,
            data_build_date=str(metadata.get("build_date", "")),
            simulations=args.simulations,
            match_window=args.match_window,
            training_scope=args.training_scope or default_scope,
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()
