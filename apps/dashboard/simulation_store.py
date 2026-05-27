from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from .config import ROOT

ArtifactTier = Literal["official", "runtime"]

DEFAULT_SIMULATION_SEED = 20260403
OFFICIAL_ARTIFACT_ROOT = ROOT / "data" / "processed" / "dashboard_simulations" / "official"
RUNTIME_ARTIFACT_ROOT = ROOT / ".cache" / "dashboard_simulations" / "runtime"
LOCK_TIMEOUT_SECONDS = 30.0
LOCK_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class ArtifactSettings:
    """Settings that uniquely identify one dashboard simulation artifact."""

    model_id: str
    model_version: str
    data_build_date: str
    simulations: int
    match_window: int
    training_scope: str
    seed: int
    bracket_head_to_head_simulations: int


@dataclass(frozen=True)
class ArtifactResult:
    """Loaded or freshly-created simulation artifact."""

    dashboard_df: pd.DataFrame
    bracket_data: dict[str, Any]
    metadata: dict[str, Any]
    source: ArtifactTier
    created_at_utc: str
    artifact_dir: Path


@dataclass(frozen=True)
class ArtifactLoadResult:
    """Artifact result plus non-fatal cache warnings."""

    artifact: ArtifactResult | None
    warnings: tuple[str, ...] = ()
    created: bool = False


def _json_default(value: Any) -> Any:
    """Convert common pandas/numpy scalars into JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def artifact_key(settings: ArtifactSettings) -> str:
    """Return a stable content key for one simulation settings bundle."""
    payload = {
        "bracket_head_to_head_simulations": int(settings.bracket_head_to_head_simulations),
        "data_build_date": str(settings.data_build_date),
        "match_window": int(settings.match_window),
        "model_id": str(settings.model_id),
        "model_version": str(settings.model_version),
        "seed": int(settings.seed),
        "simulations": int(settings.simulations),
        "training_scope": str(settings.training_scope),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def artifact_root(tier: ArtifactTier) -> Path:
    """Return the root directory for an artifact tier."""
    if tier == "official":
        return OFFICIAL_ARTIFACT_ROOT
    if tier == "runtime":
        return RUNTIME_ARTIFACT_ROOT
    raise ValueError(f"Unknown artifact tier: {tier}")


def artifact_dir(settings: ArtifactSettings, tier: ArtifactTier) -> Path:
    """Return the directory for one artifact in one tier."""
    return artifact_root(tier) / settings.model_id / artifact_key(settings)


@contextmanager
def _artifact_write_lock(directory: Path):
    """Serialize writes for one artifact key across concurrent Streamlit reruns."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.with_name(f".{directory.name}.lock")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    lock_fd: int | None = None
    while lock_fd is None:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for artifact write lock: {lock_path}")
            time.sleep(LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _required_paths(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / "probabilities.csv.gz",
        directory / "bracket.json",
        directory / "metadata.json",
    )


def _load_from_directory(directory: Path, tier: ArtifactTier) -> ArtifactResult:
    probabilities_path, bracket_path, metadata_path = _required_paths(directory)
    missing = [
        path.name
        for path in (probabilities_path, bracket_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing artifact file(s): {', '.join(missing)}")

    dashboard_df = pd.read_csv(probabilities_path, compression="gzip")
    bracket_data = json.loads(bracket_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    created_at_utc = str(metadata.get("created_at_utc", ""))
    return ArtifactResult(
        dashboard_df=dashboard_df,
        bracket_data=bracket_data,
        metadata=metadata,
        source=tier,
        created_at_utc=created_at_utc,
        artifact_dir=directory,
    )


def load_artifact(settings: ArtifactSettings) -> ArtifactLoadResult:
    """Load the newest matching artifact, preferring runtime over official."""
    warnings: list[str] = []
    for tier in ("runtime", "official"):
        directory = artifact_dir(settings, tier)
        if not directory.exists():
            continue
        try:
            return ArtifactLoadResult(artifact=_load_from_directory(directory, tier), warnings=tuple(warnings))
        except Exception as exc:  # noqa: BLE001 - cache corruption should not break the app
            warnings.append(f"Ignored corrupt {tier} simulation artifact at {directory}: {exc}")
    return ArtifactLoadResult(artifact=None, warnings=tuple(warnings))


def load_official_artifact(settings: ArtifactSettings) -> ArtifactLoadResult:
    """Load the matching official artifact without falling back to runtime caches."""
    directory = artifact_dir(settings, "official")
    if not directory.exists():
        return ArtifactLoadResult(artifact=None)
    try:
        return ArtifactLoadResult(artifact=_load_from_directory(directory, "official"))
    except Exception as exc:  # noqa: BLE001 - cache corruption should not break the app
        return ArtifactLoadResult(
            artifact=None,
            warnings=(f"Ignored corrupt official simulation artifact at {directory}: {exc}",),
        )


def save_artifact(
    settings: ArtifactSettings,
    dashboard_df: pd.DataFrame,
    bracket_data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    tier: ArtifactTier,
) -> ArtifactResult:
    """Persist a simulation artifact and return the saved result."""
    key = artifact_key(settings)
    directory = artifact_dir(settings, tier)
    temp_directory = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")

    created_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    saved_metadata = {
        **(metadata or {}),
        "artifact_key": key,
        "bracket_head_to_head_simulations": int(settings.bracket_head_to_head_simulations),
        "created_at_utc": created_at_utc,
        "data_build_date": str(settings.data_build_date),
        "match_window": int(settings.match_window),
        "model_id": str(settings.model_id),
        "model_version": str(settings.model_version),
        "seed": int(settings.seed),
        "simulations": int(settings.simulations),
        "source_tier": tier,
        "training_scope": str(settings.training_scope),
    }

    with _artifact_write_lock(directory):
        temp_directory.mkdir(parents=True, exist_ok=False)
        try:
            probabilities_path, bracket_path, metadata_path = _required_paths(temp_directory)
            dashboard_df.to_csv(probabilities_path, index=False, compression="gzip")
            bracket_path.write_text(
                json.dumps(bracket_data, default=_json_default, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            metadata_path.write_text(
                json.dumps(saved_metadata, default=_json_default, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if directory.exists():
                shutil.rmtree(directory)
            temp_directory.rename(directory)
        except Exception:
            if temp_directory.exists():
                shutil.rmtree(temp_directory, ignore_errors=True)
            raise
    return _load_from_directory(directory, tier)


def load_or_create_artifact(
    settings: ArtifactSettings,
    create_fn: Callable[[], dict[str, Any]],
    *,
    force_refresh: bool = False,
    write_tier: ArtifactTier = "runtime",
) -> ArtifactLoadResult:
    """Load an artifact or create and persist it when absent or force-refreshed."""
    warnings: list[str] = []
    if not force_refresh:
        loaded = load_artifact(settings)
        warnings.extend(loaded.warnings)
        if loaded.artifact is not None:
            return ArtifactLoadResult(artifact=loaded.artifact, warnings=tuple(warnings), created=False)

    created = create_fn()
    artifact = save_artifact(
        settings,
        dashboard_df=created["dashboard_df"],
        bracket_data=created["bracket_data"],
        metadata=created.get("metadata", {}),
        tier=write_tier,
    )
    return ArtifactLoadResult(artifact=artifact, warnings=tuple(warnings), created=True)
