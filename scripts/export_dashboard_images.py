from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)

from apps.dashboard.config import (  # noqa: E402
    BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    DATA_DIR,
    DEFAULT_RECENT_MATCH_WINDOW,
    GROUP_ORDER,
    SIMULATION_COUNT,
)
from apps.dashboard.model_registry import MODEL_REGISTRY, PRIMARY_MODEL_ID  # noqa: E402
from apps.dashboard.rendering import current_view_tables, render_export_document  # noqa: E402
from apps.dashboard.simulation_store import (  # noqa: E402
    DEFAULT_SIMULATION_SEED,
    ArtifactSettings,
    load_official_artifact,
)

ExportView = Literal["all-countries", "group"]
OFFICIAL_MODEL_IDS = tuple(
    model_id for model_id, model in MODEL_REGISTRY.items() if model.supports_official_artifact
)


@dataclass(frozen=True)
class ExportJob:
    """One dashboard table screenshot export job."""

    view: ExportView
    output_path: Path
    title: str
    group: str = "A"


def slugify(value: str) -> str:
    """Return a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "dashboard"


def default_output_filename(view: ExportView, group: str = "A") -> str:
    """Return the default PNG filename for one export view."""
    if view == "all-countries":
        return "dashboard-probabilities.png"
    return f"dashboard-group-{slugify(group)}.png"


def build_export_jobs(view: str, output_dir: Path, group: str) -> list[ExportJob]:
    """Build the screenshot jobs requested by the CLI."""
    normalized_group = group.upper()
    if view == "default":
        return [
            ExportJob(
                view="all-countries",
                output_path=output_dir / default_output_filename("all-countries", normalized_group),
                title="World Cup 2026 Probability Table",
            ),
            ExportJob(
                view="group",
                output_path=output_dir / default_output_filename("group", normalized_group),
                title=f"World Cup 2026 Group {normalized_group} Probability Table",
                group=normalized_group,
            ),
        ]
    if view == "all-countries":
        return [
            ExportJob(
                view="all-countries",
                output_path=output_dir / default_output_filename("all-countries", normalized_group),
                title="World Cup 2026 Probability Table",
            )
        ]
    if view == "group":
        return [
            ExportJob(
                view="group",
                output_path=output_dir / default_output_filename("group", normalized_group),
                title=f"World Cup 2026 Group {normalized_group} Probability Table",
                group=normalized_group,
            )
        ]
    raise ValueError(f"Unsupported export view: {view}")


def tables_for_job(dashboard_df, job: ExportJob) -> tuple[list[dict[str, object]], bool, bool]:
    """Return dashboard table definitions and layout flags for one export job."""
    if job.view == "all-countries":
        return current_view_tables(
            dashboard_df,
            "All Countries",
            "",
            simulation_count=SIMULATION_COUNT,
        ), False, False
    return current_view_tables(
        dashboard_df,
        "Single group",
        job.group,
        simulation_count=SIMULATION_COUNT,
    ), False, False


def load_official_dashboard_frame(model_id: str):
    """Load the committed official probability artifact for a model."""
    if model_id not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_id}'. Choose one of: {choices}")
    model = MODEL_REGISTRY[model_id]
    if not model.supports_official_artifact:
        raise ValueError(f"Model '{model_id}' does not have a committed official artifact.")

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    settings = ArtifactSettings(
        model_id=model.model_id,
        model_version=model.model_version,
        data_build_date=str(manifest.get("build_date", "")),
        simulations=SIMULATION_COUNT,
        match_window=DEFAULT_RECENT_MATCH_WINDOW,
        training_scope=model.default_training_scope,
        seed=DEFAULT_SIMULATION_SEED,
        bracket_head_to_head_simulations=BRACKET_HEAD_TO_HEAD_SIMULATIONS,
    )
    loaded = load_official_artifact(settings)
    if loaded.artifact is None:
        warning = f" Warning: {'; '.join(loaded.warnings)}" if loaded.warnings else ""
        raise FileNotFoundError(f"No official artifact found for {model_id} with current settings.{warning}")
    return loaded.artifact.dashboard_df


def write_export_html(document: str, temp_dir: Path, job: ExportJob) -> Path:
    """Write a standalone export document for screenshot capture."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    html_path = temp_dir / f"{job.output_path.stem}.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path


def capture_png(html_path: Path, output_path: Path, width: int, height: int) -> None:
    """Capture a PNG screenshot for a standalone HTML export document."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Install development dependencies first: python -m pip install -r requirements-dev.txt") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            card = page.locator(".wc-card").first
            if card.count() > 0:
                card.screenshot(path=str(output_path))
            else:
                page.screenshot(path=str(output_path), full_page=True)
        finally:
            browser.close()


def render_job(dashboard_df, job: ExportJob, temp_dir: Path, width: int, height: int) -> Path:
    """Render and capture one export job."""
    tables, multi_column, separate_sections = tables_for_job(dashboard_df, job)
    document = render_export_document(
        job.title,
        tables,
        multi_column=multi_column,
        separate_sections=separate_sections,
    )
    html_path = write_export_html(document, temp_dir, job)
    capture_png(html_path, job.output_path, width, height)
    return job.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local dashboard probability table screenshots.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--model", default=PRIMARY_MODEL_ID, choices=sorted(OFFICIAL_MODEL_IDS))
    parser.add_argument("--view", choices=("default", "all-countries", "group"), default="default")
    parser.add_argument("--group", default="A", choices=GROUP_ORDER)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--temp-dir", type=Path, default=ROOT / ".tmp_exports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_df = load_official_dashboard_frame(args.model)
    jobs = build_export_jobs(args.view, args.output_dir, args.group)
    for job in jobs:
        output_path = render_job(dashboard_df, job, args.temp_dir, args.width, args.height)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
