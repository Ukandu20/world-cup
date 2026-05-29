from __future__ import annotations

import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import (
    ALL_COUNTRIES_KNOCKOUT_COLUMNS,
    BRACKET_EXPORT_VIEWPORT_SIZE,
    DEFAULT_RECENT_MATCH_WINDOW,
    EXPORT_DIR,
    EXPORT_MAX_VIEWPORT_WIDTH,
    EXPORT_MIN_VIEWPORT_WIDTH,
    EXPORT_VIEWPORT_HEIGHT,
    GROUP_ORDER,
    SCREENSHOT_CHANNELS,
)
from .rendering import (
    all_teams_table_frame,
    current_form_view_tables,
    chart_subtitle,
    projected_group_table_frame,
    render_bracket_document,
    render_export_document,
)

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ExportArtifact:
    """Download-ready export artifact."""

    path: Path
    filename: str
    mime: str
    data: bytes


@dataclass(frozen=True)
class BatchExportArtifact(ExportArtifact):
    """Download-ready batch export artifact."""

    png_count: int


def build_export_artifact(path: Path, mime: str = "image/png") -> ExportArtifact:
    """Read an export file and return metadata suitable for st.download_button."""
    return ExportArtifact(path=path, filename=path.name, mime=mime, data=path.read_bytes())


def build_export_stem(filename_stem: str, export_suffix: str | None = None) -> str:
    """Build the export filename stem, adding a unique suffix when requested."""
    return filename_stem if not export_suffix else f"{filename_stem}_{export_suffix}"


def generate_export_suffix() -> str:
    """Generate a timestamp suffix so each export writes a fresh artifact."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def cleanup_export_artifacts(limit: int = 50) -> None:
    """Keep only the newest generated PNG/ZIP export artifacts."""
    if limit < 0 or not EXPORT_DIR.exists():
        return

    artifacts = [
        path
        for path in EXPORT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".zip"}
    ]
    artifacts.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_path in artifacts[limit:]:
        try:
            stale_path.unlink(missing_ok=True)
        except OSError:
            continue


def estimate_export_column_count(table: dict[str, object]) -> int:
    """Estimate the visible column count for one exported table."""
    table_kind = str(table.get("table_kind", "probability"))
    if table_kind == "form":
        frame = table.get("frame")
        if isinstance(frame, pd.DataFrame) and "v2_strength" in frame.columns:
            return 22
        return 18

    column_count = 7
    if bool(table.get("include_group_column")):
        column_count += 2
    if bool(table.get("include_ko_column")):
        column_count += len(ALL_COUNTRIES_KNOCKOUT_COLUMNS)
    return column_count


def estimate_export_viewport_size(
    tables: list[dict[str, object]],
    multi_column: bool,
) -> str:
    """Estimate a screenshot viewport wide enough for the exported content."""
    if not tables:
        return f"{EXPORT_MIN_VIEWPORT_WIDTH},{EXPORT_VIEWPORT_HEIGHT}"

    if multi_column:
        visible_columns = min(3, max(1, len(tables)))
        width = 700 * visible_columns + 120
    else:
        column_count = max(estimate_export_column_count(table) for table in tables)
        width = 560 + max(0, column_count - 1) * 104

    width = max(EXPORT_MIN_VIEWPORT_WIDTH, min(EXPORT_MAX_VIEWPORT_WIDTH, width))
    return f"{width},{EXPORT_VIEWPORT_HEIGHT}"


def export_document_png(
    filename_stem: str,
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
    export_suffix: str | None = None,
) -> Path:
    """Export a complete standalone HTML view as a PNG screenshot."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = render_export_document(
        page_title,
        tables,
        multi_column,
        separate_sections=separate_sections,
    )
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    output_path = EXPORT_DIR / f"{output_stem}.png"

    with tempfile.TemporaryDirectory(prefix="wc_export_", ignore_cleanup_errors=True) as temp_dir:
        temp_html_path = Path(temp_dir) / f"{output_stem}.html"
        temp_html_path.write_text(document, encoding="utf-8")
        page_url = temp_html_path.resolve().as_uri()
        viewport_size = estimate_export_viewport_size(tables, multi_column=multi_column)

        last_error = ""
        for channel in SCREENSHOT_CHANNELS:
            command = build_screenshot_command(
                page_url,
                output_path,
                channel,
                viewport_size=viewport_size,
            )
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return output_path
            except FileNotFoundError:
                last_error = "Playwright was not found on PATH. Install Playwright and a supported browser to use PNG export."
                break
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or exc.stdout or str(exc)).strip()

        raise RuntimeError(f"PNG export failed: {last_error}")


def build_screenshot_command(
    page_url: str,
    output_path: Path,
    channel: str,
    viewport_size: str | None = None,
) -> list[str]:
    """Build the Playwright screenshot command, optionally forcing a viewport size."""
    command = [
        "playwright.exe",
        "screenshot",
        "--full-page",
        "--wait-for-timeout",
        "1500",
        "--channel",
        channel,
    ]
    if viewport_size:
        command.extend(["--viewport-size", viewport_size])
    command.extend([page_url, str(output_path)])
    return command


def export_bracket_png(
    filename_stem: str,
    page_title: str,
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
    simulation_count: int | None = None,
    export_suffix: str | None = None,
) -> Path:
    """Export the deterministic bracket view as a PNG screenshot."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = render_bracket_document(
        page_title,
        bracket_data,
        metadata_lookup,
        simulation_count=simulation_count,
    )
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    output_path = EXPORT_DIR / f"{output_stem}.png"

    with tempfile.TemporaryDirectory(prefix="wc_export_", ignore_cleanup_errors=True) as temp_dir:
        temp_html_path = Path(temp_dir) / f"{output_stem}.html"
        temp_html_path.write_text(document, encoding="utf-8")
        page_url = temp_html_path.resolve().as_uri()

        last_error = ""
        for channel in SCREENSHOT_CHANNELS:
            command = build_screenshot_command(
                page_url,
                output_path,
                channel,
                viewport_size=BRACKET_EXPORT_VIEWPORT_SIZE,
            )
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return output_path
            except FileNotFoundError:
                last_error = "Playwright was not found on PATH. Install Playwright and a supported browser to use PNG export."
                break
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or exc.stdout or str(exc)).strip()

        raise RuntimeError(f"PNG export failed: {last_error}")


def export_current_view(
    view_mode: str,
    selected_group: str,
    tables: list[dict[str, object]],
    bracket_data: dict[str, object] | None = None,
    metadata_lookup: dict[str, dict[str, str]] | None = None,
    simulation_count: int | None = None,
) -> ExportArtifact:
    """Export the currently visible dashboard view as one PNG file."""
    export_suffix = generate_export_suffix()
    if view_mode == "Single group":
        export_path = export_document_png(
            f"group_{selected_group.lower()}_view",
            f"Group {selected_group} View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    elif view_mode == "All groups":
        export_path = export_document_png(
            "all_groups_view",
            "All Groups View",
            tables,
            multi_column=True,
            export_suffix=export_suffix,
        )
    elif view_mode == "Bracket":
        if bracket_data is None or metadata_lookup is None:
            raise ValueError("Bracket export requires bracket_data and metadata_lookup")
        export_path = export_bracket_png(
            "bracket_view",
            "Bracket View",
            bracket_data,
            metadata_lookup,
            simulation_count=simulation_count,
            export_suffix=export_suffix,
        )
    elif view_mode == "Form":
        export_path = export_document_png(
            "form_view",
            "Form View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    else:
        export_path = export_document_png(
            "all_countries_view",
            "All Countries View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )

    cleanup_export_artifacts()
    return build_export_artifact(export_path)


def export_table_view(
    filename_stem: str,
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
) -> ExportArtifact:
    """Export one table view and return a download-ready PNG artifact."""
    export_path = export_document_png(
        filename_stem,
        page_title,
        tables,
        multi_column=multi_column,
        separate_sections=separate_sections,
        export_suffix=generate_export_suffix(),
    )
    cleanup_export_artifacts()
    return build_export_artifact(export_path)


def create_export_zip(exported_paths: list[Path], filename_stem: str, export_suffix: str) -> BatchExportArtifact:
    """Package generated PNG exports into one flat ZIP artifact."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = EXPORT_DIR / f"{build_export_stem(filename_stem, export_suffix)}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for export_path in exported_paths:
            archive.write(export_path, arcname=export_path.name)
    artifact = build_export_artifact(zip_path, mime="application/zip")
    return BatchExportArtifact(
        path=artifact.path,
        filename=artifact.filename,
        mime=artifact.mime,
        data=artifact.data,
        png_count=len(exported_paths),
    )


def export_all_tables(
    probability_df: pd.DataFrame | None = None,
    form_df: pd.DataFrame | None = None,
    simulation_count: int | None = None,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
    progress_callback: ProgressCallback | None = None,
    zip_filename_stem: str = "dashboard_exports",
) -> BatchExportArtifact:
    """Export the probability tables and optionally the form table as one ZIP file."""
    exported_paths: list[Path] = []
    export_suffix = generate_export_suffix()
    export_jobs: list[tuple[str, Callable[[], Path]]] = []

    if probability_df is not None:
        for group_code in GROUP_ORDER:
            group_df = projected_group_table_frame(probability_df, group_code)
            if group_df.empty:
                continue
            export_jobs.append(
                (
                    f"Group {group_code}",
                    lambda group_code=group_code, group_df=group_df: export_document_png(
                        f"group_{group_code.lower()}",
                        f"Group {group_code}",
                        [
                            {
                                "title": f"Group {group_code}",
                                "frame": group_df,
                                "include_group_column": False,
                                "include_ko_column": False,
                                "card_subtitle": chart_subtitle("Bracket-Aligned Projected Order", simulation_count),
                                "group_pill_label": group_code,
                                "table_kind": "probability",
                            }
                        ],
                        multi_column=False,
                        export_suffix=export_suffix,
                    ),
                )
            )

        combined = all_teams_table_frame(probability_df)
        export_jobs.append(
            (
                "All Countries",
                lambda combined=combined: export_document_png(
                    "all_countries",
                    "All Countries",
                    [
                        {
                            "title": "All Countries",
                            "frame": combined,
                            "include_group_column": True,
                            "include_ko_column": True,
                            "card_subtitle": chart_subtitle("Pre-Tournament Probability Table", simulation_count),
                            "group_pill_label": None,
                            "table_kind": "probability",
                        }
                    ],
                    multi_column=False,
                    export_suffix=export_suffix,
                ),
            )
        )
    if form_df is not None:
        all_countries_tables = current_form_view_tables(
            form_df,
            "All Countries",
            "",
            form_match_window=form_match_window,
        )
        all_confederations_tables = current_form_view_tables(
            form_df,
            "All confederations",
            "",
            form_match_window=form_match_window,
        )
        export_jobs.append(
            (
                "Form All Countries",
                lambda all_countries_tables=all_countries_tables: export_document_png(
                    "form_all_countries",
                    "All Countries",
                    all_countries_tables,
                    multi_column=False,
                    export_suffix=export_suffix,
                ),
            )
        )
        export_jobs.append(
            (
                "Form All Confederations",
                lambda all_confederations_tables=all_confederations_tables: export_document_png(
                    "form_all_confederations",
                    "All Confederations",
                    all_confederations_tables,
                    multi_column=False,
                    export_suffix=export_suffix,
                ),
            )
        )
        for table in all_confederations_tables:
            export_jobs.append(
                (
                    str(table["title"]),
                    lambda table=table: export_document_png(
                        str(table["stem"]),
                        str(table["title"]),
                        [table],
                        multi_column=False,
                        export_suffix=export_suffix,
                    ),
                )
            )

    total = len(export_jobs)
    for index, (label, export_job) in enumerate(export_jobs, start=1):
        if progress_callback is not None:
            progress_callback(index, total, label)
        exported_paths.append(export_job())

    artifact = create_export_zip(exported_paths, zip_filename_stem, export_suffix)
    cleanup_export_artifacts()
    return artifact
