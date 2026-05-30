from __future__ import annotations

import subprocess
import shutil
import tempfile
import zipfile
from io import BytesIO
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
WKHTMLTOIMAGE_WINDOWS_PATHS = (
    Path("C:/Program Files/wkhtmltopdf/bin/wkhtmltoimage.exe"),
    Path("C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltoimage.exe"),
)


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


def build_export_artifact_from_bytes(filename: str, data: bytes, mime: str = "image/png") -> ExportArtifact:
    """Return metadata suitable for st.download_button without requiring a server-side file."""
    return ExportArtifact(path=Path(filename), filename=filename, mime=mime, data=data)


def build_html_export_artifact(output_stem: str, document: str) -> ExportArtifact:
    """Return a standalone HTML export when PNG rendering is unavailable."""
    return build_export_artifact_from_bytes(
        f"{output_stem}.html",
        document.encode("utf-8"),
        mime="text/html",
    )


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


def cleanup_temp_dir(temp_dir: Path | None) -> None:
    """Remove an export scratch directory without masking the original error."""
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)


def prepare_temporary_export_html(document: str, output_stem: str) -> tuple[Path, Path]:
    """Write export HTML to a scratch file, falling back if the OS temp dir is locked down."""
    errors: list[str] = []
    temp_roots: tuple[Path | None, ...] = (None, EXPORT_DIR / ".tmp")

    for temp_root in temp_roots:
        temp_dir: Path | None = None
        try:
            if temp_root is not None:
                temp_root.mkdir(parents=True, exist_ok=True)
                temp_dir = Path(tempfile.mkdtemp(prefix="wc_export_", dir=temp_root))
            else:
                temp_dir = Path(tempfile.mkdtemp(prefix="wc_export_"))
            temp_html_path = temp_dir / f"{output_stem}.html"
            temp_html_path.write_text(document, encoding="utf-8")
            return temp_dir, temp_html_path
        except OSError as exc:
            location = str(temp_root) if temp_root is not None else tempfile.gettempdir()
            errors.append(f"{location}: {exc}")
            cleanup_temp_dir(temp_dir)

    raise RuntimeError(
        "Export failed while preparing the temporary HTML file. "
        f"No writable scratch directory was available ({' | '.join(errors)})."
    )


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

    temp_dir, temp_html_path = prepare_temporary_export_html(document, output_stem)
    try:
        page_url = temp_html_path.resolve().as_uri()
        viewport_size = estimate_export_viewport_size(tables, multi_column=multi_column)

        return run_png_export(
            html_input=str(temp_html_path.resolve()),
            browser_input=page_url,
            output_path=output_path,
            viewport_size=viewport_size,
        )
    finally:
        cleanup_temp_dir(temp_dir)


def build_screenshot_command(
    page_url: str,
    output_path: Path,
    channel: str,
    viewport_size: str | None = None,
) -> list[str]:
    """Build the Playwright screenshot command, optionally forcing a viewport size."""
    command = [
        "playwright.exe" if shutil.which("playwright.exe") else "playwright",
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


def parse_viewport_size(viewport_size: str | None) -> tuple[int, int] | None:
    """Parse a Playwright-style viewport-size string into width/height integers."""
    if not viewport_size:
        return None
    width_text, height_text = viewport_size.split(",", maxsplit=1)
    return int(width_text), int(height_text)


def build_wkhtmltoimage_command(
    html_input: str,
    output_path: Path,
    viewport_size: str | None = None,
    executable: str = "wkhtmltoimage",
) -> list[str]:
    """Build a wkhtmltoimage command for rendering local export HTML to PNG."""
    command = [
        executable,
        "--format",
        "png",
        "--quality",
        "100",
        "--enable-local-file-access",
        "--javascript-delay",
        "500",
    ]
    parsed_viewport = parse_viewport_size(viewport_size)
    if parsed_viewport is not None:
        width, _height = parsed_viewport
        command.extend(["--width", str(width)])
    command.extend([html_input, str(output_path)])
    return command


def wkhtmltoimage_executables() -> list[str]:
    """Return executable candidates, including common Windows install paths outside PATH."""
    candidates: list[str] = []
    for executable in ("wkhtmltoimage", "wkhtmltoimage.exe"):
        resolved = shutil.which(executable)
        if resolved is not None:
            candidates.append(resolved)
        candidates.append(executable)

    candidates.extend(str(path) for path in WKHTMLTOIMAGE_WINDOWS_PATHS if path.exists())
    return list(dict.fromkeys(candidates))


def command_error(exc: subprocess.CalledProcessError) -> str:
    """Return the useful stderr/stdout text from a failed renderer command."""
    output = exc.stderr or exc.stdout or str(exc)
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return output.strip()


def render_png_with_wkhtmltoimage(document: str, viewport_size: str | None = None) -> bytes:
    """Render HTML to PNG bytes through wkhtmltoimage stdin/stdout."""
    last_missing: FileNotFoundError | None = None
    for executable in wkhtmltoimage_executables():
        command = build_wkhtmltoimage_command(
            "-",
            Path("-"),
            viewport_size=viewport_size,
            executable=executable,
        )
        try:
            result = subprocess.run(
                command,
                input=document.encode("utf-8"),
                check=True,
                capture_output=True,
            )
            if result.stdout:
                return result.stdout
            raise RuntimeError(f"{executable} did not return PNG bytes.")
        except FileNotFoundError as exc:
            last_missing = exc
            continue
        except subprocess.CalledProcessError:
            raise

    if last_missing is not None:
        raise last_missing
    raise FileNotFoundError("wkhtmltoimage was not found on PATH.")


def render_png_with_playwright(document: str, viewport_size: str | None = None) -> bytes:
    """Render HTML to PNG bytes through Playwright's Python API."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright was not installed.") from exc

    viewport = parse_viewport_size(viewport_size) or (EXPORT_MIN_VIEWPORT_WIDTH, EXPORT_VIEWPORT_HEIGHT)
    width, height = viewport
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(document, wait_until="networkidle")
            page.wait_for_timeout(500)
            return page.screenshot(full_page=True, type="png")
        finally:
            browser.close()


def render_png_bytes(document: str, viewport_size: str | None = None) -> bytes:
    """Render one export HTML document to PNG bytes without server-side scratch files."""
    renderer_errors: list[str] = []

    try:
        return render_png_with_wkhtmltoimage(document, viewport_size=viewport_size)
    except FileNotFoundError:
        renderer_errors.append(
            "wkhtmltoimage was not found on PATH. Install wkhtmltopdf/wkhtmltoimage to use the primary PNG exporter."
        )
    except subprocess.CalledProcessError as exc:
        renderer_errors.append(f"wkhtmltoimage: {command_error(exc)}")
    except RuntimeError as exc:
        renderer_errors.append(f"wkhtmltoimage: {exc}")

    try:
        return render_png_with_playwright(document, viewport_size=viewport_size)
    except Exception as exc:
        renderer_errors.append(f"Playwright Python renderer: {exc}")

    raise RuntimeError(f"PNG export failed: {' | '.join(renderer_errors)}")


def export_document_png_artifact(
    filename_stem: str,
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
    export_suffix: str | None = None,
) -> ExportArtifact:
    """Export a complete standalone HTML view as download-ready PNG bytes."""
    document = render_export_document(
        page_title,
        tables,
        multi_column,
        separate_sections=separate_sections,
    )
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    viewport_size = estimate_export_viewport_size(tables, multi_column=multi_column)
    try:
        data = render_png_bytes(document, viewport_size=viewport_size)
        return build_export_artifact_from_bytes(f"{output_stem}.png", data)
    except RuntimeError:
        return build_html_export_artifact(output_stem, document)


def export_bracket_png_artifact(
    filename_stem: str,
    page_title: str,
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
    simulation_count: int | None = None,
    export_suffix: str | None = None,
) -> ExportArtifact:
    """Export the deterministic bracket view as download-ready PNG bytes."""
    document = render_bracket_document(
        page_title,
        bracket_data,
        metadata_lookup,
        simulation_count=simulation_count,
    )
    output_stem = build_export_stem(filename_stem, export_suffix=export_suffix)
    try:
        data = render_png_bytes(document, viewport_size=BRACKET_EXPORT_VIEWPORT_SIZE)
        return build_export_artifact_from_bytes(f"{output_stem}.png", data)
    except RuntimeError:
        return build_html_export_artifact(output_stem, document)


def run_png_export(
    html_input: str,
    browser_input: str,
    output_path: Path,
    viewport_size: str | None = None,
) -> Path:
    """Render one export HTML document to PNG, preferring wkhtmltoimage over Playwright."""
    renderer_errors: list[str] = []

    wkhtmltoimage_found = False
    for executable in wkhtmltoimage_executables():
        command = build_wkhtmltoimage_command(
            html_input,
            output_path,
            viewport_size=viewport_size,
            executable=executable,
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return output_path
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            wkhtmltoimage_found = True
            renderer_errors.append(f"{executable}: {command_error(exc)}")
            break

    if not wkhtmltoimage_found:
        renderer_errors.append(
            "wkhtmltoimage was not found on PATH. Install wkhtmltopdf/wkhtmltoimage to use the primary PNG exporter."
        )

    for channel in SCREENSHOT_CHANNELS:
        command = build_screenshot_command(
            browser_input,
            output_path,
            channel,
            viewport_size=viewport_size,
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return output_path
        except FileNotFoundError:
            renderer_errors.append(
                "Playwright was not found on PATH. Install Playwright and a supported browser to use the fallback PNG exporter."
            )
            break
        except subprocess.CalledProcessError as exc:
            renderer_errors.append(f"Playwright {channel}: {command_error(exc)}")

    raise RuntimeError(f"PNG export failed: {' | '.join(renderer_errors)}")


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

    temp_dir, temp_html_path = prepare_temporary_export_html(document, output_stem)
    try:
        page_url = temp_html_path.resolve().as_uri()

        return run_png_export(
            html_input=str(temp_html_path.resolve()),
            browser_input=page_url,
            output_path=output_path,
            viewport_size=BRACKET_EXPORT_VIEWPORT_SIZE,
        )
    finally:
        cleanup_temp_dir(temp_dir)


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
        return export_document_png_artifact(
            f"group_{selected_group.lower()}_view",
            f"Group {selected_group} View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    elif view_mode == "All groups":
        return export_document_png_artifact(
            "all_groups_view",
            "All Groups View",
            tables,
            multi_column=True,
            export_suffix=export_suffix,
        )
    elif view_mode == "Bracket":
        if bracket_data is None or metadata_lookup is None:
            raise ValueError("Bracket export requires bracket_data and metadata_lookup")
        return export_bracket_png_artifact(
            "bracket_view",
            "Bracket View",
            bracket_data,
            metadata_lookup,
            simulation_count=simulation_count,
            export_suffix=export_suffix,
        )
    elif view_mode == "Form":
        return export_document_png_artifact(
            "form_view",
            "Form View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    else:
        return export_document_png_artifact(
            "all_countries_view",
            "All Countries View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )


def export_table_view(
    filename_stem: str,
    page_title: str,
    tables: list[dict[str, object]],
    multi_column: bool,
    separate_sections: bool = False,
) -> ExportArtifact:
    """Export one table view and return a download-ready PNG artifact."""
    return export_document_png_artifact(
        filename_stem,
        page_title,
        tables,
        multi_column=multi_column,
        separate_sections=separate_sections,
        export_suffix=generate_export_suffix(),
    )


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


def create_export_zip_from_artifacts(
    exported_artifacts: list[ExportArtifact],
    filename_stem: str,
    export_suffix: str,
) -> BatchExportArtifact:
    """Package generated PNG artifact bytes into one ZIP without server-side files."""
    filename = f"{build_export_stem(filename_stem, export_suffix)}.zip"
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in exported_artifacts:
            archive.writestr(artifact.filename, artifact.data)
    return BatchExportArtifact(
        path=Path(filename),
        filename=filename,
        mime="application/zip",
        data=buffer.getvalue(),
        png_count=len(exported_artifacts),
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
    exported_artifacts: list[ExportArtifact] = []
    export_suffix = generate_export_suffix()
    export_jobs: list[tuple[str, Callable[[], ExportArtifact]]] = []

    if probability_df is not None:
        for group_code in GROUP_ORDER:
            group_df = projected_group_table_frame(probability_df, group_code)
            if group_df.empty:
                continue
            export_jobs.append(
                (
                    f"Group {group_code}",
                    lambda group_code=group_code, group_df=group_df: export_document_png_artifact(
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
                lambda combined=combined: export_document_png_artifact(
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
                lambda all_countries_tables=all_countries_tables: export_document_png_artifact(
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
                lambda all_confederations_tables=all_confederations_tables: export_document_png_artifact(
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
                    lambda table=table: export_document_png_artifact(
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
        exported_artifacts.append(export_job())

    return create_export_zip_from_artifacts(exported_artifacts, zip_filename_stem, export_suffix)
