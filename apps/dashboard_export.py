from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .dashboard_config import (
    ALL_COUNTRIES_KNOCKOUT_COLUMNS,
    BRACKET_EXPORT_VIEWPORT_SIZE,
    DEFAULT_RECENT_MATCH_WINDOW,
    EXPORT_DIR,
    EXPORT_MAX_VIEWPORT_WIDTH,
    EXPORT_MIN_VIEWPORT_WIDTH,
    EXPORT_VIEWPORT_HEIGHT,
    GROUP_ORDER,
    SCREENSHOT_CHANNELS,
    V1_VIEW_OPTIONS,
)
from .dashboard_rendering import (
    build_bracket_html,
    build_table_html,
    current_form_view_tables,
    current_view_tables,
    render_bracket_document,
    render_export_document,
)

def build_export_stem(filename_stem: str, export_suffix: str | None = None) -> str:
    """Build the export filename stem, adding a unique suffix when requested."""
    return filename_stem if not export_suffix else f"{filename_stem}_{export_suffix}"


def generate_export_suffix() -> str:
    """Generate a timestamp suffix so each export writes a fresh artifact."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


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

    with tempfile.TemporaryDirectory(prefix="wc_export_", dir=str(EXPORT_DIR)) as temp_dir:
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
                last_error = "playwright.exe was not found on PATH."
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

    with tempfile.TemporaryDirectory(prefix="wc_export_", dir=str(EXPORT_DIR)) as temp_dir:
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
                last_error = "playwright.exe was not found on PATH."
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
) -> Path:
    """Export the currently visible dashboard view as one PNG file."""
    export_suffix = generate_export_suffix()
    if view_mode == "Single group":
        return export_document_png(
            f"group_{selected_group.lower()}_view",
            f"Group {selected_group} View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    if view_mode == "All groups":
        return export_document_png(
            "all_groups_view",
            "All Groups View",
            tables,
            multi_column=True,
            export_suffix=export_suffix,
        )
    if view_mode == "Bracket":
        if bracket_data is None or metadata_lookup is None:
            raise ValueError("Bracket export requires bracket_data and metadata_lookup")
        return export_bracket_png(
            "bracket_view",
            "Bracket View",
            bracket_data,
            metadata_lookup,
            simulation_count=simulation_count,
            export_suffix=export_suffix,
        )
    if view_mode == "Form":
        return export_document_png(
            "form_view",
            "Form View",
            tables,
            multi_column=False,
            export_suffix=export_suffix,
        )
    return export_document_png(
        "all_Countries_view",
        "All Countries View",
        tables,
        multi_column=False,
        export_suffix=export_suffix,
    )


def export_all_tables(
    probability_df: pd.DataFrame | None = None,
    form_df: pd.DataFrame | None = None,
    simulation_count: int | None = None,
    form_match_window: int = DEFAULT_RECENT_MATCH_WINDOW,
) -> list[Path]:
    """Export the probability tables and optionally the form table as PNG files."""
    exported_paths: list[Path] = []
    export_suffix = generate_export_suffix()
    if probability_df is not None:
        for group_code in GROUP_ORDER:
            group_df = projected_group_table_frame(probability_df, group_code)
            if group_df.empty:
                continue
            exported_paths.append(
                export_document_png(
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
                )
            )

        combined = all_teams_table_frame(probability_df)
        exported_paths.append(
            export_document_png(
                "all_Countries",
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
        exported_paths.append(
            export_document_png(
                "form_all_countries",
                "All Countries",
                all_countries_tables,
                multi_column=False,
                export_suffix=export_suffix,
            )
        )
        exported_paths.append(
            export_document_png(
                "form_all_confederations",
                "All Confederations",
                all_confederations_tables,
                multi_column=False,
                export_suffix=export_suffix,
            )
        )
        for table in all_confederations_tables:
            exported_paths.append(
                export_document_png(
                    str(table["stem"]),
                    str(table["title"]),
                    [table],
                    multi_column=False,
                    export_suffix=export_suffix,
                )
            )
    return exported_paths
