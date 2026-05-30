from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes for a Streamlit download button."""
    return frame.to_csv(index=False).encode("utf-8")


def table_to_download_frame(table: dict[str, object]) -> pd.DataFrame:
    """Return the DataFrame backing one dashboard table."""
    frame = table.get("frame")
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    return frame.reset_index(drop=True).copy()


def combine_table_download_frames(
    tables: list[dict[str, object]],
    section_column: str = "section",
) -> pd.DataFrame:
    """Combine dashboard table frames with a leading section label."""
    frames: list[pd.DataFrame] = []
    for table in tables:
        frame = table_to_download_frame(table)
        if frame.empty:
            continue
        section_label = str(table.get("group_pill_label") or table.get("title") or "Table")
        column_name = section_column
        if column_name in frame.columns:
            column_name = "section"
        frame.insert(0, column_name, section_label)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[section_column])
    return pd.concat(frames, ignore_index=True, sort=False)


def tables_to_download_frame(
    tables: list[dict[str, object]],
    section_column: str = "section",
) -> pd.DataFrame:
    """Return one downloadable frame for one or many dashboard tables."""
    if len(tables) == 1:
        return table_to_download_frame(tables[0])
    return combine_table_download_frames(tables, section_column=section_column)


def bracket_to_download_frame(
    bracket_data: dict[str, object],
    metadata_lookup: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Flatten deterministic bracket data into a downloadable table."""
    rows: list[dict[str, Any]] = []
    for round_index, round_data in enumerate(bracket_data.get("rounds", []), start=1):
        if not isinstance(round_data, dict):
            continue
        round_code = str(round_data.get("round_code", ""))
        round_label = str(round_data.get("round_label", round_code))
        for slot, match in enumerate(round_data.get("matches", []), start=1):
            if not isinstance(match, dict):
                continue
            home_team_id = str(match.get("home_team_id", ""))
            away_team_id = str(match.get("away_team_id", ""))
            winner_team_id = str(match.get("winner_team_id", ""))
            rows.append(
                {
                    "round_index": round_index,
                    "round_code": round_code,
                    "round_label": round_label,
                    "slot": slot,
                    "match_number": match.get("match_number"),
                    "home_team_id": home_team_id,
                    "home_team": _display_name(home_team_id, metadata_lookup),
                    "away_team_id": away_team_id,
                    "away_team": _display_name(away_team_id, metadata_lookup),
                    "winner_team_id": winner_team_id,
                    "winner_team": _display_name(winner_team_id, metadata_lookup),
                    "winner_win_prob": match.get("winner_win_prob"),
                }
            )
    return pd.DataFrame(rows)


def _display_name(team_id: str, metadata_lookup: dict[str, dict[str, str]]) -> str:
    if not team_id:
        return ""
    return metadata_lookup.get(team_id, {}).get("display_name", team_id)
