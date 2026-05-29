from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pandas as pd
import streamlit as st


GLOBAL_TEAM_QUERY_PARAM = "team"
GLOBAL_TEAM_SESSION_KEY = "global_team_id"


def _query_param_value(query_params: Any, key: str) -> str | None:
    value = query_params.get(key)
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def get_query_team_id() -> str | None:
    """Return the team id stored in the shared query parameter."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        return _query_param_value(query_params, GLOBAL_TEAM_QUERY_PARAM)
    getter = getattr(st, "experimental_get_query_params", None)
    if getter is None:
        return None
    return _query_param_value(getter(), GLOBAL_TEAM_QUERY_PARAM)


def get_global_team_id(valid_team_ids: Iterable[str], default_team_id: str) -> str:
    """Resolve the selected global team id from query params, session state, then default."""
    valid_ids = {str(team_id) for team_id in valid_team_ids}
    fallback = str(default_team_id)
    query_team_id = get_query_team_id()
    if query_team_id in valid_ids:
        st.session_state[GLOBAL_TEAM_SESSION_KEY] = query_team_id
        return query_team_id

    session_team_id = st.session_state.get(GLOBAL_TEAM_SESSION_KEY)
    if session_team_id is not None and str(session_team_id) in valid_ids:
        return str(session_team_id)

    if fallback in valid_ids:
        st.session_state[GLOBAL_TEAM_SESSION_KEY] = fallback
        return fallback
    resolved = next(iter(valid_ids), fallback)
    st.session_state[GLOBAL_TEAM_SESSION_KEY] = resolved
    return resolved


def set_global_team_id(team_id: str) -> None:
    """Persist the selected global team id in session state and the URL query string."""
    resolved_team_id = str(team_id)
    st.session_state[GLOBAL_TEAM_SESSION_KEY] = resolved_team_id
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params[GLOBAL_TEAM_QUERY_PARAM] = resolved_team_id
        return
    setter = getattr(st, "experimental_set_query_params", None)
    if setter is not None:
        setter(**{GLOBAL_TEAM_QUERY_PARAM: resolved_team_id})


def render_global_team_selectbox(
    label: str,
    team_choices: Sequence[str],
    key: str,
    format_func: Callable[[str], str] | None = None,
) -> str:
    """Render a team selectbox backed by the shared global team state."""
    if not team_choices:
        return ""
    team_ids = [str(team_id) for team_id in team_choices]
    selected_team_id = get_global_team_id(team_ids, team_ids[0])
    selected_index = team_ids.index(selected_team_id) if selected_team_id in team_ids else 0
    selected = st.selectbox(
        label,
        team_ids,
        index=selected_index,
        format_func=format_func or str,
        key=key,
    )
    set_global_team_id(str(selected))
    return str(selected)


def normalize_name(value: object) -> str:
    """Return a simple comparable key for team/country display names."""
    text = "" if value is None or pd.isna(value) else str(value)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().replace("-", " ").split())


def resolve_team_country_name(
    team_row: pd.Series | dict[str, object],
    available_country_names: Iterable[str],
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Resolve a dashboard team row to a historical-analysis country label."""
    available = [str(country) for country in available_country_names if str(country).strip()]
    exact_lookup = {country: country for country in available}
    normalized_lookup = {normalize_name(country): country for country in available}
    aliases = aliases or {}

    for column_name in ("display_name", "canonical_name", "tournament_name", "team", "team_name", "country", "team_id"):
        value = team_row.get(column_name, "") if hasattr(team_row, "get") else ""
        if value is None or pd.isna(value) or not str(value).strip():
            continue
        text = str(value)
        alias_text = aliases.get(text, text)
        for candidate in (text, alias_text):
            if candidate in exact_lookup:
                return exact_lookup[candidate]
            normalized_candidate = normalize_name(candidate)
            if normalized_candidate in normalized_lookup:
                return normalized_lookup[normalized_candidate]
    return None
