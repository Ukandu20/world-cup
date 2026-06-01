from __future__ import annotations

from datetime import datetime


def format_artifact_timestamp(value: str | None) -> str:
    """Return a compact display timestamp for artifact metadata."""
    if not value:
        return "time unavailable"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "time unavailable"
    period_label = parsed.strftime("%p").lower()
    return f"{parsed:%Y-%m-%d} @ {parsed:%I:%M}{period_label}"
