from __future__ import annotations

from datetime import datetime
from pathlib import Path


_LOG_PATH = Path.cwd() / "triangulation_debug.log"


def tri_debug(event: str, **fields: object) -> None:
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
    line = f"{timestamp} [{event}] {payload}\n" if payload else f"{timestamp} [{event}]\n"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(line)
    except OSError:
        return

