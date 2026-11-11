"""Load settings.yaml, apply CLIPFARM_* env overrides, resolve paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_config(path: Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((path or DEFAULT_CONFIG).read_text())

    # CLIPFARM_<SECTION>_<KEY>=value overrides cfg[section][key]
    for env_key, raw in os.environ.items():
        if not env_key.startswith("CLIPFARM_"):
            continue
        parts = env_key[len("CLIPFARM_"):].lower().split("_", 1)
        if len(parts) == 2 and parts[0] in cfg and parts[1] in cfg[parts[0]]:
            cfg[parts[0]][parts[1]] = _coerce(raw)

    for key, rel in cfg["paths"].items():
        p = Path(rel)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    for key in ("inbox", "library", "cache", "clips", "edits", "logs"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg
