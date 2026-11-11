"""Cinematic fan-edit generator: beat-synced intro/outro montages around a highlight clip.

    config/edit.yaml   all knobs + presets
    beats.py           tempo / beats / drop detection (NumPy)
    music.py           licensed local music library + mood selection
    moments.py         extra moments from the same VOD for the montage
    plan.py            builds the edit decision list (pure data, inspectable)
    render.py          renders the plan with FFmpeg
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT

DEFAULT_EDIT_CONFIG = ROOT / "config" / "edit.yaml"


def load_edit_config(path: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((path or DEFAULT_EDIT_CONFIG).read_text())
    lib = Path(cfg["music"]["library"])
    cfg["music"]["library"] = lib if lib.is_absolute() else ROOT / lib
    return cfg


def resolve_preset(ecfg: dict, name: str, category: str | None = None) -> dict[str, Any]:
    """Merge defaults + preset + caption style into one flat dict."""
    if name == "auto":
        name = ecfg["auto_preset"].get(category or "other", "gaming-cinematic")
    if name not in ecfg["presets"]:
        raise ValueError(f"Unknown preset {name!r}. Available: {', '.join(ecfg['presets'])}, auto")
    p = {**ecfg["defaults"], **ecfg["presets"][name], "name": name}
    style = p["caption_style"]
    if style not in ecfg["caption_styles"]:
        raise ValueError(f"Unknown caption_style {style!r}")
    p["captions"] = ecfg["caption_styles"][style]
    grade = p["color_grade"]
    if grade not in ecfg["color_grades"]:
        raise ValueError(f"Unknown color_grade {grade!r}")
    p["grade_filter"] = ecfg["color_grades"][grade]
    return p
