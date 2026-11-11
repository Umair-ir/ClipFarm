"""Truncated-JSON salvage and schema conformance for LLM output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clipfarm.highlights import RESPONSE_SCHEMA  # noqa: E402
from clipfarm.llm import conform, salvage_truncated  # noqa: E402

CLIP = {"start_line": 1, "end_line": 4, "category": "funny", "score": 8, "title": "t",
        "description": "d", "hashtags": ["#a"], "reason": "r"}


def test_salvage_keeps_complete_clips():
    full = json.dumps({"clips": [CLIP, {**CLIP, "start_line": 9}, CLIP]})
    cut = full[: full.rindex('"title"') + 5]       # third clip cut off mid-key
    data = conform(salvage_truncated(cut), RESPONSE_SCHEMA)
    assert [c["start_line"] for c in data["clips"]] == [1, 9]


def test_salvage_drops_partial_item_cut_after_nested_array():
    partial = {k: CLIP[k] for k in ("start_line", "end_line", "hashtags")}  # missing required fields
    full = json.dumps({"clips": [CLIP, partial]})
    cut = full[: full.rindex("]") ]  # cut right before the outer array closes, after the nested one
    data = conform(salvage_truncated(cut), RESPONSE_SCHEMA)
    assert len(data["clips"]) == 1


def test_salvage_handles_strings_with_brackets():
    full = json.dumps({"clips": [{**CLIP, "title": 'he said "}]" lol'}, CLIP]})
    cut = full[:-30]
    data = conform(salvage_truncated(cut), RESPONSE_SCHEMA)
    assert data["clips"][0]["title"] == 'he said "}]" lol'


def test_nothing_salvageable():
    assert salvage_truncated('{"clips": [{"start_li') is None
    assert salvage_truncated("   \n\n   ") is None


def test_conform_rejects_missing_top_level_fields():
    assert conform({"clips": []}, RESPONSE_SCHEMA) == {"clips": []}
    assert conform({}, RESPONSE_SCHEMA) is None
