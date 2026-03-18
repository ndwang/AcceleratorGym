"""Extract JSON answers from agent responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_answer(text: str) -> dict[str, Any] | None:
    """Parse a JSON answer from an agent's final response.

    Strategy:
    1. Look for ```json ... ``` fenced code block
    2. Fallback: last {...} in text
    3. Return None if nothing found
    """
    # Strategy 1: fenced json block
    pattern = r"```json\s*\n?(.*?)\n?\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass

    # Strategy 2: last {...} block
    brace_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for candidate in reversed(brace_matches):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    return None
