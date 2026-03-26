"""Canonical tool descriptions loaded from markdown files."""

from pathlib import Path

_DIR = Path(__file__).parent

TOOL_DESCRIPTIONS: dict[str, str] = {}

for md in _DIR.glob("*.md"):
    TOOL_DESCRIPTIONS[md.stem] = md.read_text().strip()
