"""Externalized prompts loaded from markdown files."""

from pathlib import Path

_DIR = Path(__file__).parent

SYSTEM_PROMPT = (_DIR / "system.md").read_text().strip()
ANSWER_INSTRUCTION = (_DIR / "answer_instruction.md").read_text().strip()

TASK_PROMPTS: dict[str, str] = {}
for md in (_DIR / "tasks").glob("*.md"):
    TASK_PROMPTS[md.stem] = md.read_text().strip()
