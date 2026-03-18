"""All benchmark tasks, aggregated."""

from __future__ import annotations

from accelbench.types import TaskDef

from accelbench.tasks.tier1 import TIER1_TASKS
from accelbench.tasks.tier2 import TIER2_TASKS
from accelbench.tasks.tier3 import TIER3_TASKS
from accelbench.tasks.tier4 import TIER4_TASKS

ALL_TASKS: list[TaskDef] = TIER1_TASKS + TIER2_TASKS + TIER3_TASKS + TIER4_TASKS

TASKS_BY_ID: dict[str, TaskDef] = {t.id: t for t in ALL_TASKS}
