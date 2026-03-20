"""AccelBench task definitions."""

from .tier1 import TIER1_TASKS
from .tier2 import TIER2_TASKS
from .tier3 import TIER3_TASKS
from .tier4 import TIER4_TASKS

ALL_TASKS = TIER1_TASKS + TIER2_TASKS + TIER3_TASKS + TIER4_TASKS

TASKS_BY_ID = {task.id: task for task in ALL_TASKS}
