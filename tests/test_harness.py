"""Tests for the benchmark harness orchestrator."""

import json
import os
import tempfile

import numpy as np
import pytest

from accelbench.harness import _safe_serialize, _select_tasks, task_seed
from accelbench.tasks import ALL_TASKS, TASKS_BY_ID


class TestTaskSeed:
    def test_deterministic(self):
        s1 = task_seed(42, "1.1")
        s2 = task_seed(42, "1.1")
        assert s1 == s2

    def test_different_tasks_different_seeds(self):
        s1 = task_seed(42, "1.1")
        s2 = task_seed(42, "1.2")
        assert s1 != s2

    def test_different_base_seeds(self):
        s1 = task_seed(42, "1.1")
        s2 = task_seed(99, "1.1")
        assert s1 != s2

    def test_produces_valid_rng_seed(self):
        s = task_seed(42, "1.1")
        rng = np.random.default_rng(s)
        val = rng.random()
        assert 0.0 <= val < 1.0


class TestSelectTasks:
    def test_select_all(self):
        tasks = _select_tasks(None, None)
        assert len(tasks) == len(ALL_TASKS)

    def test_select_by_ids(self):
        tasks = _select_tasks(["1.1", "1.2"], None)
        assert len(tasks) == 2
        assert tasks[0].id == "1.1"
        assert tasks[1].id == "1.2"

    def test_select_by_tier(self):
        tasks = _select_tasks(None, tier=1)
        assert all(t.tier == 1 for t in tasks)
        assert len(tasks) > 0

    def test_select_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Unknown task ID"):
            _select_tasks(["99.99"], None)

    def test_select_empty_tier(self):
        tasks = _select_tasks(None, tier=999)
        assert tasks == []

    def test_ids_take_precedence_over_tier(self):
        tasks = _select_tasks(["2.1"], tier=1)
        assert len(tasks) == 1
        assert tasks[0].id == "2.1"


class TestSafeSerialize:
    def test_none(self):
        assert _safe_serialize(None) is None

    def test_dict(self):
        assert _safe_serialize({"a": 1}) == {"a": 1}

    def test_list(self):
        assert _safe_serialize([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert _safe_serialize((1, 2)) == [1, 2]

    def test_numpy_integer(self):
        result = _safe_serialize(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = _safe_serialize(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_array(self):
        result = _safe_serialize(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]

    def test_nested_numpy(self):
        data = {"values": np.array([1, 2]), "count": np.int64(2)}
        result = _safe_serialize(data)
        assert result == {"values": [1, 2], "count": 2}
        # Verify it's JSON-serializable
        json.dumps(result)

    def test_plain_types_pass_through(self):
        assert _safe_serialize("hello") == "hello"
        assert _safe_serialize(42) == 42
        assert _safe_serialize(3.14) == 3.14
        assert _safe_serialize(True) is True


class TestTaskRegistry:
    """Sanity checks on the task registry itself."""

    def test_all_tasks_have_unique_ids(self):
        ids = [t.id for t in ALL_TASKS]
        assert len(ids) == len(set(ids))

    def test_tasks_by_id_matches_all_tasks(self):
        assert len(TASKS_BY_ID) == len(ALL_TASKS)
        for task in ALL_TASKS:
            assert TASKS_BY_ID[task.id] is task

    def test_all_tasks_have_required_fields(self):
        for task in ALL_TASKS:
            assert task.id
            assert task.name
            assert task.tier >= 1
            assert task.budget > 0
            assert task.prompt_template
            assert callable(task.setup)
            assert callable(task.verify)

    def test_tiers_present(self):
        tiers = {t.tier for t in ALL_TASKS}
        assert 1 in tiers
        assert 2 in tiers
