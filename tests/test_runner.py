"""Tests for the single task lifecycle runner."""

import pytest

from accelbench.runner import (
    REASON_BUDGET_EXCEEDED,
    REASON_CRASH,
    REASON_NO_ANSWER,
    REASON_SETUP_ERROR,
    REASON_TIMEOUT,
    REASON_WRONG_ANSWER,
    _merge_reasoning_trace,
    _replay_trace,
    run_task,
)
from accelbench.types import Env, TaskDef

import numpy as np


# ---------------------------------------------------------------------------
# Mock adapter that returns a canned response
# ---------------------------------------------------------------------------

class MockAdapter:
    """Adapter that returns a fixed response and optionally calls tools."""

    def __init__(self, response: str, tool_calls: list | None = None):
        self._response = response
        self._tool_calls = tool_calls or []

    def run(self, prompt, tools, call_tool):
        for name, args in self._tool_calls:
            call_tool(name, args)
        return self._response


class CrashingAdapter:
    def run(self, prompt, tools, call_tool):
        raise RuntimeError("adapter exploded")


class SlowAdapter:
    def run(self, prompt, tools, call_tool):
        import time
        time.sleep(10)
        return '```json\n{"value": 1}\n```'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_task(
    setup_data: dict | None = None,
    verify_result: bool = True,
    budget: int = 5,
    setup_raises: Exception | None = None,
) -> TaskDef:
    """Create a minimal TaskDef for testing."""
    setup_data = setup_data or {"variable": "QF:K1"}

    def setup(env):
        if setup_raises:
            raise setup_raises
        return dict(setup_data)

    def verify(answer, env, sdata):
        return verify_result

    return TaskDef(
        id="T.1",
        name="Test Task",
        tier=1,
        budget=budget,
        prompt_template="Read {variable} and report its value.",
        setup=setup,
        verify=verify,
    )


# ---------------------------------------------------------------------------
# Tests: run_task lifecycle
# ---------------------------------------------------------------------------

class TestRunTaskSuccess:
    def test_passing_task(self, machine):
        task = _simple_task(verify_result=True)
        adapter = MockAdapter('```json\n{"value": 0.5}\n```')
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is True
        assert result.task_id == "T.1"
        assert result.failure_reason is None
        assert result.extracted_answer == {"value": 0.5}
        assert result.wall_time > 0
        assert result.error is None

    def test_tool_calls_counted(self, machine):
        task = _simple_task(budget=5)
        adapter = MockAdapter(
            '```json\n{"value": 0.5}\n```',
            tool_calls=[
                ("get_variables", {"names": ["QF:K1"]}),
            ],
        )
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)
        assert result.tool_calls == 1

    def test_prompt_includes_setup_data(self, machine):
        task = _simple_task(setup_data={"variable": "QF:K1"})

        captured_prompt = {}

        class CapturingAdapter:
            def run(self, prompt, tools, call_tool):
                captured_prompt["prompt"] = prompt
                return '```json\n{"value": 1}\n```'

        rng = np.random.default_rng(42)
        run_task(task, machine, CapturingAdapter(), rng, timeout=5)
        assert "QF:K1" in captured_prompt["prompt"]


class TestRunTaskFailures:
    def test_setup_error(self, machine):
        task = _simple_task(setup_raises=ValueError("bad lattice"))
        adapter = MockAdapter("")
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is False
        assert result.failure_reason == REASON_SETUP_ERROR
        assert "Setup failed" in result.error

    def test_crash(self, machine):
        task = _simple_task()
        adapter = CrashingAdapter()
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is False
        assert result.failure_reason == REASON_CRASH
        assert "adapter exploded" in result.error

    def test_timeout(self, machine):
        task = _simple_task()
        adapter = SlowAdapter()
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=1)

        assert result.passed is False
        assert result.failure_reason == REASON_TIMEOUT
        assert "timed out" in result.error

    def test_no_answer(self, machine):
        task = _simple_task()
        adapter = MockAdapter("I don't know the answer, sorry!")
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is False
        assert result.failure_reason == REASON_NO_ANSWER
        assert "No JSON answer" in result.error

    def test_wrong_answer(self, machine):
        task = _simple_task(verify_result=False, budget=10)
        adapter = MockAdapter('```json\n{"value": 999}\n```')
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is False
        assert result.failure_reason == REASON_WRONG_ANSWER

    def test_budget_exceeded_reason(self, machine):
        task = _simple_task(verify_result=False, budget=2)
        adapter = MockAdapter(
            '```json\n{"value": 0}\n```',
            tool_calls=[
                ("get_variables", {"names": ["QF:K1"]}),
                ("get_variables", {"names": ["QD:K1"]}),
            ],
        )
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is False
        assert result.failure_reason == REASON_BUDGET_EXCEEDED

    def test_empty_response(self, machine):
        task = _simple_task()
        adapter = MockAdapter("")
        rng = np.random.default_rng(42)
        result = run_task(task, machine, adapter, rng, timeout=5)

        assert result.passed is False
        assert result.failure_reason == REASON_NO_ANSWER


class TestRunTaskMetadata:
    def test_model_and_usage_captured(self, machine):
        task = _simple_task()

        class AdapterWithMeta:
            model = "test-model-v1"
            last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

            def run(self, prompt, tools, call_tool):
                return '```json\n{"value": 1}\n```'

        rng = np.random.default_rng(42)
        result = run_task(task, machine, AdapterWithMeta(), rng, timeout=5)
        assert result.model == "test-model-v1"
        assert result.usage["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Tests: _replay_trace
# ---------------------------------------------------------------------------

class TestReplayTrace:
    def test_replays_set_variables(self, machine):
        trace = [
            {"tool": "set_variables", "arguments": {"values": {"QF:K1": 2.0}}, "result": "QF:K1 set to 2.0"},
        ]
        _replay_trace(machine, trace)
        assert machine.get("QF:K1") == 2.0

    def test_replays_reset(self, machine):
        machine.set("QF:K1", 3.0)
        trace = [
            {"tool": "reset", "arguments": {}, "result": "Machine reset to initial state"},
        ]
        _replay_trace(machine, trace)
        assert machine.get("QF:K1") == 0.5

    def test_skips_failed_calls(self, machine):
        trace = [
            {"tool": "set_variables", "arguments": {"values": {"QF:K1": 2.0}}, "result": "Error: budget exceeded"},
        ]
        _replay_trace(machine, trace)
        assert machine.get("QF:K1") == 0.5  # unchanged

    def test_skips_non_mutation_tools(self, machine):
        trace = [
            {"tool": "get_variables", "arguments": {"names": ["QF:K1"]}, "result": "QF:K1 = 0.5"},
            {"tool": "browse_devices", "arguments": {"path": "/"}, "result": "{}"},
        ]
        _replay_trace(machine, trace)
        assert machine.get("QF:K1") == 0.5

    def test_skips_reasoning_entries(self, machine):
        trace = [
            {"role": "assistant", "content": "Let me think..."},
            {"tool": "set_variables", "arguments": {"values": {"QF:K1": 2.0}}, "result": "ok"},
        ]
        _replay_trace(machine, trace)
        assert machine.get("QF:K1") == 2.0


# ---------------------------------------------------------------------------
# Tests: _merge_reasoning_trace
# ---------------------------------------------------------------------------

class TestMergeReasoningTrace:
    def test_no_reasoning(self):
        tool_trace = [{"tool": "get_variables"}]
        result = _merge_reasoning_trace(tool_trace, None)
        assert result == tool_trace

    def test_empty_reasoning(self):
        tool_trace = [{"tool": "get_variables"}]
        result = _merge_reasoning_trace(tool_trace, [])
        assert result == tool_trace

    def test_reasoning_before_tool_call(self):
        tool_trace = [
            {"tool": "get_variables"},
            {"tool": "set_variables"},
        ]
        reasoning = [
            {"content": "I should read first", "tool_call_index": 0},
            {"content": "Now I'll set", "tool_call_index": 1},
        ]
        result = _merge_reasoning_trace(tool_trace, reasoning)
        assert len(result) == 4
        assert result[0]["content"] == "I should read first"
        assert result[1]["tool"] == "get_variables"
        assert result[2]["content"] == "Now I'll set"
        assert result[3]["tool"] == "set_variables"

    def test_trailing_reasoning(self):
        tool_trace = [{"tool": "get_variables"}]
        reasoning = [
            {"content": "final thought", "tool_call_index": 99},
        ]
        result = _merge_reasoning_trace(tool_trace, reasoning)
        assert len(result) == 2
        assert result[0]["tool"] == "get_variables"
        assert result[1]["content"] == "final thought"

    def test_multiple_reasoning_before_same_tool(self):
        tool_trace = [{"tool": "get_variables"}]
        reasoning = [
            {"content": "thought 1", "tool_call_index": 0},
            {"content": "thought 2", "tool_call_index": 0},
        ]
        result = _merge_reasoning_trace(tool_trace, reasoning)
        assert len(result) == 3
        assert result[0]["content"] == "thought 1"
        assert result[1]["content"] == "thought 2"
        assert result[2]["tool"] == "get_variables"
