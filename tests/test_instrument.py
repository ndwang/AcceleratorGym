"""Tests for InstrumentedMachine budget enforcement and tool call counting."""

import json

import pytest

from accelbench.instrument import InstrumentedMachine, ToolCall, make_call_tool


class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall("get_variables", {"names": ["QF:K1"]}, "QF:K1 = 0.5")
        d = tc.to_dict()
        assert d == {
            "tool": "get_variables",
            "arguments": {"names": ["QF:K1"]},
            "result": "QF:K1 = 0.5",
        }

    def test_slots(self):
        tc = ToolCall("reset", {}, "ok")
        with pytest.raises(AttributeError):
            tc.extra = "nope"


class TestInstrumentedMachineProperties:
    def test_initial_state(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        assert im.call_count == 0
        assert im.budget == 10
        assert im.trace == []
        assert im.machine is machine

    def test_trace_returns_copy(self, machine):
        im = InstrumentedMachine(machine, budget=5)
        im.get_variables(["QF:K1"])
        t1 = im.trace
        t2 = im.trace
        assert t1 is not t2
        assert len(t1) == 1


class TestBudgetEnforcement:
    def test_calls_within_budget(self, machine):
        im = InstrumentedMachine(machine, budget=3)
        r1 = im.get_variables(["QF:K1"])
        r2 = im.get_variables(["QD:K1"])
        r3 = im.get_variables(["BPM1:X"])
        assert im.call_count == 3
        assert "Error" not in r1
        assert "Error" not in r2
        assert "Error" not in r3

    def test_budget_exceeded(self, machine):
        im = InstrumentedMachine(machine, budget=2)
        im.get_variables(["QF:K1"])
        im.get_variables(["QD:K1"])
        result = im.get_variables(["BPM1:X"])
        assert result == "Error: tool call budget exceeded"
        assert im.call_count == 2  # count doesn't increment past budget

    def test_budget_exceeded_logged_in_trace(self, machine):
        im = InstrumentedMachine(machine, budget=1)
        im.get_variables(["QF:K1"])
        im.set_variables({"QF:K1": 1.0})
        assert len(im.trace) == 2
        assert "budget exceeded" in im.trace[1].result

    def test_budget_one(self, machine):
        im = InstrumentedMachine(machine, budget=1)
        r1 = im.browse_devices()
        assert "Error" not in r1
        r2 = im.browse_devices()
        assert "budget exceeded" in r2

    def test_set_not_applied_after_budget(self, machine):
        im = InstrumentedMachine(machine, budget=1)
        im.get_variables(["QF:K1"])
        im.set_variables({"QF:K1": 9.9})
        assert machine.get("QF:K1") == 0.5  # unchanged


class TestInstrumentedTools:
    def test_browse_devices(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.browse_devices("/")
        parsed = json.loads(result)
        assert "children" in parsed

    def test_browse_devices_with_depth(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.browse_devices("/", depth=2)
        parsed = json.loads(result)
        assert "children" in parsed

    def test_query_devices(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.query_devices("SELECT COUNT(*) as cnt FROM devices")
        parsed = json.loads(result)
        assert "rows" in parsed
        assert parsed["count"] >= 1

    def test_query_devices_error(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.query_devices("DELETE FROM devices")
        parsed = json.loads(result)
        assert parsed["error"] == "query_error"

    def test_get_variables(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.get_variables(["QF:K1"])
        assert "QF:K1 = 0.5" in result
        assert "(1/m)" in result  # units

    def test_get_variables_multiple(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.get_variables(["QF:K1", "QD:K1"])
        assert "QF:K1" in result
        assert "QD:K1" in result
        assert im.call_count == 1  # batch = single call

    def test_get_variables_unknown(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.get_variables(["NONEXISTENT"])
        assert "Error" in result

    def test_set_variables(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.set_variables({"QF:K1": 1.0})
        assert "set to" in result
        assert machine.get("QF:K1") == 1.0

    def test_set_variables_limit_violation(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.set_variables({"QF:K1": 999.0})
        assert "Error" in result
        assert machine.get("QF:K1") == 0.5  # unchanged

    def test_set_variables_readonly(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        result = im.set_variables({"BPM1:X": 1.0})
        assert "Error" in result

    def test_reset(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        im.set_variables({"QF:K1": 2.0})
        result = im.reset()
        assert result == "Machine reset to initial state"
        assert machine.get("QF:K1") == 0.5


class TestTraceRecording:
    def test_trace_records_all_calls(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        im.browse_devices()
        im.get_variables(["QF:K1"])
        im.set_variables({"QF:K1": 1.0})
        im.reset()
        assert len(im.trace) == 4
        assert [tc.tool for tc in im.trace] == [
            "browse_devices",
            "get_variables",
            "set_variables",
            "reset",
        ]

    def test_trace_records_arguments(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        im.get_variables(["QF:K1", "QD:K1"])
        tc = im.trace[0]
        assert tc.arguments == {"names": ["QF:K1", "QD:K1"]}

    def test_trace_records_results(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        im.get_variables(["QF:K1"])
        tc = im.trace[0]
        assert "0.5" in tc.result


class TestMakeCallTool:
    def test_dispatches_all_tools(self, machine):
        im = InstrumentedMachine(machine, budget=20)
        call_tool = make_call_tool(im)

        r = call_tool("browse_devices", {"path": "/"})
        assert "children" in r

        r = call_tool("query_devices", {"sql": "SELECT COUNT(*) as c FROM devices"})
        assert "rows" in r

        r = call_tool("get_variables", {"names": ["QF:K1"]})
        assert "QF:K1" in r

        r = call_tool("set_variables", {"values": {"QF:K1": 1.5}})
        assert "set to" in r

        r = call_tool("reset", {})
        assert "reset" in r.lower()

    def test_unknown_tool(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        call_tool = make_call_tool(im)
        result = call_tool("nonexistent_tool", {})
        assert "unknown tool" in result.lower()

    def test_default_args(self, machine):
        im = InstrumentedMachine(machine, budget=10)
        call_tool = make_call_tool(im)
        # browse_devices with no path/depth should use defaults
        r = call_tool("browse_devices", {})
        assert "children" in r
