import pytest

from accelerator_gym.agents.llm import LLMInterface
from accelerator_gym.core.machine import Machine


@pytest.fixture
def llm(machine):
    return LLMInterface(machine)


class TestGetTools:
    def test_returns_list(self, llm):
        tools = llm.get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 7

    def test_tool_names(self, llm):
        tools = llm.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "list_variables" in names
        assert "get_variable" in names
        assert "get_variables" in names
        assert "set_variable" in names
        assert "set_variables" in names
        assert "get_state" in names
        assert "reset" in names

    def test_openai_format(self, llm):
        tools = llm.get_tools()
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_schemas_do_not_enumerate_variables(self, llm):
        tools = llm.get_tools()
        for tool in tools:
            props = tool["function"]["parameters"].get("properties", {})
            for prop in props.values():
                assert "enum" not in prop, f"Tool {tool['function']['name']} should not enumerate variables"


class TestExecuteListVariables:
    def test_list_all(self, llm):
        result = llm.execute("list_variables", {})
        assert "variables" in result
        names = [v["name"] for v in result["variables"]]
        assert "QF:K1" in names
        assert "BPM1:X" in names

    def test_list_with_filter(self, llm):
        result = llm.execute("list_variables", {"filter": "Q*"})
        names = [v["name"] for v in result["variables"]]
        assert "QF:K1" in names
        assert "QD:K1" in names
        assert "BPM1:X" not in names

    def test_list_includes_metadata(self, llm):
        result = llm.execute("list_variables", {})
        qf = next(v for v in result["variables"] if v["name"] == "QF:K1")
        assert qf["units"] == "1/m^2"
        assert qf["limits"] == [-5.0, 5.0]
        bpm = next(v for v in result["variables"] if v["name"] == "BPM1:X")
        assert bpm["read_only"] is True


class TestExecuteGetVariable:
    def test_get_existing(self, llm):
        result = llm.execute("get_variable", {"name": "QF:K1"})
        assert result["name"] == "QF:K1"
        assert result["value"] == 0.5
        assert result["units"] == "1/m^2"

    def test_get_unknown(self, llm):
        result = llm.execute("get_variable", {"name": "NONEXISTENT"})
        assert "error" in result
        assert result["error"] == "not_found"


class TestExecuteGetVariables:
    def test_get_multiple(self, llm):
        result = llm.execute("get_variables", {"names": ["QF:K1", "QD:K1"]})
        assert result["values"]["QF:K1"] == 0.5
        assert result["values"]["QD:K1"] == -0.5

    def test_get_unknown_in_list(self, llm):
        result = llm.execute("get_variables", {"names": ["QF:K1", "NOPE"]})
        assert "error" in result


class TestExecuteSetVariable:
    def test_set_valid(self, llm):
        result = llm.execute("set_variable", {"name": "QF:K1", "value": 1.0})
        assert result["success"] is True
        assert result["value"] == 1.0

    def test_set_out_of_limits(self, llm):
        result = llm.execute("set_variable", {"name": "QF:K1", "value": 100.0})
        assert "error" in result
        assert result["variable"] == "QF:K1"
        assert result["limits"] == [-5.0, 5.0]

    def test_set_read_only(self, llm):
        result = llm.execute("set_variable", {"name": "BPM1:X", "value": 1.0})
        assert "error" in result

    def test_set_unknown(self, llm):
        result = llm.execute("set_variable", {"name": "NOPE", "value": 1.0})
        assert "error" in result
        assert result["error"] == "not_found"


class TestExecuteSetVariables:
    def test_set_multiple(self, llm):
        result = llm.execute(
            "set_variables", {"values": {"QF:K1": 1.0, "QD:K1": -1.0}}
        )
        assert result["success"] is True

    def test_set_multiple_with_violation(self, llm):
        result = llm.execute(
            "set_variables", {"values": {"QF:K1": 1.0, "QD:K1": -100.0}}
        )
        assert "error" in result


class TestExecuteGetState:
    def test_get_state(self, llm):
        result = llm.execute("get_state", {})
        assert "variables" in result
        assert result["variables"]["QF:K1"] == 0.5


class TestExecuteReset:
    def test_reset(self, llm):
        llm.execute("set_variable", {"name": "QF:K1", "value": 2.0})
        result = llm.execute("reset", {})
        assert result["success"] is True
        state = llm.execute("get_variable", {"name": "QF:K1"})
        assert state["value"] == 0.5


class TestExecuteUnknownTool:
    def test_unknown_tool(self, llm):
        result = llm.execute("nonexistent_tool", {})
        assert "error" in result
        assert result["error"] == "unknown_tool"
