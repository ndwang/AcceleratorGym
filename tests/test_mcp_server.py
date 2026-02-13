import pytest

from accelerator_gym import server
from accelerator_gym.core.machine import Machine


@pytest.fixture
def mcp_machine(machine):
    """Inject machine into the server module and clean up after."""
    server._machine = machine
    yield machine
    server._machine = None


class TestListVariables:
    def test_list_all(self, mcp_machine):
        result = server.list_variables()
        assert "variables" in result
        names = [v["name"] for v in result["variables"]]
        assert "QF:K1" in names
        assert "BPM1:X" in names

    def test_list_with_filter(self, mcp_machine):
        result = server.list_variables(filter="Q*")
        names = [v["name"] for v in result["variables"]]
        assert "QF:K1" in names
        assert "QD:K1" in names
        assert "BPM1:X" not in names

    def test_list_includes_metadata(self, mcp_machine):
        result = server.list_variables()
        qf = next(v for v in result["variables"] if v["name"] == "QF:K1")
        assert qf["units"] == "1/m"
        assert qf["limits"] == [-5.0, 5.0]
        bpm = next(v for v in result["variables"] if v["name"] == "BPM1:X")
        assert bpm["read_only"] is True


class TestGetVariable:
    def test_get_existing(self, mcp_machine):
        result = server.get_variable("QF:K1")
        assert result["name"] == "QF:K1"
        assert result["value"] == 0.5
        assert result["units"] == "1/m"

    def test_get_unknown(self, mcp_machine):
        result = server.get_variable("NONEXISTENT")
        assert result["error"] == "not_found"


class TestGetVariables:
    def test_get_multiple(self, mcp_machine):
        result = server.get_variables(["QF:K1", "QD:K1"])
        assert result["values"]["QF:K1"] == 0.5
        assert result["values"]["QD:K1"] == -0.5

    def test_get_unknown_in_list(self, mcp_machine):
        result = server.get_variables(["QF:K1", "NOPE"])
        assert "error" in result


class TestSetVariable:
    def test_set_valid(self, mcp_machine):
        result = server.set_variable("QF:K1", 1.0)
        assert result["success"] is True
        assert result["value"] == 1.0

    def test_set_out_of_limits(self, mcp_machine):
        result = server.set_variable("QF:K1", 100.0)
        assert result["error"] == "validation_error"
        assert result["variable"] == "QF:K1"
        assert result["limits"] == [-5.0, 5.0]

    def test_set_read_only(self, mcp_machine):
        result = server.set_variable("BPM1:X", 1.0)
        assert "error" in result

    def test_set_unknown(self, mcp_machine):
        result = server.set_variable("NOPE", 1.0)
        assert result["error"] == "not_found"


class TestSetVariables:
    def test_set_multiple(self, mcp_machine):
        result = server.set_variables({"QF:K1": 1.0, "QD:K1": -1.0})
        assert result["success"] is True

    def test_set_multiple_with_violation(self, mcp_machine):
        result = server.set_variables({"QF:K1": 1.0, "QD:K1": -100.0})
        assert "error" in result


class TestGetState:
    def test_get_state(self, mcp_machine):
        result = server.get_state()
        assert "variables" in result
        assert result["variables"]["QF:K1"] == 0.5


class TestReset:
    def test_reset(self, mcp_machine):
        server.set_variable("QF:K1", 2.0)
        result = server.reset()
        assert result["success"] is True
        state = server.get_variable("QF:K1")
        assert state["value"] == 0.5
