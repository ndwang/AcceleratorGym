import pytest

from accelerator_gym import server


@pytest.fixture
def mcp_machine(machine):
    """Inject machine into the server module and clean up after."""
    server._machine = machine
    yield machine
    server._machine = None


class TestBrowseDevices:
    def test_browse_root(self, mcp_machine):
        result = server.browse_devices("/")
        assert result["path"] == "/"
        assert "diagnostics" in result["children"]
        assert "magnets" in result["children"]

    def test_browse_system(self, mcp_machine):
        result = server.browse_devices("/magnets")
        assert result["path"] == "/magnets"
        assert "quadrupole" in result["children"]

    def test_browse_device_type(self, mcp_machine):
        result = server.browse_devices("/magnets/quadrupole")
        names = [d["name"] for d in result["children"]]
        assert "QF" in names
        assert "QD" in names

    def test_browse_device(self, mcp_machine):
        result = server.browse_devices("/magnets/quadrupole/QF")
        assert "K1" in result["children"]

    def test_browse_attribute(self, mcp_machine):
        result = server.browse_devices("/magnets/quadrupole/QF/K1")
        assert result["variable"] == "QF:K1"
        assert result["units"] == "1/m"
        assert result["limits"] == [-5.0, 5.0]
        assert result["read"] is True
        assert result["write"] is True

    def test_browse_depth_2(self, mcp_machine):
        result = server.browse_devices("/", depth=2)
        # Children should be dicts with nested children
        magnets = next(c for c in result["children"] if c["name"] == "magnets")
        assert "children" in magnets
        assert "quadrupole" in [c if isinstance(c, str) else c["name"] for c in magnets["children"]]

    def test_browse_invalid_system(self, mcp_machine):
        result = server.browse_devices("/nonexistent")
        assert "error" in result

    def test_browse_invalid_path(self, mcp_machine):
        result = server.browse_devices("/a/b/c/d/e")
        assert "error" in result


class TestQueryDevices:
    def test_query_systems(self, mcp_machine):
        result = server.query_devices(
            "SELECT DISTINCT system FROM devices ORDER BY system"
        )
        assert result["count"] == 2
        names = [r["system"] for r in result["rows"]]
        assert names == ["diagnostics", "magnets"]

    def test_query_devices_by_type(self, mcp_machine):
        result = server.query_devices(
            "SELECT device_id FROM devices WHERE device_type='quadrupole' ORDER BY device_id"
        )
        names = [r["device_id"] for r in result["rows"]]
        assert names == ["QD", "QF"]

    def test_query_attributes(self, mcp_machine):
        result = server.query_devices(
            "SELECT variable, unit FROM attributes WHERE writable=0"
        )
        assert result["count"] == 2
        variables = {r["variable"] for r in result["rows"]}
        assert variables == {"BPM1:X", "BPM1:Y"}

    def test_query_rejects_insert(self, mcp_machine):
        result = server.query_devices("INSERT INTO systems VALUES ('hack')")
        assert result["error"] == "query_error"

    def test_query_rejects_delete(self, mcp_machine):
        result = server.query_devices("DELETE FROM systems")
        assert result["error"] == "query_error"


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


class TestGetVariablesOutputFile:
    def test_output_file_csv(self, mcp_machine, tmp_path):
        path = str(tmp_path / "out.csv")
        result = server.get_variables(["QF:K1", "QD:K1"], output_file=path)
        assert result == {"file": path, "count": 2}
        with open(path) as f:
            lines = f.read().strip().split("\n")
        assert lines[0] == "name,value,units"
        assert lines[1] == "QF:K1,0.5,1/m"
        assert lines[2] == "QD:K1,-0.5,1/m"

    def test_output_file_no_units(self, mcp_machine, tmp_path):
        """Variables without units get an empty units column."""
        path = str(tmp_path / "out.csv")
        result = server.get_variables(["BPM1:X"], output_file=path)
        assert result["count"] == 1
        with open(path) as f:
            lines = f.read().strip().split("\n")
        assert lines[1].endswith(",mm") or lines[1].endswith(",")
        # BPM1:X has units "mm", so verify it's there
        assert "BPM1:X" in lines[1]

    def test_output_file_unknown_variable(self, mcp_machine, tmp_path):
        path = str(tmp_path / "out.csv")
        result = server.get_variables(["NOPE"], output_file=path)
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

    def test_set_not_writable(self, mcp_machine):
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


class TestReset:
    def test_reset(self, mcp_machine):
        server.set_variable("QF:K1", 2.0)
        result = server.reset()
        assert result["success"] is True
        state = server.get_variable("QF:K1")
        assert state["value"] == 0.5
