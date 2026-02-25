import pytest

from accelerator_gym.core.catalog import Catalog
from accelerator_gym.core.variable import Variable


@pytest.fixture
def catalog(mock_backend, basic_config):
    """A Catalog built from basic_config devices."""
    return Catalog(basic_config.devices, mock_backend)


class TestBuildVariables:
    def test_builds_flat_registry(self, catalog):
        variables = catalog.build_variables()
        assert "QF:K1" in variables
        assert "QD:K1" in variables
        assert "BPM1:X" in variables
        assert "BPM1:Y" in variables

    def test_variable_metadata(self, catalog):
        variables = catalog.build_variables()
        qf = variables["QF:K1"]
        assert isinstance(qf, Variable)
        assert qf.units == "1/m"
        assert qf.limits == (-5.0, 5.0)
        assert qf.writable is True
        assert qf.readable is True

    def test_write_false(self, catalog):
        variables = catalog.build_variables()
        bpm = variables["BPM1:X"]
        assert bpm.writable is False
        assert bpm.readable is True

    def test_sorted_output(self, catalog):
        variables = catalog.build_variables()
        keys = list(variables.keys())
        assert keys == sorted(keys)


class TestQuery:
    def test_select_systems(self, catalog):
        rows = catalog.query(
            "SELECT DISTINCT system FROM devices ORDER BY system"
        )
        names = [r["system"] for r in rows]
        assert names == ["diagnostics", "magnets"]

    def test_select_devices(self, catalog):
        rows = catalog.query(
            "SELECT device_id FROM devices WHERE device_type='quadrupole' ORDER BY device_id"
        )
        names = [r["device_id"] for r in rows]
        assert names == ["QD", "QF"]

    def test_select_attributes(self, catalog):
        rows = catalog.query(
            "SELECT a.variable, a.unit FROM attributes a "
            "JOIN devices d ON a.device_id = d.device_id WHERE d.system='magnets'"
        )
        variables = {r["variable"] for r in rows}
        assert variables == {"QF:K1", "QD:K1"}

    def test_rejects_insert(self, catalog):
        with pytest.raises(ValueError, match="Only SELECT"):
            catalog.query("INSERT INTO systems VALUES ('hack')")

    def test_rejects_update(self, catalog):
        with pytest.raises(ValueError, match="Only SELECT"):
            catalog.query("UPDATE systems SET name='hack'")

    def test_rejects_delete(self, catalog):
        with pytest.raises(ValueError, match="Only SELECT"):
            catalog.query("DELETE FROM systems")


class TestBrowse:
    def test_root(self, catalog):
        result = catalog.browse("/")
        assert result["path"] == "/"
        assert "diagnostics" in result["children"]
        assert "magnets" in result["children"]

    def test_system(self, catalog):
        result = catalog.browse("/magnets")
        assert result["path"] == "/magnets"
        assert "quadrupole" in result["children"]

    def test_device_type(self, catalog):
        result = catalog.browse("/magnets/quadrupole")
        names = [d["name"] for d in result["children"]]
        assert "QF" in names
        assert "QD" in names

    def test_device(self, catalog):
        result = catalog.browse("/magnets/quadrupole/QF")
        assert "K1" in result["children"]

    def test_attribute(self, catalog):
        result = catalog.browse("/magnets/quadrupole/QF/K1")
        assert result["path"] == "/magnets/quadrupole/QF/K1"
        assert result["variable"] == "QF:K1"
        assert result["units"] == "1/m"
        assert result["limits"] == [-5.0, 5.0]
        assert result["read"] is True
        assert result["write"] is True

    def test_monitor_attribute(self, catalog):
        result = catalog.browse("/diagnostics/monitor/BPM1/X")
        assert result["variable"] == "BPM1:X"
        assert result["read"] is True
        assert result["write"] is False

    def test_invalid_system(self, catalog):
        result = catalog.browse("/nonexistent")
        assert "error" in result

    def test_invalid_device(self, catalog):
        result = catalog.browse("/magnets/quadrupole/NONEXISTENT")
        assert "error" in result

    def test_too_deep(self, catalog):
        result = catalog.browse("/a/b/c/d/e")
        assert "error" in result

    def test_depth_2_from_root(self, catalog):
        result = catalog.browse("/", depth=2)
        # Children should be dicts with nested children
        magnets = next(c for c in result["children"] if c["name"] == "magnets")
        assert "children" in magnets
        type_names = [c if isinstance(c, str) else c["name"] for c in magnets["children"]]
        assert "quadrupole" in type_names

    def test_depth_3_from_system(self, catalog):
        result = catalog.browse("/magnets", depth=3)
        # Should go: system -> device_type -> device instances -> attributes
        quad = next(c for c in result["children"] if c["name"] == "quadrupole")
        qf = next(d for d in quad["children"] if d["name"] == "QF")
        assert "children" in qf
        # At depth=3, device children should be attribute names
        assert "K1" in [a if isinstance(a, str) else a["name"] for a in qf["children"]]

    def test_trailing_slash(self, catalog):
        result = catalog.browse("/magnets/")
        assert result["path"] == "/magnets"
        assert "quadrupole" in result["children"]
