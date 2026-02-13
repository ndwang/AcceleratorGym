import pytest

from accelerator_gym.core.machine import Machine
from accelerator_gym.core.config import MachineConfig
from accelerator_gym.core.variable import Variable


class TestMachineRegistry:
    def test_variables_from_definitions(self, machine):
        variables = machine.variables
        assert "QF:K1" in variables
        assert "QD:K1" in variables
        assert "BPM1:X" in variables
        assert "BPM1:Y" in variables

    def test_variables_with_discovery(self, mock_backend):
        """Discovery + filtering + definitions overlay."""
        mock_backend.set_discoverable([
            Variable(name="QF:K1", description="from backend"),
            Variable(name="QD:K1", description="from backend"),
            Variable(name="BPM1:X", description="from backend"),
            Variable(name="SEXT:K2", description="should be filtered out"),
        ])
        config = MachineConfig(
            discover_include=["Q*:K1", "BPM*:X"],
            discover_exclude=[],
            definitions={
                "QF:K1": {"description": "overridden", "limits": [-3.0, 3.0]},
            },
        )
        m = Machine(mock_backend, config)
        variables = m.variables

        # QF:K1 should have overridden description + limits from definitions
        assert variables["QF:K1"].description == "overridden"
        assert variables["QF:K1"].limits == (-3.0, 3.0)

        # QD:K1 should keep backend description
        assert variables["QD:K1"].description == "from backend"

        # BPM1:X from discovery
        assert "BPM1:X" in variables

        # SEXT:K2 should be filtered out
        assert "SEXT:K2" not in variables

    def test_variables_returns_copy(self, machine):
        v1 = machine.variables
        v2 = machine.variables
        assert v1 is not v2


class TestMachineGetSet:
    def test_get(self, machine):
        assert machine.get("QF:K1") == 0.5

    def test_get_unknown(self, machine):
        with pytest.raises(KeyError, match="Unknown variable"):
            machine.get("NONEXISTENT")

    def test_set(self, machine):
        machine.set("QF:K1", 1.0)
        assert machine.get("QF:K1") == 1.0

    def test_set_unknown(self, machine):
        with pytest.raises(KeyError, match="Unknown variable"):
            machine.set("NONEXISTENT", 1.0)

    def test_set_read_only(self, machine):
        with pytest.raises(ValueError, match="read-only"):
            machine.set("BPM1:X", 1.0)

    def test_set_above_limit(self, machine):
        with pytest.raises(ValueError, match="outside limits"):
            machine.set("QF:K1", 10.0)

    def test_set_below_limit(self, machine):
        with pytest.raises(ValueError, match="outside limits"):
            machine.set("QF:K1", -10.0)

    def test_set_at_limit(self, machine):
        machine.set("QF:K1", 5.0)
        assert machine.get("QF:K1") == 5.0
        machine.set("QF:K1", -5.0)
        assert machine.get("QF:K1") == -5.0


class TestMachineMany:
    def test_get_many(self, machine):
        result = machine.get_many(["QF:K1", "QD:K1"])
        assert result == {"QF:K1": 0.5, "QD:K1": -0.5}

    def test_get_many_unknown(self, machine):
        with pytest.raises(KeyError):
            machine.get_many(["QF:K1", "NONEXISTENT"])

    def test_set_many(self, machine):
        machine.set_many({"QF:K1": 1.0, "QD:K1": -1.0})
        assert machine.get("QF:K1") == 1.0
        assert machine.get("QD:K1") == -1.0

    def test_set_many_all_or_nothing(self, machine):
        """If one value violates limits, none should be applied."""
        with pytest.raises(ValueError):
            machine.set_many({"QF:K1": 1.0, "QD:K1": -100.0})
        # QF:K1 should not have been changed
        assert machine.get("QF:K1") == 0.5

    def test_set_many_read_only(self, machine):
        with pytest.raises(ValueError, match="read-only"):
            machine.set_many({"BPM1:X": 1.0})

    def test_set_many_unknown(self, machine):
        with pytest.raises(KeyError):
            machine.set_many({"NONEXISTENT": 1.0})


class TestMachineReset:
    def test_reset(self, machine):
        machine.set("QF:K1", 2.0)
        assert machine.get("QF:K1") == 2.0
        machine.reset()
        assert machine.get("QF:K1") == 0.5
