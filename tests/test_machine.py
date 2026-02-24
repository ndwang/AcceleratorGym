import pytest

from accelerator_gym.core.machine import Machine
from accelerator_gym.core.variable import Variable


class TestMachineRegistry:
    def test_variables_from_devices(self, machine):
        variables = machine.variables
        assert "QF:K1" in variables
        assert "QD:K1" in variables
        assert "BPM1:X" in variables
        assert "BPM1:Y" in variables

    def test_variables_sorted(self, machine):
        keys = list(machine.variables.keys())
        assert keys == sorted(keys)

    def test_variables_returns_copy(self, machine):
        v1 = machine.variables
        v2 = machine.variables
        assert v1 is not v2

    def test_catalog_available(self, machine):
        assert machine.catalog is not None


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

    def test_set_not_writable(self, machine):
        with pytest.raises(ValueError, match="not writable"):
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

    def test_set_many_not_writable(self, machine):
        with pytest.raises(ValueError, match="not writable"):
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
