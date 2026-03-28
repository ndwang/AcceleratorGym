"""Tests for AccelBench task helper functions."""

import math

import numpy as np
import pytest

from accelbench.tasks._helpers import close, query_variables, global_variable
from accelbench.types import Env


class TestClose:
    def test_equal_values(self):
        assert close(1.0, 1.0) is True

    def test_within_rtol(self):
        assert close(1.0, 1.04) is True  # 4% < 5% default rtol

    def test_outside_rtol(self):
        assert close(1.0, 1.10) is False  # 10% > 5%

    def test_within_atol(self):
        assert close(1e-7, 0.0) is True  # diff < 1e-6 default atol

    def test_outside_atol(self):
        assert close(1e-5, 0.0) is False

    def test_custom_rtol(self):
        assert close(1.0, 1.09, rtol=0.10) is True
        assert close(1.0, 1.20, rtol=0.10) is False

    def test_custom_atol(self):
        assert close(0.0, 1e-4, atol=1e-3) is True
        assert close(0.0, 1e-2, atol=1e-3) is False

    def test_negative_values(self):
        assert close(-1.0, -1.04) is True

    def test_zero_both(self):
        assert close(0.0, 0.0) is True

    def test_delegates_to_math_isclose(self):
        a, b, rtol, atol = 10.0, 10.3, 0.05, 1e-6
        assert close(a, b, rtol, atol) == math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


class TestQueryVariables:
    def test_finds_quadrupoles(self, machine):
        env = Env(machine=machine, rng=np.random.default_rng(42))
        rows = query_variables(env, "quadrupole", "K1")
        assert len(rows) == 2
        device_ids = {r["device_id"] for r in rows}
        assert device_ids == {"QF", "QD"}

    def test_finds_monitors(self, machine):
        env = Env(machine=machine, rng=np.random.default_rng(42))
        rows = query_variables(env, "monitor", "X")
        assert len(rows) == 1
        assert rows[0]["device_id"] == "BPM1"

    def test_nonexistent_type_returns_empty(self, machine):
        env = Env(machine=machine, rng=np.random.default_rng(42))
        rows = query_variables(env, "sextupole", "K2")
        assert rows == []

    def test_nonexistent_attribute_returns_empty(self, machine):
        env = Env(machine=machine, rng=np.random.default_rng(42))
        rows = query_variables(env, "quadrupole", "K2")
        assert rows == []

    def test_returns_variable_column(self, machine):
        env = Env(machine=machine, rng=np.random.default_rng(42))
        rows = query_variables(env, "quadrupole", "K1")
        variables = {r["variable"] for r in rows}
        assert variables == {"QF:K1", "QD:K1"}


class TestGlobalVariable:
    """Test global_variable with a config that has a 'global' system."""

    @pytest.fixture
    def machine_with_globals(self, mock_backend):
        from accelerator_gym.core.config import MachineConfig
        from accelerator_gym.core.machine import Machine

        mock_backend._initial_state["tune.a"] = 0.25
        mock_backend._state["tune.a"] = 0.25

        config = MachineConfig(
            name="Test",
            description="Test",
            backend_type="mock",
            devices={
                "global": {
                    "lattice": {
                        "GLOBAL": {
                            "description": "Global parameters",
                            "attributes": {
                                "tune.a": {
                                    "description": "Horizontal tune",
                                    "read": True,
                                    "write": False,
                                },
                            },
                        }
                    }
                },
                "magnets": {
                    "quadrupole": {
                        "QF": {
                            "description": "Focusing quad",
                            "attributes": {
                                "K1": {
                                    "description": "Strength",
                                    "units": "1/m",
                                    "limits": [-5.0, 5.0],
                                    "read": True,
                                    "write": True,
                                }
                            },
                        }
                    }
                },
            },
        )
        return Machine(mock_backend, config)

    def test_finds_global_variable(self, machine_with_globals):
        env = Env(machine=machine_with_globals, rng=np.random.default_rng(42))
        var_name = global_variable(env, "tune.a")
        assert var_name == "GLOBAL:tune.a"

    def test_missing_global_raises(self, machine_with_globals):
        env = Env(machine=machine_with_globals, rng=np.random.default_rng(42))
        with pytest.raises(IndexError):
            global_variable(env, "nonexistent")
