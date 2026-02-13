import textwrap
from pathlib import Path

import pytest

from accelerator_gym.core.config import (
    MachineConfig,
    filter_variables,
    load_config,
    merge_definitions,
)
from accelerator_gym.core.variable import Variable


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper to write a YAML string to a temp file and return the path."""

    def _write(content: str) -> Path:
        p = tmp_path / "test.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


class TestLoadConfig:
    def test_full_config(self, tmp_yaml):
        path = tmp_yaml("""\
            machine:
              name: Test
              description: A test machine
            backend:
              type: bmad
              lattice_file: test.bmad
            variables:
              discover:
                include:
                  - "Q*:K1"
                exclude:
                  - "Q0:*"
              definitions:
                "Q1:K1":
                  description: Quad 1
                  limits: [-5.0, 5.0]
        """)
        cfg = load_config(path)
        assert cfg.name == "Test"
        assert cfg.description == "A test machine"
        assert cfg.backend_type == "bmad"
        assert cfg.backend_settings == {"lattice_file": "test.bmad"}
        assert cfg.discover_include == ["Q*:K1"]
        assert cfg.discover_exclude == ["Q0:*"]
        assert "Q1:K1" in cfg.definitions
        assert cfg.discovery_enabled is True

    def test_no_discover(self, tmp_yaml):
        path = tmp_yaml("""\
            machine:
              name: Minimal
            backend:
              type: mock
            variables:
              definitions:
                "X":
                  description: Just X
        """)
        cfg = load_config(path)
        assert cfg.discovery_enabled is False
        assert cfg.discover_include == []

    def test_empty_config(self, tmp_yaml):
        path = tmp_yaml("""\
            machine: {}
            backend:
              type: mock
        """)
        cfg = load_config(path)
        assert cfg.name == ""
        assert cfg.backend_type == "mock"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestFilterVariables:
    def test_include_only(self):
        variables = [
            Variable(name="QF:K1"),
            Variable(name="QD:K1"),
            Variable(name="BPM1:X"),
            Variable(name="BPM1:Y"),
        ]
        result = filter_variables(variables, include=["Q*:K1"], exclude=[])
        assert [v.name for v in result] == ["QF:K1", "QD:K1"]

    def test_include_and_exclude(self):
        variables = [
            Variable(name="QF:K1"),
            Variable(name="QD:K1"),
            Variable(name="Q0:K1"),
        ]
        result = filter_variables(variables, include=["Q*:K1"], exclude=["Q0:*"])
        assert [v.name for v in result] == ["QF:K1", "QD:K1"]

    def test_no_matches(self):
        variables = [Variable(name="X:Y")]
        result = filter_variables(variables, include=["Z*"], exclude=[])
        assert result == []

    def test_multiple_patterns(self):
        variables = [
            Variable(name="QF:K1"),
            Variable(name="BPM1:X"),
            Variable(name="BPM1:Y"),
            Variable(name="SEXT:K2"),
        ]
        result = filter_variables(
            variables, include=["Q*:K1", "BPM*:X", "BPM*:Y"], exclude=[]
        )
        names = [v.name for v in result]
        assert "QF:K1" in names
        assert "BPM1:X" in names
        assert "BPM1:Y" in names
        assert "SEXT:K2" not in names


class TestMergeDefinitions:
    def test_overlay_existing(self):
        discovered = {
            "Q1:K1": Variable(name="Q1:K1", description="discovered", units="m"),
        }
        definitions = {
            "Q1:K1": {"description": "overridden", "limits": [-3.0, 3.0]},
        }
        result = merge_definitions(discovered, definitions)
        var = result["Q1:K1"]
        assert var.description == "overridden"
        assert var.units == "m"  # kept from discovery
        assert var.limits == (-3.0, 3.0)

    def test_new_from_definitions(self):
        discovered = {}
        definitions = {
            "NEW:VAR": {"description": "brand new", "units": "mm", "read_only": True},
        }
        result = merge_definitions(discovered, definitions)
        assert "NEW:VAR" in result
        var = result["NEW:VAR"]
        assert var.description == "brand new"
        assert var.read_only is True

    def test_empty_both(self):
        result = merge_definitions({}, {})
        assert result == {}

    def test_discovered_only(self):
        discovered = {"A": Variable(name="A", description="from backend")}
        result = merge_definitions(discovered, {})
        assert result["A"].description == "from backend"
