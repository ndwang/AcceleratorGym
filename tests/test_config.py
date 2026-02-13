import textwrap
from pathlib import Path

import pytest

from accelerator_gym.core.config import MachineConfig, build_variables, load_config
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
        assert "Q1:K1" in cfg.definitions

    def test_minimal_config(self, tmp_yaml):
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
        assert cfg.name == "Minimal"
        assert "X" in cfg.definitions

    def test_empty_config(self, tmp_yaml):
        path = tmp_yaml("""\
            machine: {}
            backend:
              type: mock
        """)
        cfg = load_config(path)
        assert cfg.name == ""
        assert cfg.backend_type == "mock"
        assert cfg.definitions == {}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestBuildVariables:
    def test_builds_from_definitions(self):
        definitions = {
            "Q1:K1": {"description": "Quad 1", "units": "1/m^2", "limits": [-5.0, 5.0]},
            "BPM1:X": {"description": "Horizontal", "units": "mm", "read_only": True},
        }
        result = build_variables(definitions)
        assert "Q1:K1" in result
        assert result["Q1:K1"].description == "Quad 1"
        assert result["Q1:K1"].units == "1/m^2"
        assert result["Q1:K1"].limits == (-5.0, 5.0)
        assert result["BPM1:X"].read_only is True

    def test_defaults(self):
        result = build_variables({"X": {}})
        var = result["X"]
        assert var.description == ""
        assert var.units is None
        assert var.read_only is False
        assert var.limits is None

    def test_empty(self):
        assert build_variables({}) == {}

    def test_sorted_output(self):
        definitions = {"C": {}, "A": {}, "B": {}}
        result = build_variables(definitions)
        assert list(result.keys()) == ["A", "B", "C"]
