import textwrap
from pathlib import Path

import pytest

from accelerator_gym.core.config import MachineConfig, load_config


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
              init_file: test.init
            devices:
              magnets:
                quadrupole:
                  QF:
                    description: Focusing quad
                    attributes:
                      K1:
                        description: Strength
                        units: "1/m"
                        limits: [-5.0, 5.0]
        """)
        cfg = load_config(path)
        assert cfg.name == "Test"
        assert cfg.description == "A test machine"
        assert cfg.backend_type == "bmad"
        assert cfg.backend_settings == {"init_file": "test.init"}
        assert "magnets" in cfg.devices
        assert "quadrupole" in cfg.devices["magnets"]
        assert "QF" in cfg.devices["magnets"]["quadrupole"]

    def test_minimal_config(self, tmp_yaml):
        path = tmp_yaml("""\
            machine:
              name: Minimal
            backend:
              type: mock
            devices:
              systems:
                device_type:
                  X:
                    attributes:
                      val:
                        description: Just X
        """)
        cfg = load_config(path)
        assert cfg.name == "Minimal"
        assert "systems" in cfg.devices

    def test_empty_devices(self, tmp_yaml):
        path = tmp_yaml("""\
            machine: {}
            backend:
              type: mock
        """)
        cfg = load_config(path)
        assert cfg.name == ""
        assert cfg.backend_type == "mock"
        assert cfg.devices == {}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
