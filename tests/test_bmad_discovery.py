"""Tests for BmadBackend auto-discovery of lattice elements."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_tao():
    """A mock Tao instance returning a small lattice."""
    tao = MagicMock()

    # Small lattice: 2 quads, 1 sbend, 1 hkicker, 1 monitor, 1 sextupole, 1 drift
    names = ["D1", "QF", "D2", "BH1", "D3", "QD", "D4", "HC1", "BPM1", "SF1"]
    keys = [
        "Drift", "Quadrupole", "Drift", "Sbend", "Drift",
        "Quadrupole", "Drift", "Hkicker", "Monitor", "Sextupole",
    ]
    s_positions = ["0.5", "1.5", "2.0", "4.0", "4.5", "5.5", "6.0", "6.5", "7.0", "7.5"]

    tao.lat_list = MagicMock(side_effect=lambda pattern, what: {
        "ele.name": names,
        "ele.key": keys,
        "ele.s": s_positions,
    }[what])

    return tao


@pytest.fixture
def bmad_backend(mock_tao):
    """A BmadBackend with mocked pytao."""
    with patch.dict("sys.modules", {"pytao": MagicMock()}):
        from accelerator_gym.backends.bmad import BmadBackend
        backend = BmadBackend.__new__(BmadBackend)
        backend._settings = {}
        backend._tao = mock_tao
    return backend


class TestDiscoverDevices:
    def test_discovers_quadrupoles(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        quads = devices["magnets"]["quadrupole"]
        assert "QF" in quads
        assert "QD" in quads
        assert quads["QF"]["s_position"] == 1.5
        assert quads["QD"]["s_position"] == 5.5

    def test_quad_has_k1_attribute(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        attrs = devices["magnets"]["quadrupole"]["QF"]["attributes"]
        assert "K1" in attrs
        assert attrs["K1"]["write"] is True
        assert attrs["K1"]["read"] is True

    def test_discovers_sbends(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        sbends = devices["magnets"]["sbend"]
        assert "BH1" in sbends
        attrs = sbends["BH1"]["attributes"]
        assert "ANGLE" in attrs
        assert attrs["ANGLE"]["write"] is True

    def test_discovers_hkickers(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        hkickers = devices["magnets"]["hkicker"]
        assert "HC1" in hkickers
        attrs = hkickers["HC1"]["attributes"]
        assert "kick" in attrs
        assert attrs["kick"]["write"] is True

    def test_discovers_monitors(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        monitors = devices["diagnostics"]["monitor"]
        assert "BPM1" in monitors
        attrs = monitors["BPM1"]["attributes"]
        assert "orbit.x" in attrs
        assert "orbit.y" in attrs
        assert attrs["orbit.x"]["write"] is False

    def test_discovers_sextupoles(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        sextupoles = devices["magnets"]["sextupole"]
        assert "SF1" in sextupoles
        attrs = sextupoles["SF1"]["attributes"]
        assert "K2" in attrs

    def test_skips_drifts(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        # Drifts should not appear anywhere
        for system in devices.values():
            for device_type in system.values():
                for name in device_type:
                    assert not name.startswith("D") or name in ("DummyIgnore",), \
                        f"Drift element {name} should not be in device tree"

    def test_twiss_attributes_on_all_elements(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        # Every non-drift element should have Twiss attributes
        for attrs_dict in [
            devices["magnets"]["quadrupole"]["QF"]["attributes"],
            devices["magnets"]["sbend"]["BH1"]["attributes"],
            devices["diagnostics"]["monitor"]["BPM1"]["attributes"],
        ]:
            assert "beta.a" in attrs_dict
            assert "beta.b" in attrs_dict
            assert "phi.a" in attrs_dict
            assert "eta.x" in attrs_dict
            assert attrs_dict["beta.a"]["write"] is False

    def test_global_lattice_parameters(self, bmad_backend):
        devices = bmad_backend.discover_devices()
        ring = devices["global"]["lattice"]["params"]
        attrs = ring["attributes"]
        assert "tune.a" in attrs
        assert "tune.b" in attrs
        assert "chrom.a" in attrs
        assert "chrom.b" in attrs
        assert "e_tot" in attrs
        assert attrs["tune.a"]["write"] is False

    def test_user_attribute_override(self, mock_tao):
        """User-provided element_attributes override defaults."""
        with patch.dict("sys.modules", {"pytao": MagicMock()}):
            from accelerator_gym.backends.bmad import BmadBackend
            backend = BmadBackend.__new__(BmadBackend)
            backend._settings = {
                "element_attributes": {
                    "Sbend": [
                        {"name": "ANGLE", "units": "rad", "write": True},
                        {"name": "K1", "units": "1/m^2", "write": True},
                    ],
                },
            }
            backend._tao = mock_tao

        devices = backend.discover_devices()
        sbend_attrs = devices["magnets"]["sbend"]["BH1"]["attributes"]
        assert "ANGLE" in sbend_attrs
        assert "K1" in sbend_attrs
        assert sbend_attrs["K1"]["write"] is True


class TestResolveVariableName:
    def test_element_attribute(self, bmad_backend):
        result = bmad_backend.resolve_variable_name(
            "magnets", "quadrupole", "QF", "K1"
        )
        assert result == "ele::QF[K1]"

    def test_lattice_attribute_orbit(self, bmad_backend):
        result = bmad_backend.resolve_variable_name(
            "diagnostics", "monitor", "BPM1", "orbit.x"
        )
        assert result == "lat::orbit.x[BPM1]"

    def test_lattice_attribute_twiss(self, bmad_backend):
        result = bmad_backend.resolve_variable_name(
            "magnets", "quadrupole", "QF", "beta.a"
        )
        assert result == "lat::beta.a[QF]"

    def test_global_attribute(self, bmad_backend):
        result = bmad_backend.resolve_variable_name(
            "global", "lattice", "params", "tune.a"
        )
        assert result == "lat::tune.a"


class TestMachineAutoDiscovery:
    def test_empty_devices_triggers_discovery(self, mock_tao):
        """Machine uses discover_devices() when config.devices is empty."""
        with patch.dict("sys.modules", {"pytao": MagicMock()}):
            from accelerator_gym.backends.bmad import BmadBackend
            from accelerator_gym.core.config import MachineConfig
            from accelerator_gym.core.machine import Machine

            backend = BmadBackend.__new__(BmadBackend)
            backend._settings = {}
            backend._tao = mock_tao

            config = MachineConfig(
                name="Test",
                backend_type="bmad",
                devices={},  # empty — triggers discovery
            )

            machine = Machine(backend, config)
            # Should have discovered variables
            assert len(machine.variables) > 0
            # Check a known variable exists
            var_names = list(machine.variables.keys())
            assert any("QF" in name for name in var_names)

    def test_yaml_devices_take_precedence(self, mock_tao):
        """When config.devices is non-empty, discovery is not used."""
        with patch.dict("sys.modules", {"pytao": MagicMock()}):
            from accelerator_gym.backends.bmad import BmadBackend
            from accelerator_gym.core.config import MachineConfig
            from accelerator_gym.core.machine import Machine

            backend = BmadBackend.__new__(BmadBackend)
            backend._settings = {}
            backend._tao = mock_tao

            config = MachineConfig(
                name="Test",
                backend_type="bmad",
                devices={
                    "magnets": {
                        "quadrupole": {
                            "QF": {
                                "attributes": {
                                    "K1": {"read": True, "write": True},
                                },
                            },
                        },
                    },
                },
            )

            machine = Machine(backend, config)
            # Should only have the YAML-defined variable
            assert len(machine.variables) == 1
