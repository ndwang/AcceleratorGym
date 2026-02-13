import pytest

from accelerator_gym.core.variable import Variable


class TestVariable:
    def test_create_basic(self):
        v = Variable(name="Q1:K1")
        assert v.name == "Q1:K1"
        assert v.description == ""
        assert v.units is None
        assert v.read_only is False
        assert v.limits is None

    def test_create_full(self):
        v = Variable(
            name="Q1:K1",
            description="Quad strength",
            units="1/m^2",
            read_only=False,
            limits=(-5.0, 5.0),
        )
        assert v.description == "Quad strength"
        assert v.units == "1/m^2"
        assert v.limits == (-5.0, 5.0)

    def test_invalid_limits_order(self):
        with pytest.raises(ValueError, match="lower limit .* exceeds upper limit"):
            Variable(name="Q1:K1", limits=(5.0, -5.0))

    def test_validate_value_within_limits(self):
        v = Variable(name="Q1:K1", limits=(-5.0, 5.0))
        v.validate_value(0.0)  # should not raise
        v.validate_value(-5.0)  # boundary
        v.validate_value(5.0)  # boundary

    def test_validate_value_below_limit(self):
        v = Variable(name="Q1:K1", limits=(-5.0, 5.0))
        with pytest.raises(ValueError, match="outside limits"):
            v.validate_value(-6.0)

    def test_validate_value_above_limit(self):
        v = Variable(name="Q1:K1", limits=(-5.0, 5.0))
        with pytest.raises(ValueError, match="outside limits"):
            v.validate_value(6.0)

    def test_validate_read_only(self):
        v = Variable(name="BPM1:X", read_only=True)
        with pytest.raises(ValueError, match="read-only"):
            v.validate_value(1.0)

    def test_validate_no_limits(self):
        v = Variable(name="Q1:K1")
        v.validate_value(999.0)  # should not raise

    def test_equal_limits(self):
        v = Variable(name="Q1:K1", limits=(0.0, 0.0))
        v.validate_value(0.0)  # should not raise
        with pytest.raises(ValueError):
            v.validate_value(0.1)

    def test_validate_rejects_bool(self):
        v = Variable(name="Q1:K1")
        with pytest.raises(TypeError, match="expected numeric value"):
            v.validate_value(True)

    def test_validate_rejects_string(self):
        v = Variable(name="Q1:K1")
        with pytest.raises(TypeError, match="expected numeric value"):
            v.validate_value("1.0")

    def test_validate_accepts_int(self):
        v = Variable(name="Q1:K1", limits=(-5.0, 5.0))
        v.validate_value(3)  # int within limits, should not raise
