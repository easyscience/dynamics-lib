import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.convolution_base import (
    ConvolutionBase,
)
from easydynamics.sample_model import SampleModel


class TestConvolutionBase:
    @pytest.fixture
    def convolution_base(self):
        energy = np.linspace(-10, 10, 100)
        sample_model = SampleModel(name="SampleModel")
        resolution_model = SampleModel(name="ResolutionModel")
        offset = 0.0

        return ConvolutionBase(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=offset,
        )

    def test_init(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base, ConvolutionBase)
        assert isinstance(convolution_base.energy, sc.Variable)
        assert np.allclose(convolution_base.energy.values, np.linspace(-10, 10, 100))
        assert isinstance(convolution_base._sample_model, SampleModel)
        assert isinstance(convolution_base._resolution_model, SampleModel)
        assert isinstance(convolution_base.offset, Parameter)
        assert convolution_base.offset.value == 0.0

    @pytest.mark.parametrize(
        "kwargs, expected_message",
        [
            (
                {
                    "energy": "invalid",
                    "sample_model": SampleModel(),
                    "resolution_model": SampleModel(),
                    "energy_unit": "meV",
                    "offset": 0.0,
                },
                "Energy must be",
            ),
            (
                {
                    "energy": np.linspace(-10, 10, 100),
                    "sample_model": "invalid",
                    "resolution_model": SampleModel(),
                    "energy_unit": "meV",
                    "offset": 0.0,
                },
                "`sample_model` is an instance of str, but must be a SampleModel or ModelComponent.",
            ),
            (
                {
                    "energy": np.linspace(-10, 10, 100),
                    "sample_model": SampleModel(),
                    "resolution_model": "invalid",
                    "energy_unit": "meV",
                    "offset": 0.0,
                },
                "`resolution_model` is an instance of str, but must be a SampleModel or ModelComponent.",
            ),
            (
                {
                    "energy": np.linspace(-10, 10, 100),
                    "sample_model": SampleModel(),
                    "resolution_model": SampleModel(),
                    "energy_unit": 123,
                    "offset": 0.0,
                },
                "Energy_unit must be ",
            ),
            (
                {
                    "energy": np.linspace(-10, 10, 100),
                    "sample_model": SampleModel(),
                    "resolution_model": SampleModel(),
                    "energy_unit": "meV",
                    "offset": "invalid",
                },
                "Offset must be a Number or Parameter.",
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            ConvolutionBase(**kwargs)

    @pytest.mark.parametrize(
        "energy, expected_energy",
        [
            (
                1,
                sc.array(dims=["energy"], values=[1.0], unit="meV"),
            ),
            (
                1.0,
                sc.array(dims=["energy"], values=[1.0], unit="meV"),
            ),
            (
                np.linspace(-5, 5, 50),
                sc.array(dims=["energy"], values=np.linspace(-5, 5, 50), unit="meV"),
            ),
            (
                sc.array(dims=["energy"], values=np.linspace(-5, 5, 50), unit="meV"),
                sc.array(dims=["energy"], values=np.linspace(-5, 5, 50), unit="meV"),
            ),
        ],
        ids=["int", "float", "np.ndarray", "scipp.Variable"],
    )
    def test_energy_setter(self, convolution_base, energy, expected_energy):
        # WHEN
        convolution_base.energy = energy

        # THEN
        assert sc.identical(convolution_base.energy, expected_energy)

    def test_energy_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match="Energy must be a Number, a numpy ndarray or a scipp Variable.",
        ):
            convolution_base.energy = "invalid"

    def test_energy_unit_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert convolution_base.energy.unit == "meV"

    def test_energy_unit_setter_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match="Use convert_unit to change the unit between allowed types ",
        ):
            convolution_base.energy_unit = "K"

    def test_convert_energy_unit(self, convolution_base):
        # WHEN THEN
        convolution_base.convert_energy_unit("eV")

        # EXPECT
        assert convolution_base.energy.unit == "eV"
        assert convolution_base.energy_unit == "eV"
        assert np.allclose(
            convolution_base.energy.values, np.linspace(-0.01, 0.01, 100)
        )

    def test_sample_model_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base.sample_model, SampleModel)

    def test_sample_model_setter(self, convolution_base):
        # WHEN
        new_sample_model = SampleModel(name="NewSampleModel")

        # THEN
        convolution_base.sample_model = new_sample_model

        # EXPECT
        assert convolution_base.sample_model == new_sample_model

    def test_sample_model_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match="`sample_model` is an instance of str, but must be a SampleModel or ModelComponent.",
        ):
            convolution_base.sample_model = "invalid"

    def test_resolution_model_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base.resolution_model, SampleModel)

    def test_resolution_model_setter(self, convolution_base):
        # WHEN
        new_resolution_model = SampleModel(name="NewResolutionModel")

        # THEN
        convolution_base.resolution_model = new_resolution_model

        # EXPECT
        assert convolution_base.resolution_model == new_resolution_model

    def test_resolution_model_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match="`resolution_model` is an instance of str, but must be a SampleModel or ModelComponent.",
        ):
            convolution_base.resolution_model = "invalid"

    def test_offset_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base.offset, Parameter)
        assert convolution_base.offset.value == 0.0

    def test_offset_setter_parameter(self, convolution_base):
        # WHEN
        new_offset = Parameter(value=2.5, name="offset", unit="meV")

        # THEN
        convolution_base.offset = new_offset

        # EXPECT
        assert convolution_base.offset == new_offset

    def test_offset_setter_numerical(self, convolution_base):
        "Make sure the offset unique name remains the same when setting the numerical value"
        # WHEN
        convolution_base.offset = 3.5
        old_offset_unique_name = convolution_base.offset.unique_name

        # THEN
        convolution_base.offset = 3.5

        # EXPECT
        assert convolution_base.offset.value == 3.5
        assert convolution_base.offset.unique_name == old_offset_unique_name
