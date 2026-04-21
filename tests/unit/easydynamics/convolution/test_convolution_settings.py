# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.settings.convolution_settings import ConvolutionSettings


class TestConvolutionSettings:
    @pytest.fixture
    def default_convolution_settings(self):
        return ConvolutionSettings()

    def test_init(self, default_convolution_settings):
        """
        Test initialization of ConvolutionSettings with default
        parameters.
        """
        # WHEN THEN EXPECT
        assert isinstance(default_convolution_settings, ConvolutionSettings)
        assert default_convolution_settings.upsample_factor == 5
        assert default_convolution_settings.extension_factor == pytest.approx(0.2)
        assert default_convolution_settings.convolution_plan_is_valid is False

    def test_init_with_custom_parameters(self):
        """
        Test initialization of ConvolutionSettings with custom
        parameters.
        """
        # WHEN
        convolution_settings = ConvolutionSettings(
            upsample_factor=10,
            extension_factor=0.5,
        )

        # THEN EXPECT
        assert convolution_settings.upsample_factor == 10
        assert convolution_settings.extension_factor == pytest.approx(0.5)
        assert convolution_settings.convolution_plan_is_valid is False

    def test_init_with_None(self):
        """
        Test initialization of ConvolutionSettings with custom
        parameters.
        """
        # WHEN
        convolution_settings = ConvolutionSettings(
            upsample_factor=None,
            extension_factor=None,
        )

        # THEN EXPECT
        assert convolution_settings.upsample_factor is None
        assert convolution_settings.extension_factor is None
        assert convolution_settings.convolution_plan_is_valid is False

    @pytest.mark.parametrize(
        'invalid_input, expected_exception, match',
        [
            ({'extension_factor': '0.2'}, TypeError, 'must be a number'),
            ({'extension_factor': -0.1}, ValueError, 'must be non-negative'),
            ({'upsample_factor': '5'}, TypeError, 'must be a numerical value or None'),
            ({'upsample_factor': 1.0}, ValueError, 'must be greater than 1'),
            ({'upsample_factor': 0.5}, ValueError, 'must be greater than 1'),
        ],
        ids=[
            'extension_factor_not_numeric',
            'extension_factor_negative',
            'upsample_factor_not_numeric',
            'upsample_factor_equal_1',
            'upsample_factor_less_than_1',
        ],
    )
    def test_init_raises_for_invalid_input(self, invalid_input, expected_exception, match):
        """
        Test that initialization raises appropriate exceptions for
        invalid input parameters.
        """
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            ConvolutionSettings(**invalid_input)

    @pytest.mark.parametrize(
        'value',
        [2, 5.0, None],
        ids=[
            'integer_valid',
            'float_valid',
            'none_valid',
        ],
    )
    def test_upsample_factor_setter_valid(self, default_convolution_settings, value):
        settings = default_convolution_settings

        # WHEN
        # Ensure it's True first so we can test the reset
        settings.convolution_plan_is_valid = True

        # THEN
        settings.upsample_factor = value

        # EXPECT
        expected = pytest.approx(float(value)) if value is not None else None
        assert settings.upsample_factor == expected
        assert settings.convolution_plan_is_valid is False

    @pytest.mark.parametrize(
        'value, expected_exception, match',
        [
            ('5', TypeError, 'must be a numerical value or None'),
            (1.0, ValueError, 'must be greater than 1'),
            (0.5, ValueError, 'must be greater than 1'),
        ],
        ids=[
            'not_numeric',
            'equal_1',
            'less_than_1',
        ],
    )
    def test_upsample_factor_setter_invalid(
        self,
        default_convolution_settings,
        value,
        expected_exception,
        match,
    ):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            default_convolution_settings.upsample_factor = value

    @pytest.mark.parametrize(
        'value',
        [0.0, 0.2, 1, 5.5],
        ids=[
            'zero',
            'typical_fraction',
            'integer',
            'float',
        ],
    )
    def test_extension_factor_setter_valid(self, default_convolution_settings, value):

        # WHEN
        default_convolution_settings.convolution_plan_is_valid = True

        # THEN
        default_convolution_settings.extension_factor = value

        # EXPECT
        assert default_convolution_settings.extension_factor == pytest.approx(float(value))
        assert default_convolution_settings.convolution_plan_is_valid is False

    @pytest.mark.parametrize(
        'value, expected_exception, match',
        [
            ('0.2', TypeError, 'must be a number'),
            (-0.1, ValueError, 'must be non-negative'),
        ],
        ids=[
            'not_numeric',
            'negative',
        ],
    )
    def test_extension_factor_setter_invalid(
        self,
        default_convolution_settings,
        value,
        expected_exception,
        match,
    ):

        # WHEN / THEN / EXPECT
        with pytest.raises(expected_exception, match=match):
            default_convolution_settings.extension_factor = value

    @pytest.mark.parametrize(
        'value',
        [True, False],
        ids=[
            'true',
            'false',
        ],
    )
    def test_convolution_plan_is_valid_setter_valid(
        self,
        default_convolution_settings,
        value,
    ):
        # WHEN
        default_convolution_settings.convolution_plan_is_valid = not value

        # THEN
        default_convolution_settings.convolution_plan_is_valid = value

        # EXPECT
        assert default_convolution_settings.convolution_plan_is_valid is value

    @pytest.mark.parametrize(
        'value, expected_exception, match',
        [
            ('True', TypeError, 'must be True or False'),
            (1, TypeError, 'must be True or False'),
            (None, TypeError, 'must be True or False'),
        ],
        ids=[
            'string',
            'int',
            'none',
        ],
    )
    def test_convolution_plan_is_valid_setter_invalid(
        self,
        default_convolution_settings,
        value,
        expected_exception,
        match,
    ):
        # WHEN / THEN / EXPECT
        with pytest.raises(expected_exception, match=match):
            default_convolution_settings.convolution_plan_is_valid = value

    def test_repr_default(self, default_convolution_settings):
        # WHEN
        repr_str = repr(default_convolution_settings)

        # EXPECT
        assert repr_str == ('ConvolutionSettings(upsample_factor=5.0, extension_factor=0.2)')

    def test_repr_reflects_updated_values(self, default_convolution_settings):
        # WHEN
        default_convolution_settings.upsample_factor = 3
        default_convolution_settings.extension_factor = 0.5

        repr_str = repr(default_convolution_settings)

        # EXPECT
        assert repr_str == ('ConvolutionSettings(upsample_factor=3.0, extension_factor=0.5)')
