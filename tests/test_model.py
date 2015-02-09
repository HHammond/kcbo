import pytest

from kcbo.model import Model, ModelDefinitionError, ModelDecorator
from kcbo.samplers import Sampler
from kcbo.statistic import Statistic, PointEstimate
from kcbo.result import ResultSet, PointEstimateResult


class SimpleSampler(Sampler):

    """Fixture to evaluate simple models."""

    def __call__(self, model, *args, **kwargs):
        return model()


@ModelDecorator
class ValidModel(Model):
    application_method = SimpleSampler

    @classmethod
    def model(cls, a, b):
        """Test Model."""
        def inner():
            return a + b

        return inner

    @PointEstimate()
    def v(m):
        return m

    @PointEstimate(fn_type=classmethod)
    def w(cls, m):
        return cls.v(m) + 1


class ArgInspectSampler(Sampler):

    """Fixture to investigate options passed to sampler."""

    def __call__(self, model, *args, **kwargs):
        return kwargs


class ArgInspectModel(Model):
    application_method = ArgInspectSampler

    defaults = {
        'regular_default': 1,
        'overridden_default': 2
    }

    model = ValidModel.model

    @PointEstimate()
    def settings(settings):
        return settings


class TestModelDefinition(object):

    """Test suite for Model class."""

    def test_no_model_error(self):
        """Test that a model with no model function fails."""

        class ModelFunctionMissingModel(Model):
            application_method = Sampler

        error_msg = "This model does not implement the required `Model.model` "\
                    "method."

        try:
            ModelFunctionMissingModel()
            assert False
        except ModelDefinitionError as e:
            assert e.message == error_msg

    def test_bad_model_args(self):
        """Test that a model with varargs fails."""

        class BadVarargModel(Model):
            application_method = SimpleSampler

            @classmethod
            def model(a, *d):
                return lambda: None

        error_msg = "Model function cannot take varargs."

        try:
            BadVarargModel()
            assert False
        except ModelDefinitionError as e:
            assert e.message == error_msg

    def test_incompatible_statistic(self):
        """Test that incompatible statistical functions fail."""
        class IncompatibleStatisticModel(Model):
            application_method = SimpleSampler

            @classmethod
            def model(cls, a, b):
                return lambda: a + b

            @Statistic()
            def fail(w):
                raise Exception()

        error_msg = "Model has statistical functions which are incompatible "\
                    "with this model's application method. There may be an "\
                    "error in how the statistical method works or you may "\
                    "need to try subclassing this model to properly handle "\
                    "this application method. "

        try:
            IncompatibleStatisticModel(1, 2)
            assert False
        except ModelDefinitionError as e:
            assert e.message == error_msg

    def test_non_classmethod_model(self):
        """Test different types of model method.

        Instance methods should fail while static methods should succeed.
        """
        class InstanceMethodModel(Model):
            application_method = SimpleSampler

            def model(self, a, b):
                return lambda: a + b

            @PointEstimate()
            def val(w):
                return w

        error_msg = "unbound method model() must be called with "\
                    "InstanceMethodModel instance as first argument (got "\
                    "nothing instead)"

        try:
            InstanceMethodModel()
            assert False
        except ModelDefinitionError as e:
            assert e.message[0] == error_msg

        class StaticMethodModel(Model):
            application_method = SimpleSampler

            @staticmethod
            def model(a, b):
                return lambda: a + b

            @PointEstimate()
            def val(w):
                return w

        m = StaticMethodModel(1, 2)
        assert m['Val'].value == 3

    def test_model_decorator(self):
        assert ValidModel.__doc__ == ValidModel.model.__doc__

    def test_simple_model(self):
        m = ValidModel(1, 2)

        # Test that model runs
        assert m['V'].value == 3

        # Test classmethods work
        assert m['W'].value == 4

        # Test that return types are proper
        assert isinstance(m, ResultSet)
        assert isinstance(m['V'], PointEstimateResult)

    def test_settings(self):
        m = ArgInspectModel(1, 2, overridden_default=2, sample_setting=3)

        output = {
            'regular_default': 1,       # Test defualts passed properly
            'overridden_default': 2,    # Test overriding works
            'sample_setting': 3         # Test additional settings
        }

        # Make sure output is proper and settings have been passed exactly
        assert m['Settings'].value.keys() == output.keys()
        assert dict(m['Settings'].value) == output
