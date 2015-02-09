import sys
import inspect
import types

from . import samplers
from .result import ResultSet


class ModelDefinitionError(Exception):

    """Generic error type for incorrect model defitions.

    This error should be thrown to indicate that a model was not defined
    as a proper subclass of `Model`
    """

    pass


class _ModelMetaClass(type):

    """Metaclass to allow Model objects to have writable docstrings."""

    pass


def ModelDecorator(model):
    """Update function docs for a Model.

    Take docstring from `Model.model` and apply it to `Model` so that the
    callable class has the proper docstring.
    """
    if not model.__doc__:
        model.__doc__ = model.model.__doc__
    return model


class Model(object):

    """Model class for defining a statistical model.

    The Model class acts as a Factory which produces ResultSet objects, which
    are a dict-like collection of Results, which are also dict-like.

    The Model class works like a function which creates ResultSet objects. When
    a model is called it will examine its arguments and decide how to run the
    `Model.model` function properly and with smart defaults. The result is that
    the model behaves like a function but follows an API that allows users to
    easily create models by specifying only the statistical methods (with the
    @Statistic decorator) and the model function.

    Because the Model class uses __new__ and works as a factory, all
    methods added to Model and its subclasses should be staticmethods,
    classmethods or tagged with an @Statistic decorator which will convert
    them into staticmethods.
    """

    # Metaclass
    __metaclass__ = _ModelMetaClass

    # Allow custom classes of Resultset objects to be implemented in models
    _ResultSet = ResultSet

    # Required model metadata
    application_method = None
    defaults = {}

    def __new__(cls, *model_positional_args, **kwargs):
        """Function that acts as a ResultSet factory from this Model."""
        settings = cls._get_settings(kwargs)
        method = cls._get_method(settings)
        model = cls._call_model(model_positional_args, settings)
        model_output = cls._generate_model_output(method, model, settings)
        results = cls._generate_statistics(model_output)
        return cls._ResultSet(model=cls, raw=model_output, results=results)

    @classmethod
    def model(cls):
        """The model declaration function.

        The model method is the method that statistics should be generated
        from. This method should return a function which can be called at a
        later time (when the Model class is called) which will generate the
        data needed to create a Resultset from this Model's statistical
        methods.

        `Model.model` is called and should return the model data, which is then
        passed to `Model.generate_samples` which will extract the actual useful
        data from the model. For most of the default models this is a sampler,
        either from PyMC or an MCMC sampler.
        """
        raise NotImplementedError()

    @classmethod
    def _call_model(cls, args, kwargs):
        """Safely call model.

        This function simply makes defining models cleaner by not requiring all
        model functions accept arbitrary keyword arguments.

        Note: Because of how arguments are handled, positional arguments in a
        model can only be specified by position. e.g.:

        >>>def model(a,b):
        >>>    return a,b
        >>>
        >>>_call_model(b=1, a=2)  # Fails
            ModelDefinitionError
        >>>_call_model(1,2)       # succeeds
            1,2

        Instead the model should use keyword arguments for this behaviour.
        """
        argspec = inspect.getargspec(cls.model)

        if (not isinstance(cls.model, types.FunctionType)
                and cls.model.__func__ == Model.model.__func__):
            raise ModelDefinitionError(
                "This model does not implement the required `Model.model` "
                "method.")

        if argspec.varargs:
            raise ModelDefinitionError("Model function cannot take varargs.")

        try:
            if argspec.keywords is None:
                return cls.model(*args)
            else:
                return cls.model(*args, **kwargs)
        except TypeError as e:
            raise ModelDefinitionError(e), None, sys.exc_info()[2]

    @classmethod
    def _get_settings(cls, settings):
        """Return model settings.

        Return dictionary of settings with user settings overriding default
        settings.
        """
        return dict(cls.defaults.items() + settings.items())

    @classmethod
    def _get_method(cls, settings):
        """Validate and return model application method.

        Ensure that the application method chosen for this method is valid and
        return method.
        """
        method = settings.get('method', cls.application_method)

        if method is None:
            raise ValueError(
                "No model application method given or none exist for this "
                "model. This model might not have a default application "
                "method set or this function is being called improperly. ")

        return method

    @classmethod
    def _generate_statistics(cls, model_output):
        """Compute summary statistics for this model."""
        statistics = cls.get_statistics()
        summary = dict()
        for name, statistic in statistics.items():
            result_type = statistic.result_type
            try:
                summary[name] = result_type(statistic, statistic(model_output))
            except Exception as e:

                error = ModelDefinitionError(
                    "Model has statistical functions which are incompatible "
                    "with this model's application method. There may be an "
                    "error in how the statistical method works or you may "
                    "need to try subclassing this model to properly handle "
                    "this application method. ")

                # Raise error and stacktrace
                raise error, None, sys.exc_info()[2]

        return summary

    @classmethod
    def get_statistics(cls):
        """Return list of all statistical functions registered with model.

        Return list of all methods from this class which have the @Statistic
        decorator.
        """
        def get_function(fn):
            """Verify a function is a function (or at least callable)."""
            f = getattr(cls, fn, None)
            if f is not None and callable(f):
                return f

        def is_statistic(f):
            """Verify that a function is a statistic."""
            if getattr(f, 'is_statistic', False):
                return f

        statistics = {}
        for f in (get_function(f) for f in dir(cls)):
            if f is not None and is_statistic(f):
                statistics[f.clean_name] = f
        return statistics

    @classmethod
    def _generate_model_output(cls, method, model, settings):
        """Sample from model using selected sampling method."""
        sampler = method()
        if not isinstance(sampler, samplers.Sampler):
            raise TypeError(
                "This model's application method is not a valid sampler. All "
                "samplers must be subclasses of kcbo.samplers.Sampler.")

        return sampler(model, **settings)


class MonteCarloModel(Model):

    """Model using MonteCarloSampler."""

    application_method = samplers.MonteCarloSampler


class MCMCModel(Model):

    """Model using PyMC MCMCSampler."""

    application_method = samplers.MCMCSampler


class MAPModel(Model):

    """Model using PyMC MAPSampler."""

    application_method = samplers.MAPSampler
