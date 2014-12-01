import inspect
from functools import partial

from . import result


class Statistic(object):

    """A class decorator statistical methods.

    Functions with the @Statistic decorator are functions which should be used
    to create an individual Result object from the output of a specific
    model after it is sampled or processed.

    Methods with this decorator are returned as staticmethods and are not bound
    to an instance or class by default. The reason for this is to allow copying
    of already created statistical functions from another class. Since Models
    are result factories anyways, there is no added benefit of have access to
    class attributes. In the case that a function should being bound to a
    class, the `fn_type` argument should be set to `classmethod`.
    """

    def __init__(self, result_type=result.Result, fn_type=staticmethod,
                 **kwargs):
        """Receive decorator arguments.

        Any keyword arguments used in the construction of the decorator will be
        translated into parameters of the wrapped function object.
        """
        self.params = {
            # Name of analysis converted to titlecase
            'name': '',
            'description': '',          # Description copied from funcdoc
            'is_statistic': True,       # Is this analysis a statistic?
            'result_type': result_type
        }

        # Set given flags to their proper values to be added to function
        self.params.update(kwargs)
        self.fn_type = fn_type

    def __call__(self, f):
        """Decorate function.

        Update function object params with the parameters passed to this
        decorator.
        """
        # Give function necessary parameters
        self.params['name'] = f.__name__
        self.params['clean_name'] = self._clean_name(f.__name__)
        self.params['description'] = inspect.getdoc(f)

        for k, v in self.params.items():
            setattr(f, k, v)

        return self.fn_type(f)

    @staticmethod
    def _clean_name(string):
        """Convert lower_case_words string into Title Case string."""
        return string.strip().replace('_', ' ').title()

# Some premade Statistics with specific Result types`

NumericDistribution = partial(
    Statistic,
    result_type=result.NumericDistributionResult)

PointEstimate = partial(
    Statistic,
    result_type=result.PointEstimateResult)
