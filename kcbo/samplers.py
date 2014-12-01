import pymc as pm

from .utils import multi_get, combine_samplers


class Sampler(object):

    """Base sampler class for processing models.

    Subclasses of Sampler must implement the `__call__` method which will take
    the output of a `Model`'s `model` function and transform it into usable
    output that can be used by statistical methods of that model.
    """

    defaults = {}

    def __init__(self, *kwargs):
        """Update the sampler's default arguments with any given."""
        self.defaults.update(kwargs)

    def __call__(self, *args, **kwargs):
        """Apply model to produce output.

        This method must be subclassed so that a sampler may be called on a
        model.
        """
        raise NotImplementedError()

    def get_or_default(self, obj, keyword, default=None):
        """Get keyword from object or return default value."""
        return obj.get(keyword, self.defaults.get(keyword, default))


class MAPSampler(Sampler):

    """PyMC maximum a posteriori sampler."""

    def __call__(self, model, *args, **kwargs):
        """Maximum a-priori estimate

        Use PyMC's MAP method to fit model
        """

        M = pm.MAP(model)
        M.fit()
        return M


class MCMCSampler(Sampler):

    """PyMC MCMC sampler."""

    defaults = {
        'samples': 40000,
        'burns': 1000,
        'thin': 1

    }

    combine_samplers = staticmethod(combine_samplers)

    def __call__(self, model, *args, **kwargs):
        """Markov-Chain Monte Carlo

        Use PyMC's MCMC method to fit model
        """

        samples = self.get_or_default(kwargs, 'samples')
        burns = self.get_or_default(kwargs, 'burns')
        thin = self.get_or_default(kwargs, 'thin')

        mcmc = pm.MCMC(model)
        mcmc.sample(samples, burns, thin)

        return mcmc


class MonteCarloSampler(Sampler):

    """Monte Carlo Sampler."""

    defaults = {
        'samples': 100000,
    }

    def __call__(self, model, *args, **kwargs):
        """Monte Carlo Method.

        Take in a function which accepts the parameter size as the number of
        samples in the MC simulation. Return the output of the simulation as
        determined by the function from the model.
        """
        samples = multi_get(
            kwargs, ['samples', 'size'], default=self.defaults['samples'])

        return model(size=samples)
