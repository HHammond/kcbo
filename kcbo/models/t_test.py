from kcbo.model import ModelDecorator, MCMCModel
from kcbo.statistic import NumericDistribution
from kcbo.samplers import MCMCSampler, MAPSampler

import numpy as np
import pymc as pm


@ModelDecorator
class TTest(MCMCModel):
    defaults = {'samples': 40000,
                'burns': 10000,
                'thin': 1}

    @staticmethod
    def model(x, y):
        """Bayesian t-Test.

        x -- First series of observed values
        y -- Second series of observed values
        """
        # Priors
        mu1 = pm.Normal("mu_1", mu=x.mean(), tau=1.0 / x.var() / 1000.)
        mu2 = pm.Normal("mu_2", mu=y.mean(), tau=1.0 / y.var() / 1000.)
        sig1 = pm.Uniform(
            "sigma_1", lower=x.var() / 1000., upper=x.var() * 1000.)
        sig2 = pm.Uniform(
            "sigma_2", lower=y.var() / 1000., upper=y.var() * 1000.)
        v = pm.Exponential("nu", beta=1.0 / 29.)

        # Posterior
        t1 = pm.NoncentralT(
            "t_1", mu=mu1, lam=1. / sig1, nu=v, value=x, observed=True)
        t2 = pm.NoncentralT(
            "t_2", mu=mu2, lam=1. / sig2, nu=v, value=y, observed=True)

        return locals()

    @NumericDistribution()
    def difference_of_means(result):
        """Distribution of difference in posterior mean between groups."""

        return (result.mu1.trace() - result.mu2.trace())

    @NumericDistribution()
    def difference_of_varience(result):
        """Distribution of difference in variance between groups."""
        return np.sqrt(result.sig1.trace()) - np.sqrt(result.sig2.trace())

    @NumericDistribution()
    def effect_size(result):
        """Normalized difference between groups.

        A large effect size indicates that the experiment groups have a high
        chance of a true difference.
        """
        return (result.mu1.trace() - result.mu2.trace()) / \
            (np.sqrt(result.sig1.trace()) - np.sqrt(result.sig2.trace()) / 2.)


@ModelDecorator
class TTestPooled(TTest):

    @staticmethod
    def model(x, y):
        """Bayesian t-Test model using pooled data as default model parameters.

        x -- First series of observed values
        y -- Second series of observed values
        """
        pooled = np.concatenate([x, y], axis=0)

        # Priors
        mu1 = pm.Normal(
            "mu_1",
            mu=pooled.mean(),
            tau=1.0 / (pooled.var() * 1000.0)
        )
        mu2 = pm.Normal(
            "mu_2",
            mu=pooled.mean(),
            tau=1.0 / (pooled.var() / 1000.0)
        )
        sig1 = pm.Uniform(
            "sigma_1", lower=pooled.var() / 1000.0, upper=pooled.var() * 1000)
        sig2 = pm.Uniform(
            "sigma_2", lower=pooled.var() / 1000.0, upper=pooled.var() * 1000)
        v = pm.Exponential("nu", beta=1.0 / 29)

        # Posterior
        t1 = pm.NoncentralT(
            "t_1", mu=mu1, lam=1.0 / sig1, nu=v, value=x, observed=True)
        t2 = pm.NoncentralT(
            "t_2", mu=mu2, lam=1.0 / sig2, nu=v, value=y, observed=True)

        return locals()
