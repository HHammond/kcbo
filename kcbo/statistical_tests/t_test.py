from kcbo.statistical_tests.utils import StatisticalTest, statistic

import pymc as pm
import numpy as np
import pandas as pd

from itertools import combinations


class TTest(StatisticalTest):

    TYPE = 'Bayesian t-Test'

    def __init__(self, *args, **kwargs):
        self.delay_statistic = kwargs.get('delay_statistics', True)
        super(type(self), self).__init__(*args, **kwargs)

    def initialize_test(self, dataframe, groups=None, groupcol='group', valuecol='value', pooling='default', samples=40000, burns=10000, thin=1, ** kwargs):
        group_map = lambda x: x[groupcol] in groups
        if groups is None:
            groups = list(dataframe[groupcol].unique())

        self.mcmcs = {}
        self.mcmc_vars = {}

        self.groups = groups
        self.keys = list(combinations(groups, 2))

        self.df = dataframe # TODO: may have broken things with locals... fragile move
        self.groupcol = groupcol
        self.valuecol = valuecol
        self.pooling = pooling
        self.samples = samples
        self.burns = burns
        self.thin = thin

        if self.delay_statistic != True:
            local_vars = locals()
            del local_vars['self']
            map(lambda x: self.run_model(key=x, **local_vars), self.keys)
            return [self.compute_statistic(statistic=x) for x in self.statistics]

    def run_model(self, key, df=None, groups=None, groupcol=None, valuecol=None, pooling=None, samples=None, burns=None, thin=None, ** kwargs):

        if df is None:
            df = self.df
        if groups is None:
            groups = self.groups
        if groupcol is None:
            groupcol = self.groupcol
        if valuecol is None:
            valuecol = self.valuecol
        if pooling is None:
            pooling = self.pooling
        if samples is None:
            samples = self.samples
        if burns is None:
            burns = self.burns
        if thin is None:
            thin = self.thin

        group_map = lambda x: x[groupcol] in groups

        if pooling == 'all':
            pooled = df[valuecol]
        elif pooling == 'default':
            pooled = df[df.apply(group_map, axis=1)][valuecol]

        group1, group2 = key

        # Get group data
        g1 = df[df[groupcol] == group1]
        g1 = g1[valuecol]
        g2 = df[df[groupcol] == group2]
        g2 = g2[valuecol]

        # Get pooled Data
        if pooling == 'paired':
            pooled = pd.DataFrame.concat([g1, g2])

        # Setup our priors
        mu1 = pm.Normal(
            "mu_1", mu=pooled.mean(), tau=1.0 / pooled.var() / 1000.0)
        mu2 = pm.Normal(
            "mu_2", mu=pooled.mean(), tau=1.0 / pooled.var() / 1000.0)
        sig1 = pm.Uniform(
            "sigma_1", lower=pooled.var() / 1000.0, upper=pooled.var() * 1000)
        sig2 = pm.Uniform(
            "sigma_2", lower=pooled.var() / 1000.0, upper=pooled.var() * 1000)
        v = pm.Exponential("nu", beta=1.0 / 29)

        # Posterior distribution
        t1 = pm.NoncentralT(
            "t_1", mu=mu1, lam=1.0 / sig1, nu=v, value=g1[:], observed=True)
        t2 = pm.NoncentralT(
            "t_2", mu=mu2, lam=1.0 / sig2, nu=v, value=g2[:], observed=True)

        # Push our priors into a model
        model = pm.Model([t1, mu1, sig1, t2, mu2, sig2, v])

        # Generate our MCMC object and run sampler
        mcmc = pm.MCMC(model)
        mcmc.sample(iter=samples, burn=burns, thin=thin)

        self.mcmcs[key] = mcmc
        self.complete_key(key)

        return mcmc

    def summary(self, key=None):
        data = self.compute_statistic(key=self.keys)

        # param_CI_estimates = {}
        # for key in keys:
        #     param_CI_estimates[key] = {'median': self.compute_interval()


        return data

    @statistic('diff_means')
    def get_diff_means(self, key):
        """Compute difference of means from sampler

        Returns Key2 - Key1
        """
        mus_1 = self.mcmcs[key].trace('mu_1')[:]
        mus_2 = self.mcmcs[key].trace('mu_2')[:]
        return mus_2 - mus_1

    @statistic('diff_sdev')
    def get_diff_sdev(self, key):
        """Compute difference of standard deviations from sampler

        Returns Key2 - Key1
        """
        sigmas_1 = self.mcmcs[key].trace('sigma_1')[:]
        sigmas_2 = self.mcmcs[key].trace('sigma_2')[:]
        return sigmas_2 ** 0.5 - sigmas_1 ** 0.5

    @statistic('effect_size')
    def get_effect_size(self, key):
        """Compute effect size from sampler

        Returns Key2 - Key1
        """
        mus_1 = self.mcmcs[key].trace('mu_1')[:]
        mus_2 = self.mcmcs[key].trace('mu_2')[:]
        sigmas_1 = self.mcmcs[key].trace('sigma_1')[:]
        sigmas_2 = self.mcmcs[key].trace('sigma_2')[:]
        return (mus_2 - mus_1) / (np.sqrt((sigmas_2 + sigmas_1) / 2.0))

    @statistic('normality')
    def get_normality(self, key):
        """Return normality parameter"""
        return np.log(self.mcmcs[key].trace('nu')[:])

    @statistic('p_value')
    def get_p_value(self, key):
        """"Return P(Key2 > Key1 | Observed Data)"""
        return (self.get_diff_means(key) > 0).mean()


def t_test(df, groups=None, groupcol='group', valuecol='value', pooling='default', samples=40000, burns=10000, thin=1, *args, **kwargs):
    test = TTest(df, groups, groupcol, valuecol,
                 pooling, samples, burns, thin, *args, **kwargs)
    return test.summary()
