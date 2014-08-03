from kcbo.statistical_tests.utils import StatisticalTest, statistic
from kcbo.utils import output_templates

import pymc as pm
import numpy as np
import pandas as pd
from tabulate import tabulate
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

        # self.groups = groups
        self.groups = []
        self.keys = list(combinations(groups, 2))

        # TODO: may have broken things with locals... fragile move
        self.df = dataframe
        self.groupcol = groupcol
        self.valuecol = valuecol
        self.pooling = pooling
        self.samples = samples
        self.burns = burns
        self.thin = thin

        self.progress_bar = kwargs.get('progress_bar',False)
        if self.delay_statistic != True:
            local_vars = locals()
            del local_vars['self']
            map(lambda x: self.run_model(key=x, **local_vars), self.keys)
            return [self.compute_statistic(statistic=x) for x in self.statistics]

    def run_model(self, key, df=None, groups=None, groupcol=None, valuecol=None, pooling=None, samples=None, burns=None, thin=None, ** kwargs):

        if key not in self.keys:
            return None

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

        group_map = lambda x: x[groupcol] in (groups or key)

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
        mcmc.sample(iter=samples, burn=burns, thin=thin, progress_bar=self.progress_bar)

        self.mcmcs[key] = mcmc
        self.complete_key(key)

        return mcmc

    @statistic('diff_means', pairwise=True, is_distribution=True, is_estimate=True)
    def get_diff_means(self, key):
        """Compute difference of means from sampler

        Returns Key2 - Key1
        """
        mus_1 = self.mcmcs[key].trace('mu_1')[:]
        mus_2 = self.mcmcs[key].trace('mu_2')[:]
        return mus_2 - mus_1

    @statistic('diff_sdev', pairwise=True, is_distribution=True, is_estimate=True)
    def get_diff_sdev(self, key):
        """Compute difference of standard deviations from sampler

        Returns Key2 - Key1
        """
        sigmas_1 = self.mcmcs[key].trace('sigma_1')[:]
        sigmas_2 = self.mcmcs[key].trace('sigma_2')[:]
        return sigmas_2 ** 0.5 - sigmas_1 ** 0.5

    @statistic('effect_size', pairwise=True, is_distribution=True, is_estimate=True)
    def get_effect_size(self, key):
        """Compute effect size from sampler

        Returns Key2 - Key1
        """
        mus_1 = self.mcmcs[key].trace('mu_1')[:]
        mus_2 = self.mcmcs[key].trace('mu_2')[:]
        sigmas_1 = self.mcmcs[key].trace('sigma_1')[:]
        sigmas_2 = self.mcmcs[key].trace('sigma_2')[:]
        return (mus_2 - mus_1) / (np.sqrt((sigmas_2 + sigmas_1) / 2.0))

    @statistic('normality', pairwise=True, is_distribution=True, is_estimate=True)
    def get_normality(self, key):
        """Return normality parameter"""
        return np.log(self.mcmcs[key].trace('nu')[:])

    @statistic('p_value', pairwise=True, is_distribution=False, is_estimate=True)
    def get_p_value(self, key):
        """"Return P(Key2 > Key1 | Observed Data)"""
        return (self.get_diff_means(key) > 0).mean()

    def summary(self, key=None):
        data = self.compute_statistic(key=self.keys)

        return self.generate_text_description(data), data

    def generate_text_description(self, summary_data):

        summary_tables = []
        for (parameter, title) in (
            ('diff_means', 'Difference of Means'),
            ('diff_sdev', 'Difference of S.Dev'),
            ('effect_size', 'Effect Size'),
        ):
            group_summary_header = [
                'Hypothesis', title, 'P.Value', '95% CI Lower', '95% CI Upper']
            group_summary_table_data = [
                [
                    "{} < {}".format(*pair),
                    summary_data[pair]['estimate {}'.format(parameter)],
                    (summary_data[pair]['diff_means'] > 0).mean(),
                    summary_data[pair]['95_CI {}'.format(parameter)][0],
                    summary_data[pair]['95_CI {}'.format(parameter)][1],
                ]
                for pair in self.keys]

            group_summary_table = tabulate(
                group_summary_table_data, group_summary_header, tablefmt="pipe")
            summary_tables.append(group_summary_table)

        summary_tables = "\n\n".join(summary_tables)

        description = output_templates['groups estimate'].format(
            title=self.TYPE,
            groups_header="",
            groups_string="",
            groups_summary=summary_tables,
        )

        return description


def t_test(df, groups=None, groupcol='group', valuecol='value', pooling='default', samples=40000, burns=10000, thin=1, *args, **kwargs):
    """Bayesian t-Test

    Given a dataframe of the form:

    |Group  |Observed Value|
    |-------|--------------|
    |<group>|       <float>|
    ...

    Perform pairwise t-Tests on groups

    Inputs:
    dataframe -- Pandas dataframe of form above
    groups -- (optional) list of groups to look at. Excluded looks at all groups
    groupcol -- string for indexing dataframe column for groups
    valuecol -- string for indexing dataframe column for values of observations
    pooling -- strategy for using pooled data in test. 
               * 'default' -- uses pairwise pooled data
               * 'all' -- uses pooled data from all groups
    samples -- number of samples to use in MCMC
    burns -- number of burns to use in MCMC
    thin -- thinning to use in MCMC
    progress_bar -- boolean, show progress bar of sampler (PyMC progress bar)

    Returns:
    (description, raw_data)
    description: table describing output data
    raw_data: dictionary of output data

    """
    test = TTest(df, groups, groupcol, valuecol,
                 pooling, samples, burns, thin, *args, **kwargs)
    return test.summary()
