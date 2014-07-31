from __future__ import unicode_literals

from kcbo.statistical_tests.utils import StatisticalTest, statistic
from kcbo.utils import draw_four_col_table, dict_merge
from itertools import combinations
import pandas as pd
import numpy as np
import re


class LognormalMedianComparison(StatisticalTest):

    TYPE = 'Lognormal Median Comparison Test'

    def __init__(self, *args, **kwargs):
        self.delay_statistic = kwargs.get('delay_statistics', True)
        super(type(self), self).__init__(*args, **kwargs)

    def initialize_test(self, dataframe, groups=None, groupcol='group', valuecol='value', samples=100000, **kwargs):
        df = dataframe
        df = df[df[valuecol] > 0]

        if not groups:
            groups = df[groupcol].unique()

        pooled = df[valuecol]

        self.pooled = pooled
        self.groups = groups
        self.groupcol = groupcol
        self.valuecol = valuecol
        self.df = df
        self.samples = samples

        self.keys = list(combinations(groups, 2))
        self.median_distributions = {}
        self.mean_distributions = {}

    def run_model(self, *args, **kwargs):

        groups = kwargs.get('groups', self.groups)

        m, v = (self.pooled.mean(), self.pooled.var())

        compute_mu = lambda m, v: np.log(m ** 2 / np.sqrt(v + m ** 2))
        compute_var = lambda m, v: np.log(v * 1. / m ** 2 + 1)

        pooled_mean = compute_mu(m, v)
        pooled_variance = compute_var(m, v)
        pooled_tau = 1. / 1000000

        mc_samples = self.samples

        for group in groups:
            g = self.df[self.df[self.groupcol] == group][self.valuecol]

            # estimate for this group
            mu = compute_mu(g.mean(), g.var())
            var = compute_var(g.mean(), g.var())
            tau = 1. / pooled_variance

            n = g.shape[0]

            # MC Simulation to generate distribution
            mean_data = np.random.normal(
                loc=(pooled_tau * pooled_mean + tau * np.log(g).mean() * n) /
                (pooled_tau + n * tau),
                scale= np.sqrt(1. / (pooled_tau + n * tau)),
                size=mc_samples
            )

            median_data = np.exp(mean_data)

            self.median_distributions[group] = median_data
            self.mean_distributions[group] = mean_data

    def group_median_distribution(self, group):
        return self.median_distributions.get(group, [])

    def group_mean_distribution(self, group):
        return self.mean_distributions.get(group, [])

    @statistic('diff_medians')
    def diff_medians(self, groups):
        group1, group2 = groups
        return self.median_distributions[group2] - self.median_distributions[group1]

    @statistic('p_diff_medians')
    def p_diff_medians(self, groups):
        return (self.diff_medians(groups) > 0).mean()

    def summary(self, *args, **kwargs):
        data = self.compute_statistic(key=self.keys)

        credible_intervals = {}

        for pair in self.keys:
            credible_intervals[pair] = {}
            credible_intervals[pair]['95_CI diff_medians'] = self.compute_interval(
                data[pair]['diff_medians'], 0.05)

            data[pair] = {'p_diff_medians': data[pair]['p_diff_medians']}

        for group in self.groups:
            credible_intervals[group] = {}
            credible_intervals[group]['95_CI median'] = self.compute_interval(
                self.group_median_distribution(group), 0.05)
            credible_intervals[group]['95_CI mu'] = self.compute_interval(
                self.group_mean_distribution(group), 0.05)

            data[group] = {}
            data[group]['median'] = self.group_median_distribution(
                group).mean()
            data[group]['mu'] = self.group_mean_distribution(group).mean()

        summary_data = {}
        for k, v in dict_merge(data, credible_intervals).items():
            if type(v) is list:
                d = v[0]
                for w in v:
                    d.update(w)
                v = d

            summary_data[k] = v

        return summary_data

def lognormal_comparison_test(dataframe, groups=None, groupcol='group', valuecol='value', **kwargs):
    results = LognormalMedianComparison(
        dataframe, groups=None, groupcol='group', valuecol='value', **kwargs)
    return results.summary()
