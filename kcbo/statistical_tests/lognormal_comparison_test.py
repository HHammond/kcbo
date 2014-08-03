
from kcbo.statistical_tests.utils import StatisticalTest, statistic
from kcbo.utils import output_templates

import numpy as np
from tabulate import tabulate
from itertools import combinations





class LognormalMedianComparison(StatisticalTest):

    TYPE = 'Lognormal Median Comparison Test'

    def __init__(self, *args, **kwargs):
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

        tau = 1. / pooled_variance
        for group in groups:
            g = self.df[self.df[self.groupcol] == group][self.valuecol]
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

    @statistic('median', individual=True, is_distribution=True, is_estimate=True)
    def group_median_distribution(self, group):
        return self.median_distributions.get(group, [])

    @statistic('mu', individual=True, is_distribution=True, is_estimate=True)
    def group_mean_distribution(self, group):
        return self.mean_distributions.get(group, [])

    @statistic('diff_medians', is_distribution=True, pairwise=True, is_estimate=True)
    def diff_medians(self, groups):
        group1, group2 = groups
        return self.median_distributions[group2] - self.median_distributions[group1]

    @statistic('p_diff_medians', pairwise=True, is_estimate=True)
    def p_diff_medians(self, groups):
        return (self.diff_medians(groups) > 0).mean()

    def summary(self, *args, **kwargs):
        summary_data = self.compute_statistic(
            keys=list(self.keys).extend(self.groups))
        return self.generate_text_description(summary_data), summary_data

    def generate_text_description(self, summary_data):
        group_summary_header = [
            'Group', 'Median', '95% CI Lower', '95% CI Upper', 'Mu', '95% CI Lower', '95% CI Upper']
        group_summary_table_data = [
            [
                group,
                summary_data[group]['estimate median'],
                summary_data[group]['95_CI median'][0],
                summary_data[group]['95_CI median'][1],
                summary_data[group]['estimate mu'],
                summary_data[group]['95_CI mu'][0],
                summary_data[group]['95_CI mu'][1]
            ]
            for group in self.groups]

        group_summary_table = tabulate(
            group_summary_table_data, group_summary_header, tablefmt="pipe")

        comparisons_header = [
            "Hypothesis", "Difference of Medians", "P.Value", "95% CI Lower", "95% CI Upper"]
        comparisons_data = [
            [
                "{} < {}".format(*pair),
                self.diff_medians(pair).mean(),
                summary_data[pair]['p_diff_medians'],
                summary_data[pair]['95_CI diff_medians'][0],
                summary_data[pair]['95_CI diff_medians'][1],
            ] for pair in self.keys
        ]

        comparison_summary_table = tabulate(
            comparisons_data, comparisons_header, tablefmt="pipe")

        description = output_templates['groups with comparison'].format(
            title=self.TYPE,
            groups_header="Groups:",
            groups_string=", ".join(self.groups),
            groups_summary=group_summary_table,
            comparison_summary=comparison_summary_table,
        )

        return description


def lognormal_comparison_test(dataframe, groups=None, groupcol='group', valuecol='value', **kwargs):
    """Lognormal Median Comparison

    Given a dataframe of the form:

    |Group  |Observed Value|
    |-------|--------------|
    |<group>|       <float>|
    ...

    Compute estimates of the difference of medians between groups.

    Note: This test assumes that input comes from distributions with the same variance.

    Inputs:
    dataframe -- Pandas dataframe of form above
    groups -- (optional) list of groups to look at. Excluded looks at all groups
    groupcol -- string for indexing dataframe column for groups
    valuecol -- string for indexing dataframe column for values of observations

    Returns:
    (description, raw_data)
    description: table describing output data
    raw_data: dictionary of output data

    """

    results = LognormalMedianComparison(
        dataframe, groups=None, groupcol='group', valuecol='value', **kwargs)
    return results.summary()
