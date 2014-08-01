from kcbo.statistical_tests.utils import StatisticalTest, statistic
from kcbo.utils import output_templates
from itertools import combinations
from scipy.stats import beta
from tabulate import tabulate

class BetaBinomialTest(StatisticalTest):

    TYPE = 'Beta-Binomial Conversion Rate Test'

    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)

    def initialize_test(self, dataframe, groups=None, groupcol='group', successcol='successes', totalcol='total', samples=100000, **kwargs):
        df = dataframe
        self.df = df

        if not groups:
            groups = df[groupcol].unique()
        self.groups = groups
        self.groupcol = groupcol
        self.successcol = successcol
        self.totalcol = totalcol
        self.samples = samples

        self.keys = list(combinations(groups, 2))
        self.distributions = {}

    def run_model(self, *args, **kwargs):
        groups = kwargs.get('groups', self.groups)
        samples = kwargs.get('samples', self.samples)
        df = kwargs.get('df', self.df)

        for group in groups:
            group_data = df[df[self.groupcol] == group]
            total = group_data[self.totalcol]
            successes = group_data[self.successcol]

            mc_data = beta.rvs(
                successes + 1., total - successes + 1., size=samples)
            self.distributions[group] = mc_data

    @statistic('difference', is_distribution=True, is_estimate=True, pairwise=True)
    def difference(self, key):
        return self.distributions[key[1]] - self.distributions[key[0]]

    @statistic('difference_p_value', is_estimate=True, pairwise=True)
    def difference_p_value(self, key):
        return (self.difference(key) > 0).mean()

    @statistic('distribution', is_distribution=True, is_estimate=True, individual=True)
    def distribution(self, key):
        return self.distributions[key]

    def summary(self):
        stats = self.compute_statistics()

        for key in self.keys:
            stats[key]['distribution'] = stats[key]['difference']
            del stats[key]['difference']

        # return stats
        return self.generate_text_description(stats), stats
    
    def generate_text_description(self, summary_data):

        group_summary_header = [
            'Group', 'Estimate', '95% CI Lower', '95% CI Upper']
        group_summary_table_data = [
            [
                group,
                summary_data[group]['estimate distribution'],
                summary_data[group]['95_CI distribution'][0],
                summary_data[group]['95_CI distribution'][1],
            ]
            for group in self.groups]

        group_summary_table = tabulate(
            group_summary_table_data, group_summary_header, tablefmt="pipe")

        group_comparison_header = [
            'Hypothesis', 'Difference', 'P.Value', '95% CI Lower', '95% CI Upper']
        group_comparison_table_data = [
            [
                "{} < {}".format(*group),
                summary_data[group]['estimate difference'],
                summary_data[group]['difference_p_value'],
                summary_data[group]['95_CI difference'][0],
                summary_data[group]['95_CI difference'][1],
            ]
            for group in self.keys]

        group_comparison_table = tabulate(
            group_comparison_table_data, group_comparison_header, tablefmt="pipe")

        description = group_summary_table

        description = output_templates['groups with comparison'].format(
            title=self.TYPE,
            groups_header="Groups:",
            groups_string=", ".join(self.groups),
            groups_summary=group_summary_table,
            comparison_summary=group_comparison_table,
        )

        return description

def conversion_test(dataframe, groups=None, groupcol='group', successcol='conversions', totalcol='total', samples=100000, **kwargs):
    return BetaBinomialTest(dataframe, groups, groupcol, successcol, totalcol, samples, **kwargs).summary()
