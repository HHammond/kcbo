from kcbo.statistical_tests.utils import StatisticalTest, statistic
from itertools import combinations
from scipy.stats import beta


class BetaBinomialTest(StatisticalTest):


    TYPE = 'Beta-Binomial Conversion Rate Test'

    def __init__(self, *args, **kwargs):
        self.delay_statistic = kwargs.get('delay_statistics', True)
        super(type(self), self).__init__(*args, **kwargs)

    def initialize_test(self,dataframe, groups=None, groupcol='group', successcol='successes', totalcol='total', samples=100000, **kwargs):
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
            group_data = self.df[self.df[self.groupcol] == group]
            total = group_data[self.totalcol]
            successes = group_data[self.successcol]

            mc_data = beta.rvs(
                successes + 1., total - successes + 1., size=samples)
            self.distributions[group] = mc_data

    @statistic('difference')
    def difference(self, key):
        return self.distributions[key[1]] - self.distributions[key[0]]

    @statistic('difference_p_value')
    def difference_p_value(self, key):
        return (self.difference(key) > 0).mean()

    @statistic('difference_CI')
    def difference_CI(self, key, alpha=0.05):
        return self.compute_interval(self.difference(key), alpha)

    def summary(self):
        return self.compute_statistics()


def conversion_test(dataframe, groups=None, groupcol='group', successcol='conversions', totalcol='total', samples=100000, **kwargs):
    return BetaBinomialTest(dataframe, groups, groupcol, successcol, totalcol, samples, **kwargs).summary()
