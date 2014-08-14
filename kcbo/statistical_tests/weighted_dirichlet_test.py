from kcbo.statistical_tests.utils import StatisticalTest, statistic
from kcbo.utils import output_templates

from itertools import combinations
from tabulate import tabulate
import numpy
import pandas as pd

class WeightedDirichletMultinomial(StatisticalTest):

    TYPE = "Weighted Derichlet Multonimial Test"

    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)

    def initialize_test(self, dataframe, groups=None, classes=None, groupcol='group', classcol='class', observationscol='observed', totalscol='total', weights=None, samples=100000, alpha=None):
        
        self.dataframe = dataframe

        # ensure that dataframe is in the correct format
        if (self.dataframe.groupby([groupcol])[observationscol].sum() != self.dataframe.groupby([groupcol])[totalscol].unique()).any():
            if (self.dataframe.groupby([groupcol])[totalscol].unique().map(len) != 1).any():
                raise ValueError(
                    "Invalid input dataframe. Total column contains multiple values for same group")
            else:
                raise ValueError(
                    "Invalid input dataframe. Sum of observations does not equal total")
        if (self.dataframe[observationscol] < 0).any():
            raise ValueError("Input dataframe cannot contain negative values")

        if groups is None:
            groups = dataframe[groupcol].unique()
        self.groups = groups

        if classes is None:
            classes = dataframe[classcol].unique()
        self.classes = classes

        self.groupcol = groupcol
        self.classcol = classcol
        self.observationscol = observationscol
        self.totalscol = totalscol

        if weights is None:
            weights = numpy.zeros(len(self.classes))
        self.weights = weights

        if alpha is None:
            alpha = numpy.ones(len(self.classes))
        self.alpha = alpha
        self.samples = samples

        self.keys = list(combinations(self.groups,2))
        self.group_posteriors = {}

    def run_model(self, *args, **kwargs):

        dataset = self.dataframe.set_index([self.groupcol, self.classcol])

        for group in self.groups:
            occurrences = dataset.ix[group][self.observationscol][self.classes]
            p_alpha = numpy.random.dirichlet(alpha=occurrences + self.alpha, size=self.samples)
            posteriors = pd.DataFrame(dict(zip(self.classes, p_alpha.transpose())))
            self.group_posteriors[group] = posteriors
            
    @statistic('expectation', is_distribution=True, is_estimate=True, individual=True)
    def compute_expectation(self, group):
        return self.group_posteriors[group].mul(self.weights, axis=1).sum(axis=1)

    @statistic('difference_expectation', is_distribution=True, is_estimate=True, pairwise=True)
    def compute_difference_of_expectations(self, key):
        return self.compute_expectation(key[1]) - self.compute_expectation(key[0])

    @statistic('difference_expectation_p_value', is_estimate=True, pairwise=True)
    def compute_difference_of_expectations_p_value(self, key):
        return (self.compute_difference_of_expectations(key) > 0).mean()

    def summary(self):
        stats = self.compute_statistics()
        return self.generate_text_description(stats), stats

    def generate_text_description(self, summary_data):

        group_summary_header = [
            'Group', 'Expectation Value', '95% Credible Interval Lower', '95% Credible Interval Upper']
        group_summary_table_data = [
            [
                group,
                summary_data[group]['estimate expectation'],
                summary_data[group]['95_CI expectation'][0],
                summary_data[group]['95_CI expectation'][1],
            ]
            for group in self.groups]

        group_summary_table = tabulate(group_summary_table_data, group_summary_header, tablefmt="pipe")

        group_comparison_header = [
            'Hypothesis', 'Difference of Expectation', 'P.Value', '95% Credible Interval Lower', '95% Credible Interval Upper']
        group_comparison_table_data = [
            [
                "{} < {}".format(*group),
                summary_data[group]['estimate difference_expectation'],
                summary_data[group]['difference_expectation_p_value'],
                summary_data[group]['95_CI difference_expectation'][0],
                summary_data[group]['95_CI difference_expectation'][1],
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


def weighted_dirichlet_comparison_test(dataframe, groups=None, classes=None, groupcol='group', classcol='class', observationscol='observed', totalscol='total', weights=None, samples=100000, alpha=None):
    return WeightedDirichletMultinomial(dataframe, groups, classes, groupcol, classcol, observationscol, totalscol, weights, samples, alpha).summary()
