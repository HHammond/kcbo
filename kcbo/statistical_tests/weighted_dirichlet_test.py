from kcbo.statistical_tests.utils import StatisticalTest, statistic
from kcbo.utils import output_templates

from collections import Iterable
from itertools import combinations
from tabulate import tabulate
import numpy
import pandas as pd


class DirichletMultinomial(StatisticalTest):
    TYPE = "Dirichlet Proportions"

    def __init__(self, *args, **kwargs):
        super(DirichletMultinomial, self).__init__(*args, **kwargs)

    def initialize_test(self, dataframe, groups=None, classes=None, groupcol='group', classcol='class', observationscol='observed', totalscol='total', alpha=None, samples=100000):
        self.dataframe = dataframe
        self.groupcol = groupcol
        self.classcol = classcol
        self.observationscol = observationscol
        self.totalscol = totalscol

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

        if alpha is None:
            alpha = numpy.ones(len(self.classes))
        self.alpha = alpha
        self.samples = samples

        self.group_posteriors = {}

    def run_model(self, *args, **kwargs):
        dataset = self.dataframe.set_index([self.groupcol, self.classcol])
        for group in self.groups:
            occurrences = dataset.ix[group][self.observationscol][self.classes]
            p_alpha = numpy.random.dirichlet(
                alpha=occurrences + self.alpha,
                size=self.samples
            )
            posteriors = pd.DataFrame(
                dict(zip(self.classes, p_alpha.transpose()))
            )
            self.group_posteriors[group] = posteriors

            self.complete_key(group)

    @statistic('proportions', is_distribution=True, is_estimate=True, individual=True)
    def group_proportion(self, group):
        return self.group_posteriors[group]

    def summary(self):
        stats = self.compute_statistics()
        return self.generate_text_description(stats), stats

    def generate_text_description(self, summary_data):

        group_summary_header = [
            'Group', 'Class', 'Estimated Proportion', '95% Credible Interval Lower', '95% Credible Interval Upper']

        tables = {}
        col_order = [
            'group', 'class', 'estimate', '95_CI lower', '95_CI upper']
        for group in self.groups:
            n = len(self.classes)
            estimates = pd.DataFrame(
                summary_data[group][
                    'estimate proportions'], columns=['estimate']
            )
            intervals = summary_data[group]['95_CI proportions']
            intervals.columns = ['95_CI lower', '95_CI upper']

            group_string = pd.DataFrame([[group] * n, list(self.classes)]).T\
                             .set_index(1)

            group_string.columns = ['group']

            frame = pd.concat([estimates, group_string], axis=1)
            frame = pd.concat([frame, intervals], axis=1)

            frame['class'] = frame.index
            frame = frame[col_order].set_index('group')

            tables[group] = tabulate(
                frame, group_summary_header, tablefmt='pipe')

        summary_tables = "\n\n".join(
            ["{}: \n\n{}".format(group, table) for (group, table) in tables.items()])

        description = output_templates['groups estimate'].format(
            title=self.TYPE,
            groups_header="Groups:",
            groups_string=", ".join(self.groups),
            groups_summary=summary_tables,
        )

        return description


class WeightedDirichletMultinomial(DirichletMultinomial):

    TYPE = "Weighted Derichlet Multonimial Test"

    def __init__(self, *args, **kwargs):
        super(WeightedDirichletMultinomial, self).__init__(*args, **kwargs)

    def initialize_test(self, dataframe, groups=None, classes=None, groupcol='group', classcol='class', observationscol='observed', totalscol='total', weights=None, weight_function=None, samples=100000, alpha=None):
        DirichletMultinomial.initialize_test(
            self, dataframe=dataframe, groups=groups, classes=classes, groupcol=groupcol,
            classcol=classcol, observationscol=observationscol, totalscol=totalscol, alpha=alpha, samples=samples)

        self.weight_function = weight_function
        if weights is None:
            weights = numpy.ones(len(self.classes))
        self.weights = weights

        # if len(self.weights) != len(self.classes):
            # raise ValueError("Weights do not match classes")

        self.keys = list(combinations(self.groups, 2))
        self.group_posteriors = {}

    @statistic('expectation', is_distribution=True, is_estimate=True, individual=True)
    def compute_expectation(self, group):
        if self.weight_function is not None:
            return self.group_posteriors[group].apply(self.weight_function, axis=1, raw=False, reduce=True)
        else:
            return self.group_posteriors[group].mul(self.weights, axis=1).sum(axis=1)

    @statistic('difference_expectation', is_distribution=True, is_estimate=True, pairwise=True)
    def compute_difference_of_expectations(self, key):
        return self.compute_expectation(key[1]) - self.compute_expectation(key[0])

    @statistic('difference_expectation_p_value', is_estimate=True, pairwise=True)
    def compute_difference_of_expectations_p_value(self, key):
        return (self.compute_difference_of_expectations(key) > 0).mean()

    def generate_text_description(self, summary_data):

        group_summary_header = [
            'Group', 'Estimated Value', '95% Credible Interval Lower', '95% Credible Interval Upper']
        group_summary_table_data = [
            [
                group,
                summary_data[group]['estimate expectation'],
                summary_data[group]['95_CI expectation'][0],
                summary_data[group]['95_CI expectation'][1],
            ]
            for group in self.groups]

        group_summary_table = tabulate(
            group_summary_table_data, group_summary_header, tablefmt="pipe")

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

        proportions = DirichletMultinomial.generate_text_description(
            self, summary_data)
        proportions = "\n".join(proportions.split('\n')[4:])

        description = output_templates['groups with comparison'].format(
            title=self.TYPE,
            groups_header="Groups:",
            groups_string=", ".join(self.groups),
            groups_summary="{}\n\n{}".format(proportions, group_summary_table),
            comparison_summary=group_comparison_table,
        )

        return description

def dirichlet_multinomial_proportions(dataframe, alpha, groups, classes, groupcol, classcol, observationscol, totalscol, samples, *args, **kwargs):
    return DirichletMultinomial(dataframe, groups=None, classes=None, groupcol='group', classcol='class', observationscol='observed', totalscol='total', alpha=None, samples=100000, *args, **kwargs)

def weighted_dirichlet_comparison_test(dataframe, groups=None, classes=None, groupcol='group', classcol='class', observationscol='observed', totalscol='total', weights=None, weight_function=None, samples=100000, alpha=None):
    return WeightedDirichletMultinomial(dataframe=dataframe, groups=groups, classes=classes, groupcol=groupcol, classcol=classcol, observationscol=observationscol, totalscol=totalscol, weights=weights, weight_function=weight_function, samples=samples, alpha=alpha).summary()
