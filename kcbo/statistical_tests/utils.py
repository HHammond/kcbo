from kcbo.utils import listify, dictify
import numpy as np
import pandas as pd
from collections import Iterable
from itertools import chain
from tabulate import tabulate

class statistic(object):

    """Decorator for annotating a class method as a statistical computation.

    This allows the statistical test to run compute_statistic on a method.
    """

    def __init__(self, statistic_name=None, is_distribution=False, is_estimate=False, pairwise=False, individual=False, ignore_intervals=False, **kwargs):
        self.statistic_name = statistic_name

        self.is_distribution = is_distribution
        self.is_estimate = is_estimate

        self.pairwise = pairwise
        self.individual = individual

        self.ignore_intervals = ignore_intervals

        self.kwargs = kwargs

    def __call__(self, f):
        f.is_statistic = True
        f.statistic_name = self.statistic_name

        f.is_distribution = self.is_distribution
        f.is_estimate = self.is_estimate

        f.pairwise = self.pairwise
        f.individual = self.individual

        f.ignore_intervals = self.ignore_intervals

        if self.is_estimate:
            f.estimate_function = self.kwargs.get('estimate_function', np.mean)
        if self.pairwise:
            f.hypotheses = self.kwargs.get('hypothesis_string', self.base_pairwise_hypothesis)
        elif self.individual:
            f.hypotheses = self.kwargs.get('hypothesis_string', self.base_individual_hypothesis)
        
        return f

    @staticmethod
    def base_pairwise_hypothesis(key):
        return "{} < {}".format(*key)

    @staticmethod
    def base_individual_hypothesis(group):
        return "{} > 0".format(group)


class StatisticalTest(object):

    """Base class for statistical testing objects

    Behaviours:

    __init__ -- Declare the class and setup necessary params
    __call__ -- Allows class to be called as a function
                Default behavior is to perform test actions on given args
    run_test -- Code to perform specific test
    compute_statistics -- Code to compute an individual statistics
    """

    TYPE = 'Generic Statistical Test'
    ALLOW_COMPLETED_KEYS = True

    def __init__(self, *args, **kwargs):

        self.data = None
        self.completed = []
        self.keys = None
        self.groups = None

        self.initialize_statistics()
        self.initialize_test(*args, **kwargs)

    def initialize_test(self, *args, **kwargs):
        """Perform Test"""
        raise NotImplementedError()

    def initialize_statistics(self):
        """Load statistics from this test and set object properties to compute them"""

        def is_statistic(obj, f):
            try:
                obj.__getattribute__(f).is_statistic
                return obj.__getattribute__(f).statistic_name, obj.__getattribute__(f)
            except:
                return None

        self.statistics = {}
        for value in map(lambda func: is_statistic(self, func), dir(self)):
            if value is not None:
                self.statistics[value[0]] = value[1]

        self.distributions = {
            k: v for (k, v) in self.statistics.items() if v.is_distribution}

        map(lambda s: setattr(self, s, self.statistics[s]), self.statistics)

    def run_model(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def compute_interval(distribution, alpha):
        alpha_lower, alpha_upper = (alpha / 2.0, 1 - alpha / 2.0)
        lower_func = lambda dist: np.percentile(dist, 100 * alpha_lower)
        upper_func = lambda dist: np.percentile(dist, 100 * alpha_upper)

        if type(distribution) is pd.DataFrame:
            frame = pd.concat([distribution.apply(lower_func, axis=0), distribution.apply(upper_func, axis=0)], axis=1)
            return frame

        return lower_func(distribution), upper_func(distribution)

    def compute_statistic(self, keys=None, **kwargs):
        if keys is None:
            keys = list(chain((self.keys or []), self.groups)) or []
        
        data = {}
        for key in keys:
            key_data = {}

            if not self.complete_key(key):
                self.run_model(key)
            
            if isinstance(key, Iterable) and not (type(key) is str or type(key) is unicode) and key in self.keys:
                applicable_statistics = {k:v for (k,v) in self.statistics.items() if v.pairwise}
            else:
                applicable_statistics = {k:v for (k,v) in self.statistics.items() if v.individual}
            
            for name, statistic in applicable_statistics.items():
                key_data[name] = statistic(key)
                if statistic.is_distribution and not statistic.ignore_intervals:
                    key_data["95_CI {}".format(name)] = self.compute_interval(key_data[name],0.05)
                if statistic.is_estimate and statistic.is_distribution:
                    key_data["estimate {}".format(name)] = statistic.estimate_function(key_data[name])

            data[key] = key_data

        return data

    def generate_tables(self, data):
        raise NotImplementedError()

    def compute_statistics(self, *args, **kwargs):
        """Mirror compute_statistic"""
        return self.compute_statistic(*args, **kwargs)

    def complete_key(self, key):
        """Mark a key as completed/computed"""
        if key not in self.completed:
            self.completed.append(key)

    def summary(self):
        """Return quick summary, i.e. R style output"""
        raise NotImplementedError()
