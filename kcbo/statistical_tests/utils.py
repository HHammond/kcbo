from kcbo.utils import listify, dictify
import numpy as np


class statistic(object):

    """Decorator for annotating a class method as a statistical computation.

    This allows the statistical test to run compute_statistic on a method.
    """

    def __init__(self, statistic_name=None):
        self.statistic_name = statistic_name

    def __call__(self, f):
        f.is_statistic = True
        f.statistic_name = self.statistic_name
        return f


class pairwise_statistic(statistic):

    def __init__(self, statistic_name=None):
        super(pairwise_statistic, self).__init__(statistic_name)


class StatisticalTest(object):

    """Base class for statistical testing objects

    Behaviours:

    __init__ -- Declare the class and setup necessary params
    __call__ -- Allows class to be called as a function
                Default behavior is to perform test actions on given args
    run_test -- Code to perform specific test
    compute_statistics -- Code to compute an individual statistic
    """

    TYPE = 'Generic Statistical Test'
    ALLOW_COMPLETED_KEYS = True

    def __init__(self, *args, **kwargs):

        self.data = None
        self.completed = []
        self.keys = None

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
                pass
            return None

        self.statistics = {}

        for value in map(lambda x: is_statistic(self, x), dir(self)):
            if value is not None:
                self.statistics[value[0]] = value[1]

        map(lambda s: setattr(self, s, self.statistics[s]), self.statistics)

    def run_model(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def compute_interval(distribution, alpha):
        alpha_lower, alpha_upper = (alpha/2.0, 1-alpha/2.0)
        return np.percentile(distribution, 100*alpha_lower), np.percentile(distribution, 100*alpha_upper)    

    @dictify
    def compute_statistic(self, key=None, statistic=None, **kwargs):
        if key is not None and not isinstance(key, type([])):
            keys = [key]
        else:
            keys = self.keys

        if statistic is None:
            statistic = self.statistics.keys()

        if statistic is not None and not isinstance(statistic, type([])):
            statistic = [statistic]

        for key in keys:
            data = {}

            if key in self.keys and key not in self.completed:
                self.run_model(key)

            for s in statistic:
                data[s] = self.statistics[s](key)
            
            yield key, data

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
