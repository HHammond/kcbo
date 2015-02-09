from functools import partial

import numpy as np

from kcbo.model import ModelDecorator, MonteCarloModel
from kcbo.statistic import NumericDistribution
from kcbo.utils import combine_samplers, multi_get


@ModelDecorator
class BetaBinomialDifference(MonteCarloModel):
    defaults = {'samples': 100000}

    @classmethod
    def model(cls, group1_successes, group1_total,
              group2_successes, group2_total, **kwargs):
        """Beta-binomial difference model.

        Computes the probability that group1 converts better than group2 using
        Monte-carlo simulation.

        Args:
          group1_successes :
            Integer representing number of successful events from group 1
          group1_total :
            Integer representing total possible successes
          group2_successes :
            Similar to group1_successes
          group2_total :
            Similar to group2_total

        Keyword Arguments:
          alpha :
            Prior value on number of successes (for both groups). By default
            set to 1.
          beta :
            Prior value on number of failures (for both groups). Default is 1.
          alpha1 :
            Prior value on number of successes for group1. Used only if
            specified.
          beta1 :
            Prior value on number of failures for group1. Used only if
            specified.
          alpha2 :
            Similar to alpha1 for group2
          beta2 :
            Similar to beta1 for group2
        """

        alpha1 = multi_get(kwargs, ['alpha1', 'alpha'], default=1)
        beta1 = multi_get(kwargs, ['beta1', 'beta'], default=1)
        alpha2 = multi_get(kwargs, ['alpha2', 'alpha'], default=1)
        beta2 = multi_get(kwargs, ['beta2', 'beta'], default=1)

        group1 = cls._beta_binomial_sampler(
            group1_successes,
            group1_total,
            alpha1,
            beta1)

        group2 = cls._beta_binomial_sampler(
            group2_successes,
            group2_total,
            alpha2,
            beta2)

        return combine_samplers(group1, group2)

    @staticmethod
    def _beta_binomial_sampler(successes, total, alpha, beta):
        """Create Monte-carlo sampler for group."""
        a = successes + alpha
        b = total - successes + beta
        return partial(np.random.beta, a=a, b=b)

    @NumericDistribution()
    def difference_in_proportion(groups):
        """Probability of difference between groups.

        This should be interpreted as the probability of a difference between
        groups from this test. A positive value indicates that group 1
        outperformed group 2 and a negative value indicates that group 2
        performed better than group 1.
        """
        group1, group2 = groups
        return group1 - group2
