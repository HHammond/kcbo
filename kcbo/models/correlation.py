from kcbo.model import ModelDecorator, MCMCModel
from kcbo.statistic import NumericDistribution

import numpy as np
import pymc as pm


@ModelDecorator
class Correlation(MCMCModel):
    defaults = {
        'samples': 10000,
        'burns': 1500,
        'thin': 1,
    }

    @classmethod
    def model(cls, x, y, **kwargs):
        """Bayesian Pearson Correlation model.

        x -- First series of data
        y -- Second series of data
        """

        normalize = lambda z: (z - z.mean()) / z.std()
        x = normalize(x)
        y = normalize(y)

        rho = pm.Uniform('rho', lower=-1, upper=1)

        @pm.deterministic
        def Tau(s1=x.std(), s2=y.std(), rho=rho):
            """Create precision matrix."""
            cov = np.matrix(
                [[s1 ** 2, s1 * s2 * rho], [s1 * s2 * rho, s2 ** 2]])
            return np.linalg.inv(cov)

        @pm.observed
        def obs(value=zip(x, y), mean=(x.mean(), y.mean()), prec=Tau):
            """Add MVNormal log-likelihoods."""
            return sum(pm.mv_normal_like(d, mean, prec) for d in value)

        return locals()

    @NumericDistribution()
    def rho_distribution(result):
        """Distribution of `rho` parameter.

        The `rho` parameter of this model indicates the strength of the
        relationship between classes. A value close to -1 or 1 indicates a
        (respectively) very strong negative or positive relationship. Values
        closer to zero indicate a weak relationship.
        """
        return result.rho.trace()
