import pymc as pm
import numpy as np
import pandas as pd

from itertools import combinations
from collections import namedtuple

from kcbo.Result import Result


def t_test(df, groups=None, groupcol='group', valuecol='value', pooling='default', samples=40000, burns=10000, thin=1, ** kwargs):
    """Bayesian implementation of standard t-Test.

       See http://blog.henryhhammond.com/bayesian-t-test/ for details.

       Options:
       - pooling:
        - 'all': uses pooled data from all data in df
        - 'pairs': uses pooled data from only pairs being compared
        - 'default': uses pooled data of all data in groups in df

        If you have 4 groups in your dataframe, A, B, C, and D, the behavior is:

        - all: A,B,C,D in pooled
        - paired: for A vs B pool only A and B
        - default: for comparing A,B,C pooled data will be A,B,C for every group
      - sampler options:
       - burns: <int> for start point for MCMC
       - samples: <int> for number of samples to take
       - thin: <int> rate at which sampler iterates through samples
    """

    # Named tuples for now, eventually this should change to work with Result
    # objects
    Data = namedtuple(
        'Data', ['mus_1', 'mus_2', 'sigmas_1', 'sigmas_2', 'nu', 'diff_mu', 'diff_sigma', 'effect', 'normality'])
    Statistics = namedtuple(
        'Statistics', ['difference_means', 'difference_variances', 'effect', 'p_value'])
    if not groups:
        groups = list(df[groupcol].unique())

    # Filter dataframe to match pooling options
    group_map = lambda x: x[groupcol] in groups

    # Setup Pooling behavior
    if pooling == 'all':
        pooled = df[valuecol]
    elif pooling == 'default':
        pooled = df[df.apply(group_map, axis=1)][valuecol]

    # Prepare MCMCs
    mcmcs = {}
    data = {}
    statistics = {}

    # Build and sample our models
    for group1, group2 in combinations(groups, 2):
        # unpack groups

        # Get group data
        g1 = df[df[groupcol] == group1][valuecol]
        g2 = df[df[groupcol] == group2][valuecol]

        # Get pooled Data
        if pooling == 'pairs':
            pooled = pd.DataFrame.concat([g1, g2])

        # Setup our priors
        mu1 = pm.Normal(
            "mu_1", mu=pooled.mean(), tau=1.0 / pooled.var() / 1000.0)
        mu2 = pm.Normal(
            "mu_2", mu=pooled.mean(), tau=1.0 / pooled.var() / 1000.0)
        sig1 = pm.Uniform(
            "sigma_1", lower=pooled.var() / 1000.0, upper=pooled.var() * 1000)
        sig2 = pm.Uniform(
            "sigma_2", lower=pooled.var() / 1000.0, upper=pooled.var() * 1000)
        v = pm.Exponential("nu", beta=1.0 / 29)

        t1 = pm.NoncentralT(
            "t_1", mu=mu1, lam=1.0 / sig1, nu=v, value=g1[:], observed=True)
        t2 = pm.NoncentralT(
            "t_2", mu=mu2, lam=1.0 / sig2, nu=v, value=g2[:], observed=True)

        # Push our priors into a model
        model = pm.Model([t1, mu1, sig1, t2, mu2, sig2, v])

        # Generate our MCMC object and run sampler
        mcmc = pm.MCMC(model)
        # TODO: make thin a variable specified by user
        mcmc.sample(iter=samples, burn=burns, thin=thin)

        mcmcs[(group1, group2)] = mcmc

        # Get distributions from MCMC
        mus_1 = mcmc.trace('mu_1')[:]
        mus_2 = mcmc.trace('mu_2')[:]
        sigmas_1 = mcmc.trace('sigma_1')[:]
        sigmas_2 = mcmc.trace('sigma_2')[:]
        nu = mcmc.trace('nu')[:]

        # compute summary statistics
        diff_mu = mus_2 - mus_1
        diff_sigma = sigmas_2 ** 0.5 - sigmas_1 ** 0.5
        effect = (mus_2 - mus_1) / (np.sqrt((sigmas_2 + sigmas_1) / 2.0))
        normality = np.log(nu)

        # compute directional pvalue
        pval = (effect > 0).mean()

        # Export to objects
        data[(group1, group2)] = Data(mus_1, mus_2, sigmas_1, sigmas_2, nu,
                       diff_mu, diff_sigma, effect, normality)
        statistics[(group1, group2)] = Statistics(
            (diff_mu > 0).mean(), (diff_sigma > 0).mean(), (effect > 0).mean(), pval)

    # TODO: Finish Results object and allow export to that format
    # results = Result('Bayesian t-Test')
    # results.data = data
    # results.statistics = statistics
    # results.groups = groups

    return data, statistics
