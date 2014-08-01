#     import numpy as np
# import pandas as pd

# # MCMC Samplers
# import pymc as pm
# import pystan as stan


# def Pearson_Correlation(df, groups=None, samples=40000, burns=10000, thin=1, ** kwargs):
#     """Bayesian Pearons's Correlation Coeficient.

#     Compute Bayesian approximation of Person's correlation coefficient.

#     """

#     if groups is None:
#         groups = df.columns

#     if len(groups) != 2:
#         raise ValueError(
#             "Can only compute Pearson correlation on 2 groups, %s used." % len(groups))

#     # build model

#     mu1 = pm.Normal('mu1',
#                     mu=df[groups[0]].mean(),
#                     tau=1. / df[groups[0]].std()
#                     )

#     mu2 = pm.Normal('mu2',
#                     mu=df[groups[1]].mean(),
#                     tau=1. / df[groups[1]].std()
#                     )

#     sig1 = pm.Uniform('sigma1',
#                       df[groups[0]].var() / 100.,
#                       df[groups[0]].var() * 100.
#                       )

#     sig2 = pm.Uniform('sigma2',
#                       df[groups[1]].var() / 100.,
#                       df[groups[1]].var() * 100.
#                       )

#     rho = pm.Uniform('rho',
#                      lower=-1,
#                      upper=1,
#                      # get corr from 2x2 matrix computatoin
#                      value=df.corr().ix[0, 1]
#                      )

#     @pm.deterministic
#     def Tau(sig1=sig1, sig2=sig2, rho=rho):
#         # Compute Tau = Sigma^-1

#         a = sig1 ** 2
#         d = sig2 ** 2
#         b = c = rho * sig1 * sig2

#         T = 1. / (a * d - b * c) * np.matrix([[d, -b], [-c, a]])
#         return T

#     d = pm.MvNormal('d',
#                     mu=(mu1, mu2),
#                     tau = Tau,
#                     value = df,
#                     observed=True
#                     )

#     # Compose model
#     model = pm.Model({
#         'mu1': mu1,
#         'mu2': mu2,
#         'sig1': sig1,
#         'sig2': sig2,
#         'rho': rho,
#         'd': d,
#         'Tau': Tau
#     })

#     # Sample model
#     mcmc = pm.MCMC(model)
#     mcmc.sample(iter=samples, burn=burns, thin=thin)

#     # Get data out of sampler
#     rhos = mcmc.trace('rho')

#     # Determine if probability of 0 in distribution of rho
#     p = rhos > 0
#     p = max(1 - p, p)

#     return rhos, p


# def Generate_Random_Data(mus, sigmas, rho, n, colnames=['x', 'y'], seed=None):
#     """Generate random data for Pearson's Correlation testing.

#     params:
#         mus:    array of two values for distribution means
#         sigmas: arrya of two values for distribution standard error
#         rho:    correlation coefficient of data, between -1 and 1 inclusive
#         n:      length of dataset to be returned
#     returns:
#         2xn dataframe
#     """

#     cov_mat = np.matrix([
#         [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
#         [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2]])

#     # seed numpy random
#     if seed:
#         np.random.seed(seed)

#     rv = np.random.multivariate_normal(mean=mus, cov=cov_mat, size=n)
#     return pd.DataFrame(rv, columns=colnames)
