import numpy as np
import pandas as pd
import pymc as pm

"""
Bayesian ANOAVA (BANOVA) test.
"""


def ANOVA_oneway(df, groups=None, groupcol='group', valuecol='value', samples=40000, burns=10000):
    """Bayesian Oneway Anova (BANOVA)

    This test is an implementation of Kruschke's Oneway ANOVA. The goal of this test is to create a Bayesian implementation of the traditional ANOVA.

    Given data in a table of 'group', 'value' we perform an ANOVA on the data.

    """

    # Filter dataframe to specified groups
    group_map = lambda x: x[groupcol] in groups
    data = df[df.map(groupmap)]

    n = data.shape[1]

    # Build the model
