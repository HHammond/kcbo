# KCBO

A Bayesian testing framework written in Python.

KCBO is a toolkit for anyone who wants to do Bayesian data analysis without worrying about the implementation details for a certain test.

Currently KCBO is very much alpha/pre-alpha software and only implements a three tests. [Here](https://github.com/HHammond/kcbo/blob/master/Objectives.md) is a list of future objectives for the project. I have high hopes that soon KCBO will implement all the usual tests and that it will grow into something very powerful for analysts.

## KCBO Philosophy

*The goal of KCBO is to provide an easy to use, Bayesian framework to the masses.*

The Bayesian philosophy and framework provide an excellent structure for both asking and answering questions. Bayesian statistics allow us to ask questions in a more natural manner and derive incredibly powerful solutions.

Researchers and analysts shouldn't spend hours reading academic papers and finding which conjugate priors they need, which type of sampler their MCMC should have, or when to use MC or MCMC. Software should take care of that computing and researchers should take care of producing insights.

The world is ready for a good, clean, and easy to use Bayesian framework. The goal of KCBO is to provide that framework.

## Installation

KCBO is available through PyPI and on [Github here](https://github.com/HHammond/kcbo).

### Installation from PyPI:

```bash
pip install kcbo
```

### Installation from Source:

```bash
git clone https://github.com/HHammond/kcbo
cd kcbo
make install
```
Or to install without the makefile:

```bash
git clone https://github.com/HHammond/kcbo
cd kcbo
python setup.py sdist install
```

If any of this fails, you may need to install numpy (`pip install numpy`) in order to install some dependencies of KCBO, then retry installing it.

### Currently Available Tests

There are currently three tests implemented in the KCBO library:

* **Lognormal-Difference of medians:** used to compare medians of log-normal distributed data with the same variance. The one caveat to this test is that since KCBO uses a conjugate prior to the lognormal and hence assumes that all data has the same variance.

* **Bayesian t-Test:** an implementation of Kruschke's t-Test used to compare differences in mean and standard deviation for normally distributed samples.

* **Beta-Binomial difference test:** test of conversion to success using the Beta-Binomial model. Popular in A/B testing and estimation.

* **Correlation Test:** a Bayesian interpretation of the classical Pearson's Correlation test. 

Documentation for these tests will be available on Github soon.

## Example Usage

Import your tests

```python
from kcbo import BetaBinomialDifference, Correlation, TTest
```

Simply call tests with IPython repr-friendly output:

```
In [10]: b = BetaBinomialDifference(150,300,200,300)

In [11]: b
Out[11]:
                  Difference In Proportion
------------------------------------------------------------
Probability of difference between groups.

This should be interpreted as the probability of a
difference between groups from this test. A positive value
indicates that group 1 outperformed group 2 and a negative
value indicates that group 2 performed better than group 1.
------------------------------------------------------------
Summary                            Quantiles
---------------------------------  -------------------------
Estimate Mean           -0.165825  Quantile 2.5%   -0.242767
Estimate Median         -0.166032  Quantile 25%    -0.192604
Estimate Variance        0.001564  Quantile 50%    -0.166032
95% HPD Interval lower  -0.242767  Quantile 75%    -0.139352
95% HPD Interval upper  -0.087717  Quantile 97.5%  -0.087717
```

When working in the IPython notebook output is formatted as an HTML table:

<table style='min-width:100%'>
<tr><th colspan=4 style='text-align:center;'>Difference In Proportion</th></tr>
<tr><td colspan=4>
<p><strong>Probability of difference between groups.</strong></p>
<p>
This should be interpreted as the probability of a difference between groups from this test. A positive value indicates that group 1 outperformed group 2 and a negative value indicates that group 2 performed better than group 1.
</p></td></tr>
<tr>
<th colspan=2>Summary Statistics</th>
<th colspan=2>Quantiles</th>
</tr>
<tr>
<td style='text-align: left;'>Estimate Mean</td>
<td style='text-align: right;'>-0.16542</td>
<td style='text-align: left;'>Quantile 2.5%</td>
<td style='text-align: right;'>-0.24191</td>
</tr>
<tr>
<td style='text-align: left;'>Estimate Median</td>
<td style='text-align: right;'>-0.16543</td>
<td style='text-align: left;'>Quantile 25%</td>
<td style='text-align: right;'>-0.19237</td>
</tr>
<tr>
<td style='text-align: left;'>Estimate Variance</td>
<td style='text-align: right;'>0.00156</td>
<td style='text-align: left;'>Quantile 50%</td>
<td style='text-align: right;'>-0.16543</td>
</tr>
<tr>
<td style='text-align: left;'>95% HPD Interval lower</td>
<td style='text-align: right;'>-0.24191</td>
<td style='text-align: left;'>Quantile 75%</td>
<td style='text-align: right;'>-0.13877</td>
</tr>
<tr>
<td style='text-align: left;'>95% HPD Interval upper</td>
<td style='text-align: right;'>-0.08792</td>
<td style='text-align: left;'>Quantile 97.5%</td>
<td style='text-align: right;'>-0.08792</td>
</tr>
</table>


Individual statistics from tests can be accessed as keys from a dictionary using the name of the statistic in capital case.

## Building Models

KCBO provides an API which allows easy creation of new statistical models and analysis. 

Models are defined by subclassing the `kcbo.models.Model` class and implementing the `model` method. The `model` method returns a context which is passed to a `Sampler` which processes the model and feeds the output to each of the statistical methods of the model. 

### Constructing a PyMC model:

For this example we will implement the disaster model from the PyMC 2 tutorial.

We have the original model from the tutorial: 

```python
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np

switchpoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')
early_mean = Exponential('early_mean', beta=1.)
late_mean = Exponential('late_mean', beta=1.)

@deterministic(plot=False)
def rate(s=switchpoint, e=early_mean, l=late_mean):
    ''' Concatenate Poisson means '''
    out = np.empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out

disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
```

To make this work with KCBO, we need to make it into a PyMC model. We do this by wrapping the code in a function as a closure which returns its `locals`: 

```python
def model(disasters_array):
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')
    early_mean = Exponential('early_mean', beta=1.)
    late_mean = Exponential('late_mean', beta=1.)
    
    @deterministic(plot=False)
    def rate(s=switchpoint, e=early_mean, l=late_mean):
        ''' Concatenate Poisson means '''
        out = np.empty(len(disasters_array))
        out[:s] = e
        out[s:] = l
        return out
    
    disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
    return locals()
```

The only thing left to do to make this a KCBO model is to make it into an `MCMCModel` and give it some statistical methods.

```python
from kcbo.model import MCMCModel

class DisasterModel(MCMCModel):
    
    @classmethod
    def model(cls, disasters_array):
        """Disaster model from PyMC tutorial."""
        
        switchpoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')
        early_mean = Exponential('early_mean', beta=1.)
        late_mean = Exponential('late_mean', beta=1.)
        
        @deterministic(plot=False)
        def rate(s=switchpoint, e=early_mean, l=late_mean):
            ''' Concatenate Poisson means '''
            out = np.empty(len(disasters_array))
            out[:s] = e
            out[s:] = l
            return out
        
        disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
        
        return locals()
```

This model runs but it doesn't give us anything useful yet. The last thing this model needs is summary statistics. The parameter of interest in this model is the switchpoint. Since the data we're interested in is a numeric array of samples it is best represented as a `NumericDistribution`. 

```python
from kcbo.statistic import NumericDistribution
from kcbo.model import MCMCModel

class DisasterModel(MCMCModel):
    
    @classmethod
    def model(cls, disasters_array):
        """Disaster model from PyMC tutorial."""
        
        switchpoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')
        early_mean = Exponential('early_mean', beta=1.)
        late_mean = Exponential('late_mean', beta=1.)
        
        @deterministic(plot=False)
        def rate(s=switchpoint, e=early_mean, l=late_mean):
            ''' Concatenate Poisson means '''
            out = np.empty(len(disasters_array))
            out[:s] = e
            out[s:] = l
            return out
        
        disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
        
        return locals()

    @NumericDistribution()
    def switchpoint(model):
        return model.switchpoint.trace()[:]
```

Now when you run the model `DisasterModel(disasters_array)` the model yields a summary of data from the switchpoint parameter.

```
                        Switchpoint                         
------------------------------------------------------------
Summary                            Quantiles
---------------------------------  -------------------------
Estimate Mean           39.994308  Quantile 2.5%   36.000000
Estimate Median         40.000000  Quantile 25%    39.000000
Estimate Variance        5.994378  Quantile 50%    40.000000
95% HPD Interval lower  36.000000  Quantile 75%    41.000000
95% HPD Interval upper  46.000000  Quantile 97.5%  46.000000
```

If we want to make the output more explicit we can update the docstring of the `switchpoint` method with a description of the model and the output table will provide a description of the parameter.

```
                        Switchpoint                         
------------------------------------------------------------
Estimate of changepoint date.

Changepoint date as number of years since 1851.
------------------------------------------------------------
Summary                            Quantiles
---------------------------------  -------------------------
Estimate Mean           39.936590  Quantile 2.5%   36.000000
Estimate Median         40.000000  Quantile 25%    39.000000
Estimate Variance        5.912518  Quantile 50%    40.000000
95% HPD Interval lower  36.000000  Quantile 75%    41.000000
95% HPD Interval upper  46.000000  Quantile 97.5%  46.000000
```

