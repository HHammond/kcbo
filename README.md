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

```
pip install kcbo
```

### Installation from Source:

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

* **Conversion Test or Beta-Binomial difference test:** test of conversion to success using the Beta-Binomial model. Popular in A/B testing and estimation.

Documentation for these tests will be available on Github soon.

## Example Usage

Import your tests

```python
from kcbo import lognormal_comparison_test, t_test, conversion_test
```

Simply call them

```
summary, data = conversion_test(dataframe, groupcol='group',successcol='successes',totalcol='trials')
```

See the IPython notebook for more:

http://nbviewer.ipython.org/github/HHammond/kcbo/blob/master/KCBO%20Demonstration.ipynb
