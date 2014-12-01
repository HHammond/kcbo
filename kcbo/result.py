from functools import partial
import textwrap

import numpy as np
from tabulate import tabulate

from .utils import Interval, html_table_inner


class ResultSet(dict):

    """A dict-like class to manage a collection of Results."""

    def __init__(self, model, raw, results):
        super(ResultSet, self).__init__()
        self.model = model
        self.raw = raw
        self.update(results)

    def __str__(self):
        out = []
        for v in self.values():
            out.append(str(v))
        return "\n".join(out)

    def __html__(self):
        """Format results as html.

        For Result object not supporting `__html__` or `_repr_html_`, wrap them
        with a `<pre>` block.
        """
        out = []
        text_format = "<pre>{}</pre>"
        for v in self.values():
            if hasattr(v, '__html__'):
                out.append(v.__html__())
            elif hasattr(v, '_repr_html_'):
                out.append(v._repr_html_())
            else:
                out.append(text_format.format(str(v)))

        return "<br />".join(out)

    def __repr__(self):
        """Represent resultset as strings."""
        return str(self)

    def _repr_html_(self):
        """Represent resultset as html blocks."""
        return self.__html__()

    def plot(self, sns, *args, **kwargs):
        """Plot results using sns instance.

        Iterator over plottable results. Returns tuples of the form:

            (<statistic>, <plot>)

        Note: requires seaborn to create plots.
        """
        def gen_plots():
            for title, result in self.items():
                if hasattr(result, 'plot'):
                    plot = result.plot(sns, *args, **kwargs)
                    yield title, plot

        if len(self.keys()) == 1:
            key, result = self.items()[0]
            return key, result.plot(sns)

        return gen_plots()


class Result(dict):

    """A Result is a dict-like class representing the output from a model.

    Results allow you to get a quick summary and individual metrics about
    statistical results.

    Result instances are NOT supposed to be used to process a model, only the
    output from a model. In order to create custom processing of a model you
    should create a custom Sampler class to handle your model's output.

    A Sampler should return a distribution and a Result should return summary
    information about that distribution with dict-like access.

    The default Result class is suitable for MCMC estimates which return a list
    or distribution of results. To create different Result types this class
    should be subclassed and the model. In general a custom Result only needs
    to extend dict.
    """

    def __init__(self, statistic, output):
        """Initialize Result object from statistical output."""
        super(Result, self).__init__()
        self._statistic = statistic
        self._value = output

    @property
    def name(self):
        return self._statistic.clean_name

    @property
    def value(self):
        return self._value


class PointEstimateResult(Result):

    """Result class representing output as a single value."""

    def __init__(self, statistic, output):
        super(PointEstimateResult, self).__init__(statistic, output)
        self['value'] = self.value

    # TODO: implement nice repl formatting


class NumericDistributionResult(Result):

    """Result class representing output as a numeric distribution."""

    def __init__(self, statistic, output):

        super(NumericDistributionResult, self).__init__(statistic, output)

        self['95% HPD interval'] = self._HPD_interval
        self['quantiles'] = self.quantiles
        self['median'] = self.median
        self['mean'] = self.mean
        self['standard deviation'] = self.std
        self['n'] = self.n

    @property
    def mean(self):
        return np.mean(self._value)

    @property
    def median(self):
        return np.median(self._value)

    @property
    def std(self):
        return np.std(self._value)

    @property
    def var(self):
        return np.var(self._value)

    @property
    def n(self):
        return self._value.shape[0]

    @property
    def quantiles(self):
        return self._quantiles()

    def _quantiles(self, points=(2.5, 25, 50, 75, 97.5)):
        quantiles = {}
        for q in points:
            quantiles[q] = np.percentile(self._value, q)
        return quantiles

    @property
    def _HPD_interval(self):
        return self.HPD_interval(95)

    def HPD_interval(self, alpha=95):
        return self.compute_interval(self._value, alpha)

    @staticmethod
    def compute_interval(distribution, alpha=95):
        """Compute a 100-alpha confidence interval from distributional data."""
        alpha_lower, alpha_upper = (
            100 - alpha) / 2., 100 - ((100 - alpha) / 2.)
        return Interval(np.percentile(distribution, alpha_lower),
                        np.percentile(distribution, alpha_upper))

    def plot(self, sns, *args, **kwargs):
        """Plot distribution with seaborn.

        This method is a thin wrapper around sns.distplot.

        The functionality of this method may be reproduced by using this Result
        object's `value` to extract the data being plotted here.
        """
        p = sns.distplot(self.value, norm_hist=True, *args, **kwargs)
        sns.plt.title(self.name)
        return p

    def _summary_table_rows(self):
        hpd = self.HPD_interval()

        table = [
            ['Estimate Mean', self.mean],
            ['Estimate Median', self.median],
            ['Estimate Variance', self.var],
            ['95% HPD Interval lower', hpd.lower],
            ['95% HPD Interval upper', hpd.upper]
        ]

        return table

    def _quantile_table_rows(self):
        table = []
        for q, v in sorted(self.quantiles.items()):
            table.append(["Quantile {}%".format(q), v])
        return table

    def __html__(self):

        description = self._statistic.description
        if description:
            paragraph_header = "<p><strong>{}</strong></p>"
            paragraph = "<p>{}</p>"

            # Wrap description paragraphs around width of this table
            paragraphs = [p for p in description.split("\n\n")]

            if paragraphs:
                paragraphs[0] = paragraph_header.format(paragraphs[0])
                if len(paragraphs) > 1:
                    paragraphs[
                        1:] = [
                        paragraph.format(p) for p in paragraphs[
                            1:]]

            description = "\n".join(paragraphs)

        summary = self._summary_table_rows()
        quantiles = self._quantile_table_rows()

        data = []

        for l, r in zip(summary, quantiles):
            data.append(l + r)

        inner = html_table_inner(data)

        table = """
        <table style='min-width:100%'>
        <tr><th colspan=4 style='text-align:center;'>{title}</th></tr>
        <tr><td colspan=4>{description}</td></tr>
        <tr>
            <th colspan=2>Summary Statistics</th>
            <th colspan=2>Quantiles</th>
        </tr>

        {inner}

        </table>
        """.format(title=self.name,
                   description=description,
                   inner=inner)

        return table

    def __str__(self, fmt='text'):

        build_table = partial(tabulate, floatfmt='0.6f', tablefmt='plain')

        summary = build_table(self._summary_table_rows()).split("\n")
        quantiles = build_table(self._quantile_table_rows()).split("\n")

        table_data = zip(summary, quantiles)
        table = tabulate(table_data, headers=["Summary", "Quantiles"])

        max_len = max(map(len, table.splitlines()))
        max_len = max(len(str(self.name)), max_len)
        max_len = max(max_len, 40)

        description = self._statistic.description

        out = []
        out.append(self.name.center(max_len))
        out.append("-" * max_len)

        if description:
            # Wrap description paragraphs around width of this table
            paragraphs = [textwrap.fill(p, max_len)
                          for p in description.split("\n\n")]
            out.append("\n\n".join(paragraphs))
            out.append("-" * max_len)

        out.append(table)
        out.append("\n")
        return "\n".join(out)

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        return self.__html__()
