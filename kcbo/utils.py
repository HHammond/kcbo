import json

def multi_get(obj, keys, default=None):
    """An extension of an objects `.get` function.

    Get values by a chain or heirarchy of keys.
    """
    for key in keys:
        val = obj.get(key, None)
        if val is not None:
            return val
    return default


class Interval(object):

    """Simple class representing a closed bounded interval.

    This implementation is only useful for checking containment. Intervals are
    either wholly open or wholly closed.
    """

    def __init__(self, lower=float('-inf'), upper=float('inf'), closed=True):
        self.lower = lower
        self.upper = upper
        self.closed = closed

    def __contains__(self, v):
        """Check this interval for containment of number or Interval."""
        if isinstance(v, (int, long, float, complex)):
            if self.closed:
                return self.lower <= v <= self.upper
            else:
                return self.lower < v < self.upper

        elif isinstance(v, Interval):
            return v.lower in self and v.upper in self

        else:
            raise ValueError(
                "Interval can only contain numerics or Intervals.")

    def to_json(self):
        return json.dumps({
            'lower': self.lower,
            'upper': self.upper,
            'closed': self.closed
            })

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "({}, {})".format(self.lower, self.upper)


def combine_samplers(*samplers):
    """Join a collection of Monte-Carlo samplers to share input arguments.

    Used to combine multiple samplers needed in a Monte Carlo simulation into a
    single function which can be called and returl all needed samples.
    """
    def partials(*args, **kwargs):
        return [model(*args, **kwargs) for model in samplers]

    return partials


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def html_table_inner(data, floatfmt='0.5f'):
    """Create the inner data for an HTML table from input list.
    """
    out = []

    float_format = "{{:{fmt}}}".format(fmt=floatfmt)

    cell_template = "<td {format}>{content}</td>"
    align_right = "style='text-align: right;'"
    align_default = "style='text-align: left;'"
    for row in data:
        out.append("<tr>")
        for i, cell in enumerate(row):
            align = align_default

            if is_numeric(cell):
                cell = float_format.format(cell)
                align = align_right

            out.append(cell_template.format(content=cell, format=align))
        out.append("</tr>")
    return "\n".join(out)
