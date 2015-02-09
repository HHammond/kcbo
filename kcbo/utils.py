import json
import numbers


def is_numeric(obj):
    return isinstance(obj, numbers.Number)


def multi_get(obj, keys, default=None):
    """An extension of an objects `.get` function.

    Get values by a chain or heirarchy of keys.
    """
    for key in keys:
        val = obj.get(key, default)
        if val is not default:
            return val
    return default


class Interval(object):

    """Simple class representing a closed bounded interval.

    This implementation is only useful for checking containment. Intervals are
    either wholly open or wholly closed.
    """

    def __init__(self, lower, upper, closed=False):
        self.lower = lower
        self.upper = upper
        self.closed = closed

        if not is_numeric(lower) or not is_numeric(upper):
            raise ValueError("Interval only accepts numeric bounds.")

        if self.lower > self.upper:
            raise ValueError("Lower bound is greater than upper bound.")

    def __contains__(self, v):
        """Check this interval for containment of number or Interval."""
        if is_numeric(v) and not isinstance(v, complex):
            if self.closed:
                return self.lower <= v <= self.upper
            else:
                return self.lower < v < self.upper

        elif isinstance(v, Interval):

            if v == self:
                return True

            if not self.closed and not v.closed:
                if self.lower == v.lower and v.upper in self:
                    return True
                if self.upper == v.upper and v.lower in self:
                    return True

            return v.lower in self and v.upper in self

        else:
            raise ValueError(
                "Interval can only contain numerics or Intervals.")

    def __eq__(self, other):
        return (self.closed == other.closed
                and self.lower == other.lower
                and self.upper == other.upper)

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
        for cell in row:
            align = align_default

            if is_numeric(cell):
                if not isinstance(cell, numbers.Integral):
                    cell = float_format.format(cell)
                align = align_right

            out.append(cell_template.format(content=cell, format=align))
        out.append("</tr>")
    return "\n".join(out)
