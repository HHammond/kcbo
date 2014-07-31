import functools
import operator

def listify(f):
    """Convert generator output to a list"""

    @functools.wraps(f)
    def listify_helper(*args, **kwargs):
        output = f(*args, **kwargs)

        if kwargs.get('as_generator'):
            return output
        elif output is not None:
            return list(output)
        else:
            return []
    return listify_helper


def dictify(f):
    """Convert generator output to a dict"""

    @functools.wraps(f)
    def dictify_helper(*args, **kwargs):
        output = f(*args, **kwargs)

        if kwargs.get('as_generator'):
            return output
        elif output is not None:
            return dict(output)
        else:
            return {}

    return dictify_helper


def draw_four_col_table(columns, rows, width=80):
    header = "{0:<20}{1:>20}{2:>20}{3:>20}".format(*columns)
    body = []

    for row in rows:
        out = "{0:<20}{1:>20}{2:>20}{3:>20}".format(*row)
        body.append(out)

    return "{header}\n{divider}\n{body}".format(
        header=header,
        divider="=" * width,
        body="\n".join(body)
    )


def dict_merge(*args):
    d = {}
    keys = reduce(operator.add, (k.keys() for k in args))
    for key in keys:
        d[key] = []
        for k in args:
            if key in k:
                d[key] += [k[key]]
        if len(d[key]) == 1:
            d[key] = d[key][0]
    return d
