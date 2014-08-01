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

output_templates = {
    'groups estimate': """
            {title:^80}

            {groups_header} {groups_string}

            {groups_summary}
            """,
    'groups comparison': """
            {title:^80}

            {groups_header} {groups_string}

            {groups_summary}
            """,

    'groups with comparison': """
            {title:^80}

            {groups_header} {groups_string}

            Estimates:
            
            {groups_summary}

            Comparisions:

            {comparison_summary}
        """
}

output_templates = {k: "\n".join([line.lstrip() for line in v[1:].split('\n')])
                    for (k, v) in output_templates.items()}

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
