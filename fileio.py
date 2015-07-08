from __future__ import absolute_import, division, print_function
import os
import subprocess


def get_git_hash():
    """Returns the shortened git SHA of the current commit."""
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.check_output(cmd).strip()


def makedirs_safe(dirname):
    """Make a directory, prompting the user if it already exists."""
    if os.path.isdir(dirname):
        s = raw_input('%s exists, overwrite? (y/n) ' % dirname)
        if s != 'y':
            raise Exception
    else:
        os.makedirs(dirname)


def makedirs_soft(dirname):
    """Make a directory, if it doesn't already exist. Otherwise do nothing."""
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def format_parameter(p):
    """Format a value as a string appropriate for use in a directory name.

    For use when constructing a directory name that encodes the parameters
    of a model. Specially handled type cases are,

    - `None` is represented as 'N'.

    - `bool` is represented as '1' or '0'.

    Parameters
    ----------
    p: various

    Returns
    -------
    p_str: str
        Formatted parameter.
    """
    if isinstance(p, float):
        return '{:.3g}'.format(p)
    elif p is None:
        return 'N'
    elif isinstance(p, bool):
        return '{:d}'.format(p)
    else:
        return '{}'.format(p)


def reprify(obj, fields):
    """Make a string representing an object from a subset of its attributes.

    Parameters
    --------
    obj: object
        The object which is to be represented.
    fields: list[str]
        Strings matching the object's attributes to include in the
        representation.

    Returns
    -------
    field_strs: list[str]
        Strings, each representing a field and its value,
        formatted as '`field`=`value`'
    """
    return ','.join(['='.join([f, format_parameter(obj.__dict__[f])])
                    for f in fields])
