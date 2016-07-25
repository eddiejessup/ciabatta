"""
Helper functions for constructing Python objects.
"""
from __future__ import (division, unicode_literals, absolute_import,
                        print_function)


def make_repr_str(obj, fields):
    """Make a string representing an object from a subset of its attributes.

    Can be used as a decent default implementation of __repr__.

    Works with well-behaved objects. May go wrong if there are circular
    references and general oddness going on.

    Returns a constructor-like string like "Class(`arg1`=arg1, ...)"

    Note that no object lookup is performed, other than to get the class name.
    It is up to the calling function to provide the fields, in the `fields`
    object.

    Parameters
    ----------
    obj: object
        The object to be represented.
    fields: list[tuple]
        Tuples of length 2, where
        the first element represents the name of the field, and
        the second its value.
    Returns
    -------
    repr_str: str
        String representing the object.
    """
    args = ', '.join(['{}={}'.format(*f) for f in fields])
    return '{}({})'.format(obj.__class__.__name__, args)
