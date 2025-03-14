"""
Helper functions which don't sit anywhere nicely.
"""
__all__ = [
    'alphanum_key', 'readable_idxs', 'sorted_readable',
]

import re


def alphanum_key(s):
    """
    Turn a string into a list of chunks, each containing strings and numbers.
    Useful key for human-readable sorting.

    Parameters
    ----------
    s : str
        String to be converted.

    Returns
    -------
    list
        List of strings and numbers.

    """
    tryint = lambda c: int(c) if c.isdigit() else c
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def readable_idxs(unsorted):
    return sorted(range(len(unsorted)), key=lambda k: alphanum_key(unsorted[k]))


def sorted_readable(x):
    return sorted(x, key=alphanum_key)