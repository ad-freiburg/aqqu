"""
A few simple utitlity functions.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
from itertools import tee


def edit_distance(s1, s2):
    """
    Computes the edit distance between s1 and s2 ignoring casing
    >>> edit_distance('this is a house', 'This is not a house')
    4
    """
    s1 = s1.lower()
    s2 = s2.lower()
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def pairwise(iterable):
    """
    Pairs each element in iterable with the next one
    >>> list(pairwise(['a', 'b', 'c']))
    [('a', 'b'), ('b', 'c')]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def triplewise(iterable):
    """
    Ménage à trois for each element in iterable with the next one and the one
    after that
    >>> list(triplewise(['a', 'b', 'c', 'd']))
    [('a', 'b', 'c'), ('b', 'c', 'd')]
    """
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
