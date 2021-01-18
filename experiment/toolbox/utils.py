#  Copyright (c) 2021. Robin Thibaut, Ghent University

import itertools


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [list(itertools.combinations(combi, i)) for i in range(1, combi[-1] + 1)]  # Get all possible wel combinations
    cb = [item for sublist in cb for item in sublist][::-1]  # Flatten and reverse to get all combination at index 0.
    return cb
