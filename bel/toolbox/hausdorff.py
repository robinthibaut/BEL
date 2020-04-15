import numpy as np
from scipy.spatial.distance import cdist


def modified_distance(a, b):
    d = cdist(a, b)
    fhd = np.mean(np.min(d, axis=0))
    rhd = np.mean(np.min(d, axis=1))

    return max(fhd, rhd)
