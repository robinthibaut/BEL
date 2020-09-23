#  Copyright (c) 2020. Robin Thibaut, Ghent University
import numpy as np
from sklearn.preprocessing import PowerTransformer


class TargetIO:

    """Perform data transformation on the target feature"""

    def __init__(self):
        self.gaussian_transformers = {}

    def gaussian_distribution(self,
                              original_array,
                              name: str = 'gd'):
        """

        :param original_array:
        :param name:
        :return:
        """
        # Ensure Gaussian distribution in original_array Each vector for each original_array components will be
        # transformed one-by-one by a different operator, stored in yj.
        yj = [PowerTransformer(method='yeo-johnson', standardize=True) for _ in range(original_array.shape[0])]
        self.gaussian_transformers[name] = yj  # Adds the gaussian distribution transformers object to the dictionary
        # Fit each PowerTransformer with each component
        [yj[i].fit(original_array[i].reshape(-1, 1)) for i in range(len(yj))]
        # Transform the original distribution.
        original_array_gaussian \
            = np.concatenate([yj[i].transform(original_array[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

        return original_array_gaussian

    def gaussian_inverse(self, original_array, name: str = 'gd'):
        yj = self.gaussian_transformers[name]
        back2 \
            = np.concatenate([yj[i].inverse_transform(original_array[i].reshape(-1, 1)) for i in range(len(yj))],
                             axis=1).T

        return back2
