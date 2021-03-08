#  Copyright (c) 2021. Robin Thibaut, Ghent University

"""
Collection of functions to use the PCA class from scikit-learn.
"""

import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loguru import logger

__all__ = ['PC']


class PC:
    def __init__(self,
                 name: str,
                 training: np.array = None,
                 roots: list = None,
                 directory: str = None):
        """
        Given a set of training data and one observation (optional), performs necessary dimension reduction
        and transformations.
        :param name: str: name of the dataset (e.g. 'data', 'target'...)
        :param training: numpy.ndarray: Training dataset
        :param roots: list: List containing uuid of training roots
        :param directory: str: Path to the folder in which to save the pickle
        """

        self.directory = directory
        self.name = name  # str, name of the object
        self.roots = roots  # Name of training roots

        self.training_shape = training.shape  # Original shape of dataset
        self.n_samples = self.training_shape[0]
        self.n_components = self.training_shape[1]

        self.obs_shape = None  # Original shape of observation

        # Divide the sample by their standard deviation.
        self.scaler = StandardScaler(with_mean=False)
        # The samples are automatically scaled by scikit-learn PCA()
        self.operator = PCA()  # PCA operator (scikit-learn instance)
        # self.transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.pipe = make_pipeline(self.scaler, self.operator, verbose=False)

        self.n_pc_cut = None  # Number of components to keep

        # Training set - physical space - flattened array
        self.training_physical = np.array(
            [item for sublist in training for item in sublist]).reshape(len(training), -1)
        # Number of training samples
        self.n_training = len(self.training_physical)
        self.training_pc = None  # Training PCA scores

        self.test_root = None  # List containing uuid of observation
        self.predict_physical = None  # Observation set - physical space
        self.predict_pc = None  # Observation PCA scores

    def training_fit_transform(self):
        """
        Instantiate the PCA object and transforms training data to scores.
        :return: numpy.ndarray: PC training
        """

        self.pipe.fit(self.training_physical)
        self.training_pc = self.pipe.transform(self.training_physical)

        return self.training_pc

    def test_transform(self,
                       test: np.array,
                       test_root: list):
        """
        Transforms observation to PC scores.
        :param test: numpy.ndarray: Observation array
        :param test_root: list: List containing observation id (str)
        :return: numpy.ndarray: Observation PC
        """
        self.test_root = test_root

        self.obs_shape = test.shape

        # Flattened array
        self.predict_physical = np.array(
            [item for sublist in test for item in sublist]).reshape(len(test), -1)

        # Transform prediction data into principal components
        pc_prediction = self.pipe.transform(self.predict_physical)
        self.predict_pc = pc_prediction

        return pc_prediction

    def perc_2_comp(self, perc: float):
        """
        Given an explained variance percentage, returns the number of components
        necessary to obtain that level.
        :param perc: float: Percentage between 0 and 1
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)
        self.n_pc_cut = len(np.where(evr <= perc)[0])

        return self.n_pc_cut

    def perc_comp(self, n_c: int):
        """
        Returns the explained variance percentage given a number of components n_c.
        :param n_c: int: Number of components to keep
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)

        return evr[n_c - 1]

    def comp_refresh(self, n_comp: int = None):
        """
        Given a number of components to keep, returns the PC array with the corresponding shape.
        :param n_comp: int: Number of components
        :return:
        """

        if n_comp is not None:
            self.n_pc_cut = n_comp  # Assign the number of components in the class for later use

        # Reloads the original training components
        pc_training = self.training_pc.copy()
        pc_training = pc_training[:, :self.n_pc_cut]  # Cut

        if self.predict_pc is not None:
            pc_prediction = self.predict_pc.copy()  # Reloads the original test components
            pc_prediction = pc_prediction[:, :self.n_pc_cut]  # Cut
            return pc_training, pc_prediction

        else:
            return pc_training

    def random_pc(self, n_rand: int):
        """
        Randomly selects PC components from the original training matrix.
        :param n_rand: int: Number of random PC to use
        :return numpy.ndarray: Random PC scores
        """
        rand_rows = np.random.choice(
            self.n_samples, n_rand)  # Selects n_posts rows from the training array
        # Extracts those rows, from the number of
        score_selection = self.training_pc[rand_rows, self.n_pc_cut:]
        # components used until the end of the array.

        # For each column of shape n_samples, n_components, selects a random PC component to add.
        test = [np.random.choice(score_selection[:, i])
                for i in range(score_selection.shape[1])]

        return np.array(test)

    def custom_inverse_transform(self,
                                 pc_to_invert,
                                 n_comp: int = None):
        """
        Inverse transform PC based on the desired number of PC (stored in the shape of the argument).
        The self.operator.components contains all components.
        :param pc_to_invert: np.array: (n_samples, n_components) PC array to back-transform to physical space
        :param n_comp: int: Number of components to back-transform with
        :return: numpy.ndarray: Back transformed array
        """
        if n_comp is None:
            n_comp = self.n_pc_cut

        # TODO: (optimization) only fit after dimension check
        op_cut = make_pipeline(self.scaler, PCA(n_components=n_comp))
        op_cut.fit(self.training_physical)

        inv = op_cut.inverse_transform(pc_to_invert[:, :n_comp])

        return inv

    def reset_(self):
        """
        Deletes (resets) the observation properties in the object.
        """
        self.predict_pc = None
        self.predict_physical = None
        # Re-dumps pca object
        joblib.dump(self, os.path.join(self.directory, f'{self.name}_pca.pkl'))
        logger.info(f'Target properties of {self.name} reset to {self.predict_pc}')
