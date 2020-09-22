#  Copyright (c) 2020. Robin Thibaut, Ghent University


"""
Collection of functions to use the PCA class from scikit-learn.
"""

import os
import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAIO:
    # TODO: add properties
    def __init__(self, name: str, training=None, roots: list = None, directory: str = None):
        """
        Given a set of training data and one observation (optional), performs necessary dimension reduction
        and transformations.
        :param name: str: name of the dataset (e.g. 'data', 'target'...)
        :param training: np.array: Training dataset
        :param roots: list: List containing uuid of training roots
        :param directory: str: Path to the folder in which to save the pickle
        """

        self.directory = directory
        self.name = name  # str, name of the object
        self.roots = roots  # Name of training roots

        self.training_shape = training.shape  # Original shape of dataset
        self.obs_shape = None  # Original shape of observation

        self.operator = None  # PCA operator (scikit-learn instance)
        self.ncomp = None  # Number of components to keep

        # Training set - physical space - flattened array
        self.training_physical = np.array([item for sublist in training for item in sublist]).reshape(len(training), -1)
        self.n_training = len(self.training_physical)  # Number of training samples
        self.training_pc = None  # Training PCA scores

        self.test_root = None  # List containing uuid of observation
        self.predict_physical = None  # Observation set - physical space
        self.predict_pc = None  # Observation PCA scores

    def pca_training_transformation(self):
        """
        Instantiate the PCA object and transforms training data to scores.
        :return: np.array: PC training
        """

        pca_operator = PCA()
        self.operator = pca_operator
        self.operator.fit(self.training_physical)  # Principal components

        # Transform training data into principal components
        self.training_pc = self.operator.transform(self.training_physical)

        return self.training_pc

    def pca_test_transformation(self, test, test_root: list):
        """
        Transforms observation to PC scores.
        :param test: np.array: Observation array
        :param test_root: list: List containing observation id (str)
        :return: np.array: Observation PC
        """
        self.test_root = test_root

        self.obs_shape = test.shape

        # Flattened array
        self.predict_physical = np.array([item for sublist in test for item in sublist]).reshape(len(test), -1)

        # Transform prediction data into principal components
        pc_prediction = self.operator.transform(self.predict_physical)
        self.predict_pc = pc_prediction

        return pc_prediction

    def n_pca_components(self, perc: float):
        """
        Given an explained variance percentage, returns the number of components
        necessary to obtain that level.
        :param perc: float: Percentage between 0 and 1
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)
        self.ncomp = len(np.where(evr <= perc)[0])

        return self.ncomp

    def perc_pca_components(self, n_c: int):
        """
        Returns the explained variance percentage given a number of components n_c.
        :param n_c: int: Number of components to keep
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)

        return evr[n_c - 1]

    def pca_refresh(self, n_comp: int = None):
        """
        Given a number of components to keep, returns the PC array with the corresponding shape.
        :param n_comp: int: Number of components
        :return:
        """

        if n_comp is not None:
            self.ncomp = n_comp  # Assign the number of components in the class for later use

        pc_training = self.training_pc.copy()  # Reloads the original training components
        pc_training = pc_training[:, :self.ncomp]  # Cut

        if self.predict_pc is not None:
            pc_prediction = self.predict_pc.copy()  # Reloads the original test components
            pc_prediction = pc_prediction[:, :self.ncomp]  # Cut
            return pc_training, pc_prediction

        else:
            return pc_training

    def pc_random(self, n_posts: int):
        """
        Randomly selects PC components from the original training matrix.
        :param n_posts: int: Number of random PC to use
        :return np.array: Random PC scores
        """
        r_rows = np.random.choice(self.training_pc.shape[0], n_posts)  # Selects n_posts rows from the training array
        score_selection = self.training_pc[r_rows, self.ncomp:]  # Extracts those rows, from the number of components
        # used until the end of the array.

        # For each column of shape n_sim-ncomp, selects a random PC component to add.
        test = [np.random.choice(score_selection[:, i]) for i in range(score_selection.shape[1])]

        return np.array(test)

    def inverse_transform(self, pc_to_invert, n_comp: int = None):
        """
        Inverse transform PC based on the desired number of PC (stored in the shape of the argument).
        The self.operator.components contains all components.
        :param pc_to_invert: np.array: PC array to back-transform to physical space
        :param n_comp: int: Number of components to back-transform with
        :return: Back transformed array
        """
        if n_comp is None:
            n_comp = self.ncomp
        # TODO: double check
        inv = np.dot(pc_to_invert, self.operator.components_[:n_comp, :]) + self.operator.mean_

        return inv

    def reset_(self):
        """
        Deletes (resets) the observation properties in the object.
        """
        self.predict_pc = None
        self.predict_physical = None
        # Re-dumps pca object
        joblib.dump(self, os.path.join(self.directory, f'{self.name}_pca.pkl'))
        print(f'Target properties reset to {self.predict_pc}')
