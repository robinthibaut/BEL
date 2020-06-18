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
    def __init__(self, name, training, roots=None, directory=None):
        """

        :param name: name of the parameter on which to perform operations
        :param training: Training dataset

        """
        self.directory = directory
        self.name = name  # str, name of the object
        self.roots = roots  # Name of training roots

        self.shape = training.shape

        self.operator = None  # PCA operator
        self.ncomp = None  # Number of components

        # Training set - physical space
        self.training_physical = np.array([item for sublist in training for item in sublist]).reshape(len(training), -1)
        self.n_training = len(self.training_physical)  # Number of training data
        self.training_pc = None  # Training set - PCA space

        self.predict_physical = None  # Prediction set - physical space
        self.predict_pc = None  # Prediction set - PCA space

    # def pca_tp(self, n_training):
    #     """
    #     Given an arbitrary size of training data, splits the original array accordingly.
    #     The data used for prediction are after the training slice, i.e. the n_training last elements of the raw data
    #     :param n_training:
    #     :return: training, test
    #     """
    #     self.n_training = n_training
    #     # Flattens the array
    #     d_original = np.array([item for sublist in self.raw_data for item in sublist]).reshape(len(self.raw_data), -1)
    #     # Splits into training and test according to chosen n_training.
    #     d_t = d_original[:self.n_training]
    #     self.training_physical = d_t
    #     d_p = d_original[self.n_training:]
    #     self.predict_physical = d_p
    #
    #     return d_t, d_p

    def pca_training_transformation(self):
        """
        Instantiate the PCA object and transforms both training and test data.
        Depending on the value of the load parameter, it will create a new one or load a previously computed one.
        :return: PC training, PC test
        """

        pca_operator = PCA()
        self.operator = pca_operator
        self.operator.fit(self.training_physical)  # Principal components

        # Transform training data into principal components
        pc_training = self.operator.transform(self.training_physical)
        self.training_pc = pc_training

        return pc_training

    def pca_test_transformation(self, test):
        """
        Instantiate the PCA object and transforms both training and test data.
        Depending on the value of the load parameter, it will create a new one or load a previously computed one.
        :return: PC training, PC test
        """
        self.predict_physical = np.array([item for sublist in test for item in sublist]).reshape(len(test), -1)
        # Transform prediction data into principal components

        pc_prediction = self.operator.transform(self.predict_physical)
        self.predict_pc = pc_prediction

        return pc_prediction

    def n_pca_components(self, perc):
        """
        Given an explained variance percentage, returns the number of components
        necessary to obtain that level.
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)
        nc = len(np.where(evr <= perc)[0])

        self.ncomp = nc

        return nc

    def perc_pca_components(self, n_c):
        """
        Returns the explained variance percentage given a number of components n_c.
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)

        return evr[n_c - 1]

    def pca_refresh(self, n_comp=None):
        """
        Given a number of components to keep, returns the PC array with the corresponding shape.
        :param n_comp:
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

    def pc_random(self, n_posts):
        """
        Randomly selects PC components from the original training matrix dtp
        """
        r_rows = np.random.choice(self.training_pc.shape[0], n_posts)  # Selects n_posts rows from the training array
        score_selection = self.training_pc[r_rows, self.ncomp:]  # Extracts those rows, from the number of components
        # used until the end of the array.

        # For each column of shape n_sim-ncomp, selects a random PC component to add.
        test = [np.random.choice(score_selection[:, i]) for i in range(score_selection.shape[1])]

        return np.array(test)

    def inverse_transform(self, pc_to_invert):
        """
        Inverse transform PC based on the desired number of PC (stored in the shape of the argument).
        The self.operator.components contains all components.
        :param pc_to_invert: PC array
        :return: Back transformed array
        """
        # TODO: double check
        inv = np.dot(pc_to_invert, self.operator.components_[:pc_to_invert.shape[1], :]) + self.operator.mean_

        return inv

    def reset_(self):
        self.predict_pc = None
        self.predict_physical = None
        joblib.dump(self, os.path.join(self.directory, f'{self.name}_pca.pkl'))
        print(f'Target properties reset to {self.predict_pc}')
