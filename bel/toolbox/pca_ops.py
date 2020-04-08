from os.path import join as jp
import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAOps:

    def __init__(self, name, raw_data, directory='temp'):
        """

        @param name: name of the parameter on which to perform operations
        @param raw_data: original dataset
        """
        self.directory = directory
        self.name = name
        self.raw_data = raw_data  # raw data
        self.n_training = None  # Number of training data
        self.operator = None  # PCA operator
        self.ncomp = None  # Number of components
        self.d0 = None  # Original data
        self.dt = None  # Training set - physical space
        self.dp = None  # Prediction set - physical space
        self.dtp = None  # Training set - PCA space
        self.dpp = None  # Prediction set - PCA space

    def pca_tp(self, n_training):
        """
        Given an arbitrary size of training data, splits the original array accordingly
        @param n_training:
        @return: training, test
        """
        self.n_training = n_training
        # Flattens the array
        d_original = np.array([item for sublist in self.raw_data for item in sublist]).reshape(len(self.raw_data), -1)
        self.d0 = d_original
        # Splits into training and test according to chosen n_training.
        d_t = d_original[:self.n_training]
        self.dt = d_t
        d_p = d_original[self.n_training:]
        self.dp = d_p

        return d_t, d_p

    def pca_transformation(self, load=False):
        """
        Instantiate the PCA object and transforms both training and test data.
        Depending on the value of the load parameter, it will create a new one or load a previously computed one,
        stored in the 'temp' folder.
        @param load:
        @return: PC training, PC test
        """
        # TODO: Change here
        if not load:
            pca_operator = PCA()
            self.operator = pca_operator
            pca_operator.fit(self.dt)  # Principal components
            joblib.dump(pca_operator, jp(self.directory, '{}_pca_operator.pkl'.format(self.name)))
        else:
            pca_operator = joblib.load(jp(self.directory, '{}_pca_operator.pkl'.format(self.name)))
            self.operator = pca_operator

        pc_training = pca_operator.transform(self.dt)  # Principal components
        self.dtp = pc_training
        pc_prediction = pca_operator.transform(self.dp)
        self.dpp = pc_prediction

        return pc_training, pc_prediction

    def n_pca_components(self, perc):
        """
        Given an explained variance percentage, returns the number of components necessary to obtain that level.
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)
        nc = len(np.where(evr <= perc)[0])

        return nc

    def perc_pca_components(self, n_c):
        """
        Returns the explained variance percentage given a number of components n_c
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)

        return evr[n_c - 1]

    def pca_refresh(self, n_comp):
        """
        Given a number of components to keep, returns the PC array with the corresponding shape.
        @param n_comp:
        @return:
        """

        self.ncomp = n_comp  # Assign the number of components in the class for later use

        pc_training = self.dtp.copy()  # Reloads the original training components
        pc_training = pc_training[:, :n_comp]  # Cut

        pc_prediction = self.dpp.copy()  # Reloads the original test components
        pc_prediction = pc_prediction[:, :n_comp]  # Cut

        return pc_training, pc_prediction

    def pc_random(self, n_posts):
        """
        Randomly selects PC components from the original training matrix dtp
        """
        r_rows = np.random.choice(self.dtp.shape[0], n_posts)  # Selects n_posts rows from the dtp array
        score_selection = self.dtp[r_rows, self.ncomp:]  # Extracts those rows, from the number of components used until
        # the end of the array.

        # For each column of shape n_sim-ncomp, selects a random PC component to add.
        test = [np.random.choice(score_selection[:, i]) for i in range(score_selection.shape[1])]

        return np.array(test)

    def inverse_transform(self, pc_to_invert):
        """
        Inverse transform PC
        @param pc_to_invert: PC array
        @return:
        """
        inv = np.dot(pc_to_invert, self.operator.components_[:pc_to_invert.shape[1], :]) + self.operator.mean_

        return inv