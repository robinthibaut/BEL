#  Copyright (c) 2021. Robin Thibaut, Ghent University
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dcca.linear_cca import linear_cca
from dcca.models import create_model
from dcca.objectives import cca_loss
from inn.flow import MSE, MMD_multiscale, tfk
from inn.utils import NBatchLogger, UpdateLossFactor
from keras.callbacks import ModelCheckpoint
from skbel.learning.bel import BEL
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from bel4ed import Setup
from bel4ed.datasets import i_am_root, load_dataset


def kernel_bel():
    """
    Set all BEL pipelines. This is the blueprint of the framework.
    """
    n_pc_pred, n_pc_targ = 200, 200
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                KernelPCA(
                    n_components=n_pc_pred,
                    kernel="rbf",
                    fit_inverse_transform=False,
                    alpha=1e-5,
                ),
            ),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                KernelPCA(
                    n_components=n_pc_targ,
                    kernel="rbf",
                    fit_inverse_transform=True,
                    alpha=1e-5,
                ),
            ),
        ]
    )

    # Initiate BEL object
    bel_model = BEL(
        X_pre_processing=X_pre_processing,
        Y_pre_processing=Y_pre_processing,
    )

    return bel_model


def init_bel():
    """
    Set all BEL pipelines
    :return:
    """
    n_pc_pred, n_pc_targ = (
        Setup.HyperParameters.n_pc_predictor,
        Setup.HyperParameters.n_pc_target,
    )
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_pc_pred)),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_pc_targ)),
        ]
    )

    # Canonical Correlation Analysis
    # Number of CCA components is chosen as the min number of PC

    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 5)

    # Pipeline after CCA
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )
    Y_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Initiate BEL object
    bel_model = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        cca=cca,
    )

    # Set PC cut
    bel_model.X_n_pc = n_pc_pred
    bel_model.Y_n_pc = n_pc_targ

    return bel_model


def train_model(
    model,
    train_set_x1,
    train_set_x2,
    test_set_x1,
    test_set_x2,
    valid_set_x1,
    valid_set_x2,
    epoch_num,
    batch_size,
):
    """
    trains the model
    # Arguments
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively. data should be packed
        like ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        epoch_num: number of epochs to train the model
        batch_size: the size of batches
    # Returns
        the trained model
    """

    # best weights are saved in "temp_weights.hdf5" during training
    # it is done to return the best model based on the validation loss
    checkpointer = ModelCheckpoint(
        filepath="temp_weights.h5",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    # used dummy Y because labels are not used in the loss function
    model.fit(
        [train_set_x1, train_set_x2],
        np.zeros(len(train_set_x1)),
        batch_size=batch_size,
        epochs=epoch_num,
        shuffle=True,
        validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))),
        callbacks=[checkpointer],
    )

    model.load_weights("temp_weights.h5")

    results = model.evaluate(
        [test_set_x1, test_set_x2],
        np.zeros(len(test_set_x1)),
        batch_size=batch_size,
        verbose=1,
    )

    print("loss on test data: ", results)

    results = model.evaluate(
        [valid_set_x1, valid_set_x2],
        np.zeros(len(valid_set_x1)),
        batch_size=batch_size,
        verbose=1,
    )
    print("loss on validation data: ", results)
    return model


def test_model(model, data1, outdim_size, apply_linear_cca):
    """produce the new features by using the trained model
    # Arguments
        model: the trained model
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively.
            Data should be packed like
            ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        outdim_size: dimension of new features
        apply_linear_cca: if to apply linear CCA on the new features
    # Returns
        new features packed like
            ((new X for train - view 1, new X for train - view 2, Y for train),
            (new X for validation - view 1, new X for validation - view 2, Y for validation),
            (new X for test - view 1, new X for test - view 2, Y for test))
    """

    # producing the new features
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k][0], data1[k][1]])
        r = int(pred_out.shape[1] / 2)
        new_data.append([pred_out[:, :r], pred_out[:, r:], data1[k][1]])

    # based on the DCCA paper, a linear CCA should be applied on the output of the networks because
    # the loss function actually estimates the correlation when a linear CCA is applied to the output of the networks
    # however it does not improve the performance significantly
    if apply_linear_cca:
        w = [None, None]
        m = [None, None]
        print("Linear CCA started!")
        w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], outdim_size)
        print("Linear CCA ended!")

        # Something done in the original MATLAB implementation of DCCA, do not know exactly why;)
        # it did not affect the performance significantly on the noisy MNIST dataset
        # s = np.sign(w[0][0,:])
        # s = s.reshape([1, -1]).repeat(w[0].shape[0], axis=0)
        # w[0] = w[0] * s
        # w[1] = w[1] * s
        ###

        for k in range(3):
            data_num = len(new_data[k][0])
            for v in range(2):
                new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
                new_data[k][v] = np.dot(new_data[k][v], w[v])

    return new_data


def inner_cca_objective(y_true, y_pred):
    """
    It is the loss function of CCA as introduced in the original paper.
    """

    r1 = 1e-4
    r2 = 1e-4
    eps = 1e-12
    o1 = o2 = int(y_pred.shape[1] // 2)

    # unpack (separate) the output of networks for view 1 and view 2
    H1 = tf.transpose(a=y_pred[:, 0:o1])
    H2 = tf.transpose(a=y_pred[:, o1 : o1 + o2])

    m = tf.shape(input=H1)[1]

    H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
    H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

    SigmaHat12 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
        H1bar, H2bar, transpose_b=True
    )  # [dim, dim]
    SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
        H1bar, H1bar, transpose_b=True
    ) + r1 * tf.eye(o1)
    SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
        H2bar, H2bar, transpose_b=True
    ) + r2 * tf.eye(o2)

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = tf.linalg.eigh(SigmaHat11)
    [D2, V2] = tf.linalg.eigh(SigmaHat22)  # Added to increase stability

    posInd1 = tf.compat.v1.where(tf.greater(D1, eps))
    D1 = tf.gather_nd(D1, posInd1)  # get eigen values that are larger than eps
    V1 = tf.transpose(
        a=tf.nn.embedding_lookup(params=tf.transpose(a=V1), ids=tf.squeeze(posInd1))
    )

    posInd2 = tf.compat.v1.where(tf.greater(D2, eps))
    D2 = tf.gather_nd(D2, posInd2)
    V2 = tf.transpose(
        a=tf.nn.embedding_lookup(params=tf.transpose(a=V2), ids=tf.squeeze(posInd2))
    )

    SigmaHat11RootInv = tf.matmul(
        tf.matmul(V1, tf.linalg.tensor_diag(D1 ** -0.5)), V1, transpose_b=True
    )  # [dim, dim]
    SigmaHat22RootInv = tf.matmul(
        tf.matmul(V2, tf.linalg.tensor_diag(D2 ** -0.5)), V2, transpose_b=True
    )

    Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    if use_all_singular_values:
        corr = tf.sqrt(tf.linalg.trace(tf.matmul(Tval, Tval, transpose_a=True)))
    else:
        [U, V] = tf.linalg.eigh(tf.matmul(Tval, Tval, transpose_a=True))
        U = tf.gather_nd(U, tf.compat.v1.where(tf.greater(U, eps)))
        kk = tf.reshape(tf.cast(tf.shape(input=U), tf.int32), [])
        K = tf.minimum(kk, outdim_size)
        w, _ = tf.nn.top_k(U, k=K)
        corr = tf.reduce_sum(input_tensor=tf.sqrt(w))

    return -corr


class Trainer(tfk.Model, ABC):
    def __init__(
        self,
        model,
    ):
        super(Trainer, self).__init__()
        self.model = model

        self.w1 = 5.0
        self.w2 = 1.0
        self.w3 = 10.0
        self.loss_factor = 1.0
        self.loss_fit = inner_cca_objective
        self.loss_latent = MMD_multiscale
        self.loss_backward = MMD_multiscale

    def train_step(self, data):
        x_data, y_data = data
        # x = x_data[:, :self.x_dim]
        # y = y_data[:, -self.y_dim:]
        # z = y_data[:, : self.z_dim]
        # y_short = tf.concat([z, y], axis=-1)

        # Forward loss
        with tf.GradientTape() as tape:
            y_out = self.model(x_data)
            pred_loss = self.w1 * self.loss_fit(
                y_data, y_out
            )  # [zeros, y] <=> [zeros, yhat]
            output_block_grad = tf.concat(
                [y_out, y_out], axis=-1
            )  # take out [z, y] only (not zeros)
            latent_loss = self.w2 * self.loss_latent(
                y_out, output_block_grad
            )  # [z, y] <=> [zhat, yhat]
            forward_loss = pred_loss + latent_loss
        grads_forward = tape.gradient(forward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_forward, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            x_rev = self.model.inverse(y_data)
            # rev_loss = self.w3 * self.loss_factor * self.loss_fit(x_rev, x_data)
            rev_loss = self.w3 * self.loss_factor * self.loss_backward(x_rev, x_data)
        grads_backward = tape.gradient(rev_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads_backward, self.model.trainable_weights)
        )

        total_loss = forward_loss + latent_loss + rev_loss
        return {
            "total_loss": total_loss,
            "forward_loss": forward_loss,
            "latent_loss": latent_loss,
            "rev_loss": rev_loss,
        }

    def test_step(self, data):
        x_data, y_data = data
        return NotImplementedError


############
# Parameters Section

# the path to save the final learned features
save_to = "./new_features.gz"

# the size of the new space learned by the model (number of the new features)
outdim_size = 30

# size of the input for view 1 and view 2
input_shape1 = 200
input_shape2 = 200

# number of layers with nodes in each one

size = 4
layer_sizes1 = [size * 2, size, size // 2, outdim_size]
layer_sizes2 = [size * 2, size, size // 2, outdim_size]

# the parameters for training the network
learning_rate = 1e-3  # 1e-1
epoch_num = 100
batch_size = 24

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-5  # 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False  # False

# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
apply_linear_cca = False

# end of parameters section
############

# Building, training, and producing the new features by DCCA


model.build(input_shape=(200,))
model.summary()

# Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
# Datasets get stored under the datasets folder of user's Keras folder
# normally under [Home Folder]/.keras/datasets/
X, Y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    train_size=3000,
    test_size=1000,
)

trainer = Trainer(model, 1, 1, 4)
trainer.compile(optimizer="adam")


X_valid, y_valid = X_test[:500], y_test[:500]
X_test, y_test = X_test[500:], y_test[500:]

bel = kernel_bel()
bel.fit(X_train, y_train)

X_train = bel.X_pre_processing.transform(X_train)
X_test = bel.X_pre_processing.transform(X_test)
X_valid = bel.X_pre_processing.transform(X_valid)

y_train = bel.Y_pre_processing.transform(y_train)
y_test = bel.Y_pre_processing.transform(y_test)
y_valid = bel.Y_pre_processing.transform(y_valid)


# Train the model

dataset = (X_train, y_train)
LossFactor = UpdateLossFactor(100)
logger = NBatchLogger(100, 100)
hist = trainer.fit(
    dataset,
    batch_size=32,
    epochs=100,
    callbacks=[logger, LossFactor],
    verbose=0,
)


model = train_model(
    model, X_train, y_train, X_test, y_test, X_valid, y_valid, epoch_num, batch_size
)

output = model.predict([X_test, y_test])

data1 = [[X_train, y_train], [X_valid, y_valid], [X_test, y_test]]
new_data = test_model(model, data1, outdim_size, apply_linear_cca)

for i in range(outdim_size):
    plt.plot(new_data[1][0][:, i], new_data[1][1][:, i], "ro")
    plt.show()
