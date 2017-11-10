import keras
import numpy as np
from keras import backend as K
from keras.layers import Reshape, Dense, Lambda


def _mergeMoG(inputs):
    inputs, mu, sigma, alpha = inputs

    int = -0.5 * K.square((mu - K.expand_dims(inputs,2)) / sigma)
    return K.logsumexp(int - K.log(sigma) - (0.5 * np.log(2 * np.pi)) + K.log(alpha), axis=2)


def createMoGLayer(input_layer, previous_layer, n_mixtures):
    n_outputs = input_layer._keras_shape[1]
    mu = Reshape((n_outputs, n_mixtures))(Dense(n_outputs*n_mixtures, name="mog_mu")(previous_layer))
    sigma = Reshape((n_outputs, n_mixtures))(Lambda(lambda x: K.exp(x))(Dense(n_outputs*n_mixtures, name="mog_sigma", bias_initializer=keras.initializers.Ones())(previous_layer)))
    temp_alpha = Reshape((n_outputs, n_mixtures))(Lambda(K.exp)(Dense(n_outputs*n_mixtures, name="mog_alpha", bias_initializer=keras.initializers.Zeros())(previous_layer)))
    alpha = Lambda(lambda x:  x / K.expand_dims(K.sum(x, axis=2), 2))(temp_alpha)
    output = Lambda(_mergeMoG, output_shape=(n_outputs,))([input_layer, mu, sigma, alpha])
    return output


def maximize_prediction(y_true, y_pred):
    return - y_pred