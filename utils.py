import keras
import numpy as np
from keras import backend as K
from keras.layers import Reshape, Dense, Lambda


def _merge_mog(inputs):
    inputs, mu, sigma, alpha = inputs
    intermediate = -0.5 * K.square((mu - K.expand_dims(inputs, 2)) / sigma)
    return K.logsumexp(intermediate - K.log(sigma) - (0.5 * np.log(2 * np.pi)) + K.log(alpha), axis=2)


def mog_layer(input_layer, previous_layer, n_mixtures):
    n_outputs = K.int_shape(input_layer)[1]
    mu = Reshape((n_outputs, n_mixtures))(
        Dense(n_outputs*n_mixtures, name="mog_mu")(
            previous_layer))
    sigma = Reshape((n_outputs, n_mixtures))(
        Lambda(lambda x: K.exp(x))(
            Dense(n_outputs*n_mixtures, name="mog_sigma",
                  bias_initializer=keras.initializers.Ones())(
                previous_layer)))
    temp_alpha = Reshape((n_outputs, n_mixtures))(
        Lambda(K.exp)(
            Dense(n_outputs*n_mixtures, name="mog_alpha",
                  bias_initializer=keras.initializers.Zeros())(
                previous_layer)))
    alpha = Lambda(lambda x:  x / K.expand_dims(K.sum(x, axis=2), 2))(temp_alpha)
    output = Lambda(_merge_mog, output_shape=(n_outputs,))([input_layer, mu, sigma, alpha])
    return output


def maximize_prediction(_y_true, y_pred):
    return -y_pred
