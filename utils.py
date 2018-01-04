import keras
import numpy as np
from keras import backend as K
from keras.layers import Reshape, Dense, Lambda
from keras.regularizers import Regularizer


def _merge_mog(inputs):
    inputs, mu, sigma, alpha = inputs
    intermediate = -0.5 * K.square((mu - K.expand_dims(inputs, 2)) / sigma)
    return logsumexp(intermediate - K.log(sigma) - (0.5 * np.log(2 * np.pi)) + K.log(alpha), axis=2)


def mog_layer(input_layer, previous_layer, n_mixtures, regularize_activations=False):
    if regularize_activations:
        activity_regularizer = ExpConstraint(-2, 10)
    else:
        activity_regularizer = None
    n_outputs = K.int_shape(input_layer)[1]
    mu = Reshape((n_outputs, n_mixtures))(
        Dense(n_outputs * n_mixtures, name="mog_mu")(
            previous_layer))
    sigma = Reshape((n_outputs, n_mixtures))(
        Lambda(lambda x: K.exp(x), output_shape=(n_outputs * n_mixtures,))(
            Dense(n_outputs * n_mixtures, name="mog_sigma",
                  activity_regularizer=activity_regularizer,
                  bias_initializer=keras.initializers.Ones())(
                previous_layer)))
    # Implement softmax here as it has to work on n_mixtures
    temp_alpha = Lambda(lambda x: K.exp(x - K.max(x, axis=2, keepdims=True)), output_shape=(n_outputs, n_mixtures))(
        Reshape((n_outputs, n_mixtures))(Dense(n_outputs * n_mixtures, name="mog_alpha",
                                               activity_regularizer=activity_regularizer,
                                               bias_initializer=keras.initializers.Zeros())(
            previous_layer)))
    alpha = Lambda(lambda x: x / K.expand_dims(K.sum(x, axis=2), 2), output_shape=(n_outputs, n_mixtures))(temp_alpha)
    output = Lambda(_merge_mog, output_shape=(n_outputs,))([input_layer, mu, sigma, alpha])
    return output


def maximize_prediction(_y_true, y_pred):
    return -y_pred


def logsumexp(x, axis=None):
    max_x = K.max(x, axis=axis)
    return max_x + K.log(K.sum(K.exp(x - K.expand_dims(max_x, axis=axis)), axis=axis))


class ExpConstraint(Regularizer):
    def __init__(self, min, max):
        self.min = K.cast_to_floatx(min)
        self.max = K.cast_to_floatx(max)

    def __call__(self, x):
        return K.sum(K.exp(1 * (K.relu(self.min - x) + K.relu(x - self.max)))) - K.prod(K.cast(K.shape(x), K.floatx()))

    def get_config(self):
        return {'min': float(self.min), 'max': float(self.max)}
