import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Lambda, Concatenate
from keras.models import Model

def total_masked_logdensity(input):
    component_logdensity, mask = input
    D = K.expand_dims(K.constant(K.int_shape(mask)[1]),0)
    total = K.sum(K.abs(mask-1) * component_logdensity, axis=1) * D / (D - K.sum(mask, axis=1))
    return total


def create_input_layers(input_size):
    input_layer = Input(shape=(input_size,))
    mask_layer = Input(shape=(input_size,))
    masked_input_layer = Input(shape=(input_size*2,))
    #.concatenate([x[0] * x[1], K.repeat_elements(K.expand_dims(x[1], 0), x[0].shape[0], 0)]))([input, mask])
    return masked_input_layer, input_layer, mask_layer


class NadeMaskLayer(Layer):

    def __init__(self, **kwargs):
        seed = kwargs.pop("seed")
        super(NadeMaskLayer, self).__init__(**kwargs)
        self._deterministic = seed is not None
        if self._deterministic:
            self.seed = seed
        else:
            self.seed = np.random.random_integers(0, 10e6)


    def build(self, input_shape):
        super(NadeMaskLayer, self).build(input_shape)
        self._shape = input_shape

    def call(self, x):
        from theano import tensor as T
        from theano.tensor.shared_randomstreams import RandomStreams
        if K.backend() == "theano":
            import theano
            mask_rng = RandomStreams(self.seed)

            ints = mask_rng.random_integers(size=K.expand_dims(x.shape[0], 0), high=x.shape[1] - 1)

            def set_value_at_position(i, x):
                zeros = T.zeros_like(x[0, :])
                return T.set_subtensor(zeros[:i], 1)

            result, updates = theano.scan(fn=set_value_at_position,
                                          outputs_info=None,
                                          sequences=ints,
                                          non_sequences=x)
            mask = mask_rng.shuffle_row_elements(result)
        elif K.backend() == "tensorflow":
            import tensorflow as tf
            tf.set_random_seed(self.seed)
            ints = tf.random_uniform(shape=K.expand_dims(tf.shape(x)[0],0) ,
                                     maxval=x.shape[1],
                                     dtype=tf.int32)
            result = tf.sequence_mask(ints, maxlen=x.shape[1])
            parallel_iterations = self._deterministic and 1 or 10
            mask = tf.cast(tf.map_fn(tf.random_shuffle, result, parallel_iterations=parallel_iterations), K.floatx())
        else:
            raise NotImplementedError()
        return K.concatenate([x*mask, mask])

    def compute_output_shape(self, input_shape):
        return input_shape[0], 2 * input_shape[1]


def logdensity_model(inner_model):
    # TODO Orderings still missing
    input_size = inner_model.input_shape[1][1]
    # This returns a tensor
    inputs = Input(shape=(input_size,))
    batch_size = K.shape(inputs)[0]
    mask = np.zeros((1,input_size))
    outs = []
    for i in range(input_size):
        bn_mask = K.repeat(K.constant(mask), batch_size)[0]
        masked_input = Lambda(lambda x: K.concatenate([x * bn_mask, bn_mask]))(inputs)
        result = Lambda(lambda x: K.expand_dims(x[i]), output_shape=(1,))(inner_model([masked_input, inputs, Lambda(lambda x: bn_mask, name="mask_"+str(i))(inputs)]))
        outs.append(result)
        mask[0, i] = 1
    if len(outs) == 1:
        intermediate = outs[0]
    else:
        intermediate = Concatenate(axis=0)(outs)
    outputs = Lambda(lambda x: K.logsumexp(x), output_shape=(1,))(intermediate)
    return Model(inputs=inputs, outputs=outputs)


def training_model(inner_model, mask_seed=None):
    input_size = inner_model.input_shape[0][1] // 2
    inputs = Input(shape=(input_size,))
    mask_layer = NadeMaskLayer(seed=mask_seed, name="maskedinput")(inputs)
    masked_input = Lambda(lambda x: x[:,:input_size],name="input", output_shape=(input_size, ))(mask_layer)
    mask = Lambda(lambda x:  x[:, input_size:], name="mask", output_shape=(input_size, ))(mask_layer)
    inner_output = inner_model([mask_layer, inputs, mask])
    output = Lambda(total_masked_logdensity, output_shape=(1,))([inner_output, mask])
    return Model(inputs=inputs, outputs=output)

