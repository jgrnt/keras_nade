import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Lambda, Concatenate
from keras.models import Model

def total_masked_logdensity(input):
    component_logdensity, mask = input
    D = mask.shape[1]
    total = K.sum(K.switch(mask, K.zeros_like(component_logdensity), component_logdensity), axis=1) * D / (D - mask.sum(axis=1))
    return total


def create_input_layers(input_size):
    input_layer = Input(shape=(input_size,))
    mask_layer = Input(shape=(input_size,))
    masked_input_layer = Input(shape=(input_size*2,))
    #.concatenate([x[0] * x[1], K.repeat_elements(K.expand_dims(x[1], 0), x[0].shape[0], 0)]))([input, mask])
    return masked_input_layer, input_layer, mask_layer


class NadeMaskLayer(Layer):

    def __init__(self, mask_size, **kwargs):
        super(NadeMaskLayer, self).__init__(**kwargs)
        self.mask_size = mask_size

    def build(self, input_shape):
        super(NadeMaskLayer, self).build(input_shape)
        self._shape = input_shape

    def call(self, x):
        from theano import tensor as T
        from theano.tensor.shared_randomstreams import RandomStreams
        import theano
        mask_rng = RandomStreams(1)

        ints = mask_rng.random_integers(size=[5], high=self.mask_size - 1)
        mask = T.zeros(([self.mask_size]))

        def set_value_at_position(i, mask):
            zeros = T.zeros_like(mask)
            return T.set_subtensor(zeros[:i], 1)

        result, updates = theano.scan(fn=set_value_at_position,
                                      outputs_info=None,
                                      sequences=ints,
                                      non_sequences=mask)
        mask = mask_rng.shuffle_row_elements(result)
        return K.concatenate([x*mask, mask])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.mask_size)

def logdensity_model(inner_model):
    input_size = inner_model.input_shape[1][1]
    # This returns a tensor
    inputs = Input(shape=(input_size,))
    batch_size = inputs.shape[0]
    mask = np.zeros(input_size)
    outs = []
    for i in range(input_size):
        bn_mask = K.repeat_elements(K.expand_dims(K.constant(mask), 0), batch_size, 0)
        masked_input = Lambda(lambda x: K.concatenate([x * mask.copy(), bn_mask]))(inputs)
        result = Lambda(lambda x: K.expand_dims(x[i]), output_shape=(1,))(inner_model([masked_input, inputs, Lambda(lambda x: bn_mask, name="mask_"+str(i))(inputs)]))
        outs.append(result)
        mask[i] = 1
    if len(outs) == 1:
        intermediate = outs[0]
    else:
        intermediate = Concatenate(axis=0)(outs)
    outputs = Lambda(lambda x: K.logsumexp(x))(intermediate)
    return Model(inputs=inputs, outputs=outputs)


def training_model(inner_model):
    input_size = inner_model.input_shape[0][1] // 2
    inputs = Input(shape=(input_size,))
    mask_layer = NadeMaskLayer(input_size,  name="maskedinput")(inputs)
    masked_input = Lambda(lambda x: x[:,:input_size],name="input", output_shape=(input_size, ))(mask_layer)
    mask = Lambda(lambda x:  x[:, input_size:], name="mask", output_shape=(input_size, ))(mask_layer)
    inner_output = inner_model([mask_layer, inputs, mask])
    output = Lambda(total_masked_logdensity)([inner_output, mask])
    return Model(inputs=inputs, outputs=output)

