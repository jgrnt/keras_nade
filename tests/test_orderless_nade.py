import numpy as np
from keras import optimizers
from keras.engine.topology import Container
from keras.layers import Dense, Lambda, K, Add, Activation, add

import utils

from orderless_nade import logdensity_model, training_model, create_input_layers
from utils import createMoGLayer


def test_compare_original_nade():
    import h5py
    masked_input_layer, input_layer, mask_layer = create_input_layers(2)

    mog = Container(inputs=[masked_input_layer, input_layer, mask_layer], outputs=createMoGLayer(input_layer, Activation("relu")(add([Dense(16)(Lambda(lambda x: x[:,:2])(masked_input_layer)),Dense(16, use_bias=False)(mask_layer)])),1))
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer], outputs=mog([masked_input_layer, input_layer, mask_layer]))

    model = training_model(inner_model)
    model.compile(loss=utils.maximize_prediction, optimizer="sgd")
    with h5py.File("tests/original_nade_weights.hdf5") as h:
        model.weights[0].set_value(h["final_model/parameters/W1"][()].astype(np.float32))
        model.weights[1].set_value(h["final_model/parameters/b1"][()].astype(np.float32))
        model.weights[2].set_value(h["final_model/parameters/Wflags"][()].astype(np.float32))
        model.weights[3].set_value(h["final_model/parameters/V_alpha"][()].T.reshape((16, 2)).astype(np.float32))
        model.weights[4].set_value(h["final_model/parameters/b_alpha"][()].reshape((2)).astype(np.float32))
        model.weights[5].set_value(h["final_model/parameters/V_sigma"][()].T.reshape((16, 2)).astype(np.float32))
        model.weights[6].set_value(h["final_model/parameters/b_sigma"][()].reshape((2)).astype(np.float32))
        model.weights[7].set_value(h["final_model/parameters/V_mu"][()].T.reshape((16, 2)).astype(np.float32))
        model.weights[8].set_value(h["final_model/parameters/b_mu"][()].reshape((2)).astype(np.float32))

    np.random.seed(1)
    assert np.allclose(np.array([-3.33089394, -2.55555928, -4.85813281, -4.85442475, -1.92244674]),
                       model.predict(np.random.normal(size=(5,2))))


def test_train_logdensity():
    masked_input_layer, input_layer, mask_layer = create_input_layers(2)
    mog = Container(inputs=[masked_input_layer, input_layer, mask_layer], outputs=createMoGLayer(input_layer, Activation("relu")(Add()([Dense(16)(Lambda(lambda x: x[:,:2], output_shape=(2,))(masked_input_layer)),Dense(16, use_bias=False)(mask_layer)])),1))
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer], outputs=mog([masked_input_layer, input_layer, mask_layer]))

    model = training_model(inner_model)
    model.compile(loss=utils.maximize_prediction, optimizer=optimizers.SGD(lr=0.1),)
    fit_met = model.fit(np.random.normal(0, size=(200, 2)),
              np.zeros(shape=(200, 2)),
              batch_size=5,
              verbose=0,
              epochs=2)
    test_model = logdensity_model(inner_model)
    test_model.compile(optimizer="sgd", loss=utils.maximize_prediction)
    real=test_model.evaluate(np.random.normal(0, size=(50, 2)), np.zeros(shape=(50, 2)), verbose=0)
    wrong=test_model.evaluate(np.random.normal(1, size=(50, 2)), np.zeros(shape=(50, 2)), verbose=0)
    assert real < wrong



