import numpy as np
from keras import optimizers
from keras.engine.topology import Container
from keras.layers import Dense, Lambda, K, Add, Activation, add

import utils

from orderless_nade import logdensity_model, training_model, create_input_layers
from utils import mog_layer


def test_compare_original_nade():
    """ Compare output computation with github.com/MarcCote/NADE

    This test use weights learned with reference implementation.
    Following parameters where used
    orderlessNADE.py --theano --form MoG --dataset simple.hdf5 --samples_name 0 --hlayers 1
    --n_components 1 --epoch_size 10000 --momentum 0.0 --units 16 --training_route training
    --no_validation --batch_size 5

    Training data consisted of 10000 samples drawn from normal(mean=0, sigma=1)
    The architecture here is the same as it would be created by reference implementation.
    """
    import h5py
    masked_input_layer, input_layer, mask_layer = create_input_layers(2)

    mog = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                    outputs=mog_layer(input_layer, Activation("relu")
                    (add([Dense(16)(Lambda(lambda x: x[:, :2])(masked_input_layer)),
                          Dense(16, use_bias=False)(mask_layer)])), 1))
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                            outputs=mog([masked_input_layer, input_layer, mask_layer]))

    model = training_model(inner_model, mask_seed=1)
    model.compile(loss=utils.maximize_prediction, optimizer="sgd")
    with h5py.File("tests/original_nade_weights.hdf5") as h:
        model.set_weights([
            h["final_model/parameters/W1"][()].astype(np.float32),
            h["final_model/parameters/b1"][()].astype(np.float32),
            h["final_model/parameters/Wflags"][()].astype(np.float32),
            h["final_model/parameters/V_alpha"][()].T.reshape((16, 2)).astype(np.float32),
            h["final_model/parameters/b_alpha"][()].reshape(2).astype(np.float32),
            h["final_model/parameters/V_sigma"][()].T.reshape((16, 2)).astype(np.float32),
            h["final_model/parameters/b_sigma"][()].reshape(2).astype(np.float32),
            h["final_model/parameters/V_mu"][()].T.reshape((16, 2)).astype(np.float32),
            h["final_model/parameters/b_mu"][()].reshape(2).astype(np.float32)
        ])

    np.random.seed(1)
    output = model.predict(np.random.normal(size=(5, 2)))
    # Different random generation leads to different masks
    if K.backend() == "tensorflow":
        assert np.allclose(np.array([-2.20870864, -2.12633744, -4.85813326, -3.63397837, -1.89778014]), output)
    elif K.backend() == "theano":
        assert np.allclose(np.array([-3.33089394, -2.55555928, -4.85813281, -4.85442475, -1.92244674]), output)
    else:
        raise NotImplementedError()


def test_compare_original_nade_reg():
    """ Same as test_compare_original_nade,
     but with regularization of activities in MOG layer enabled. Should not influence the output
    """
    import h5py
    masked_input_layer, input_layer, mask_layer = create_input_layers(2)

    mog = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                    outputs=mog_layer(input_layer, Activation("relu")
                    (add([Dense(16)(Lambda(lambda x: x[:, :2])(masked_input_layer)),
                          Dense(16, use_bias=False)(mask_layer)])), 1, True))
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                            outputs=mog([masked_input_layer, input_layer, mask_layer]))

    model = training_model(inner_model, mask_seed=1)
    model.compile(loss=utils.maximize_prediction, optimizer="sgd")
    with h5py.File("tests/original_nade_weights.hdf5") as h:
        model.set_weights([
            h["final_model/parameters/W1"][()].astype(np.float32),
            h["final_model/parameters/b1"][()].astype(np.float32),
            h["final_model/parameters/Wflags"][()].astype(np.float32),
            h["final_model/parameters/V_alpha"][()].T.reshape((16, 2)).astype(np.float32),
            h["final_model/parameters/b_alpha"][()].reshape(2).astype(np.float32),
            h["final_model/parameters/V_sigma"][()].T.reshape((16, 2)).astype(np.float32),
            h["final_model/parameters/b_sigma"][()].reshape(2).astype(np.float32),
            h["final_model/parameters/V_mu"][()].T.reshape((16, 2)).astype(np.float32),
            h["final_model/parameters/b_mu"][()].reshape(2).astype(np.float32)
        ])

    np.random.seed(1)
    output = model.predict(np.random.normal(size=(5, 2)))
    # Different random generation leads to different masks
    if K.backend() == "tensorflow":
        assert np.allclose(np.array([-2.20870864, -2.12633744, -4.85813326, -3.63397837, -1.89778014]), output)
    elif K.backend() == "theano":
        assert np.allclose(np.array([-3.33089394, -2.55555928, -4.85813281, -4.85442475, -1.92244674]), output)
    else:
        raise NotImplementedError()


def test_train_logdensity():
    masked_input_layer, input_layer, mask_layer = create_input_layers(2)
    mog = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                    outputs=mog_layer(input_layer, Activation("relu")
                    (Add()([Dense(16)(Lambda(lambda x: x[:, :2], output_shape=(2,))(masked_input_layer)),
                            Dense(16, use_bias=False)(mask_layer)])), 1))
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                            outputs=mog([masked_input_layer, input_layer, mask_layer]))
    model = training_model(inner_model)
    model.compile(loss=utils.maximize_prediction, optimizer=optimizers.SGD(lr=0.1),)
    model.fit(np.random.normal(0, size=(200, 2)),
              np.zeros(shape=(200, 1)),
              batch_size=10,
              verbose=0,
              epochs=5)
    test_model = logdensity_model(inner_model)
    test_model.compile(optimizer="sgd", loss=utils.maximize_prediction)
    real = test_model.evaluate(np.random.normal(0, size=(50, 2)), np.zeros(shape=(50, 1)), verbose=0)
    wrong = test_model.evaluate(np.random.normal(1, size=(50, 2)), np.zeros(shape=(50, 1)), verbose=0)
    assert real < wrong


def test_orderings():
    masked_input_layer, input_layer, mask_layer = create_input_layers(6)
    mog = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                    outputs=mog_layer(input_layer, Activation("relu")
                    (Add()([Dense(16)(Lambda(lambda x: x[:, :2], output_shape=(2,))(masked_input_layer)),
                            Dense(16, use_bias=False)(mask_layer)])), 1))
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                            outputs=mog([masked_input_layer, input_layer, mask_layer]))
    model = training_model(inner_model)
    model.compile(loss=utils.maximize_prediction, optimizer=optimizers.SGD(lr=0.1),)
    model.fit(np.random.normal(0, size=(1000, 6)),
              np.zeros(shape=(1000, 1)),
              batch_size=10,
              verbose=0,
              epochs=1)

    test_set = np.random.normal(0, size=(5, 6))
    real_ld = np.log(np.sum(np.exp(-0.5 * test_set ** 2 - 0.5 * np.log(2 * np.pi))))

    test_1_model = logdensity_model(inner_model)
    test_1_model.compile(optimizer="sgd", loss=utils.maximize_prediction)
    test_8_model = logdensity_model(inner_model, num_of_orderings=8)
    test_8_model.compile(optimizer="sgd", loss=utils.maximize_prediction)

    ld_1 = test_1_model.evaluate(test_set, np.zeros(shape=(5, 1)), verbose=0)
    ld_8 = test_8_model.evaluate(test_set, np.zeros(shape=(5, 1)), verbose=0)
    assert abs(-real_ld - ld_8) < abs(-real_ld - ld_1)
