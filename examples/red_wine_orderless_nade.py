import h5py
import numpy as np
import keras.backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.engine.topology import Container
from keras.layers import Activation, Lambda, Dense, Add

from orderless_nade import create_input_layers, training_model
from utils import mog_layer, maximize_prediction

# parameters
n_components = 10
n_epochs = 157
lr = 0.02
n_loops = 20
batch_size = 100


with h5py.File("red_wine.hdf5") as h_red_wine:
    training_data = np.vstack([h_red_wine["folds/1/training"][str(i)]["data"][()] for i in range(1, 9)])
    validation_data = np.repeat(h_red_wine["folds/1/training/9/data"][()], n_loops - 1, axis=0)
    test_data = h_red_wine["folds/1/tests/1/data"][()]
    mean, std = np.mean(training_data, axis=0), np.std(training_data, axis=0),

    training_data = (training_data - mean) / std
    validation_data = (validation_data - mean) / std
    test_data = (test_data - mean) / std

    masked_input_layer, input_layer, mask_layer = create_input_layers(training_data.shape[1])
    first_layer = Activation("relu")(
        Add()([
            Dense(100)(Lambda(lambda x: x[:, :11], output_shape=(11,))(masked_input_layer)),
            Dense(100, use_bias=False)(mask_layer)]))
    second_layer = Dense(100, activation="relu")(first_layer)
    mog = mog_layer(input_layer, second_layer, n_mixtures=10)
    inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                            outputs=mog)

    model = training_model(inner_model, mask_seed=1)
    model.compile(loss=maximize_prediction, optimizer=optimizers.SGD(momentum=0.9, lr=lr))

    def scheduler(epoch):
        if epoch:
            return K.get_value(model.optimizer.lr) - lr / (n_epochs + 1)
        return lr

    change_lr = LearningRateScheduler(scheduler)

    model.fit(training_data,
              np.zeros((training_data.shape[0], 1)),
              batch_size=batch_size,
              validation_data=(validation_data, np.zeros((validation_data.shape[0], 1))),
              callbacks=[change_lr],
              epochs=n_epochs)
