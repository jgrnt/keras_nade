# A Keras implementation of NADE

This project aims to provide an generic implementation of [NADE](https://arxiv.org/abs/1605.02226) models using Keras.

It is heavily inspired by the reference implementation https://github.com/MarcCote/NADE

Currently only Deep Orderless (R-)NADE is supported.
## Orderless Nade
### Usage

To show the complete usage of this package an [example](https://github.com/jgrnt/keras_nade/blob/master/examples/red_wine_orderless_nade.py) is included. It reimplements an [example used in the reference implementation](https://github.com/MarcCote/NADE/blob/master/deepnade/run_orderless_nade.sh).

Due to the varying structure in training and inference, we cannot create the same model for both scenarios. This library provides functions to create training and inference models given the intermediate structure. This `inner_model` get as input three layers:
  * `masked_input_layer`: concatenation of the already masked input plus the mask
  * `input_layer`: the  unmasked input
  * `mask_layer`: the mask it self (could also be derived from `masked_input_layer`)

`orderless_nade.training_model(inner_model)` creates a model, which takes the input and generates randomized masks in order to train different conditional densities. The model created by `orderless_nade.logdensity_model(inner_model,num_of_orderings=1)` instead generates for a number of orderings masks and computes the logdensity over all this masks. For this to work the
weights of the `inner_model` need to be trained beforehand.



```python
# Helper function to create input_layers in the right size
masked_input_layer, input_layer, mask_layer = orderless_nade.create_input_layers(input_size=2)
mog = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                    outputs=mog_layer(input_layer, Activation("relu")
                    (Add()([Dense(16)(Lambda(lambda x: x[:, :2], output_shape=(2,))(masked_input_layer)),
                            Dense(16, use_bias=False)(mask_layer)])), 1))

 # Layer outputing logdensities with same shape as input_size, using aboves input layers
 # Example for RNADE with Mixture of Gaussians can be found in tests
outputs = ...

              
inner_model = Container(inputs=[masked_input_layer, input_layer, mask_layer],
                        outputs=[outputs])

training_model = orderless_nade.training_model(inner_model)

 # We normally try to maximize the predicted density therefore a custom loss function to
 # maximize the density is used.
training_model.compile(loss=utils.maximize_prediction)

training_model.fit(...)

# As we use the same inner_model, we reuses the trained weights
test_model = orderless_nade.logdensity_model(inner_model)
```