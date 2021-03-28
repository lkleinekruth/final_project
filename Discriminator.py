import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers


# - discriminator must grow during the training process
# - this is achieved by inserting a new input layer to support
#   the larger input image followed by a new block of layers
# - the output of this new block is then downsampled
# - additionally, the new image is also downsampled directly
#   and passed through the old input processing layer before it
#   is combined with the output of the new block (-> kind of residual)

# - the output of the new block that is downsampled and the output
#   of the old input processing layer are combined using a weighted
#   average, where the weighting is controlled by a new hyperparameter
#   called alpha
# - the weighted sum is calculated: Output = ((1-alpha)*fromRGB)+(alpha*NewBlock)
# - the weighted average of the two pathways is then fed into the rest of the existing model
# - initial weighting is completely biased towards old input (alpha = 0)
#   then linearly increased over training iterations so that the new block is given more weight
#   until the output is entirely the product of the new block (alpha = 1)
#   (then old pathway can be removed)

# - the "fromRGB" layers are implemented as 1x1 conv2D
# - a block is comprised of 2 conv2D with 3x3 filters and leaky ReLU (0.2)
#   followed by a downsampling layer (average pooling 2x2, 2x2)
# - output involved 2 conv2D with 3x3 and 4x4 filters, followed by a fully
#   connected layer that outputs the single value prediction (true/false)
#   with linear activation

# - trained by WGAN-GP loss
# - model weights are initialized using he_normal

# - minibatch std layer at the beginning of output block (each layer
#   uses local response normalization referred to as pixel-wise normalization)

# - predefine all of the models prior to training and carefully use keras functional
#   API to ensure that layers are shared across the models and continue training

# - first: define custom layer to use when fading in a new higher resolution input image and block
#   - the new layer must take two sets of activation maps with the same dimension (w, h, c)
#     and add them together using a weighted sum


class WeightedSum(layers.Add):
    """
    This class extends tf.keras.layers.Add and uses the hyperparameter 'alpha'
    to control the contribution of each input.
    The layer assumes 2 inputs: - the output of existing layers
                                - the output of the new layers
    Alpha is defined as a backend variable meaning that it can be changed via changing
    the value of the variable.
    """

    def __init__(self, alpha=0.0):
        super(WeightedSum, self).__init__()
        # tf.keras.backend.variable(alpha) returns variable instance with Keras metadata
        self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')

    # outputs a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of 2 inputs
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class Minibatch_std(layers.Layer):
    """
    This class implements the mini batch standard deviation layer, which is inserted towards the end of the discriminator.
    The idea is to add a simple constant feature map so that the batch of generated samples will share similar features
    with the batch of real samples.
    It is derived from the standard deviation of all features in a batch across spatial locations.
    """

    def __init__(self):
        super(Minibatch_std, self).__init__()

        def call(self, inputs):
            # calculate the mean for each pixel across channels
            mean = tf.keras.backend.mean(inputs, axis=0, keepdims=True)
            # calculate the squared differences between pixel values and mean
            squ_diff = tf.keras.backend.square(inputs - mean)
            # calculate the variance
            mean_squ_diff = tf.keras.backend.mean(squ_diff, axis=0, keepdims=True)
            # add a small value to avoid problems when calculating std
            mean_squ_diff += 1e-8
            # calculate std
            std = tf.keras.backend.sqrt(mean_squ_diff)
            # calculate mean std across each pixel
            mean_pixel = tf.keras.backend.mean(std, keepdims=True)
            # scaling up to the size of the input feature map for each sample
            shape = tf.keras.backend.shape(inputs)
            minib_featuremap = tf.keras.backend.tile(mean_pixel, (shape[0], shape[1], shape[2], 1))
            # concatenate original input with the mini batch feature map
            output = tf.keras.backend.concatenate([inputs, minib_featuremap], axis=-1)
            return output


# - define a discriminator model that takes 4x4 color image as input and outputs a prediction (real/fake)
#   - the model is comprised of a 1x1 input processing layer (fromRGB) and an output block


class Define_Discriminator(Model):
    """
    This is the first discriminator model, which takes an 8x8x3 image as input
    """

    def __init__(self, input_shape=(8, 8, 3)):
        super(Define_Discriminator, self).__init__()

        self.model_list = list()
        self.layer_list = [
            # not sure about number of filters
            layers.Conv2D(filters=64,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer='he_normal',
                          input_shape=input_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128,
                          (3, 3),
                          padding='same',
                          kernel_initializer='he_normal'),
            Minibatch_std(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128,
                          (4, 4),
                          padding='same',
                          kernel_initializer='he_normal'),
            Minibatch_std(),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(units=1)
        ]

    @tf.function
    def call(self, x, n_blocks=6):
        for layer in self.layer_list:
            x = layer(x)
        self.model_list.append([x, x])
        # create submodels
        for i in range(1, n_blocks):
            # get prior model without the fade-in
            old_model = self.model_list[i - 1][0]
            # create new model for next resolution
            models = Intermediate_Discriminator(old_model)
            # store model
            self.model_list.append(models)

        return self.model_list


# maybe better to define as layer? Still problem with passing on old model and other call arguments
class Intermediate_Discriminator(Model):
    """
    This is the intermediate discriminator model. It returns a list of two defined models (a straight through and a fade in).
    It takes the old model as argument and the number of input layers (default 6).
    """

    def __init__(self, old_model):
        super(Intermediate_Discriminator, self).__init__()

        # get shape of existing model
        self.in_shape = list(self.old_model.input.shape)
        # define new input shape as double the size
        self.input_shape = (self.in_shape[-2].value * 2, self.in_shape[-2].value * 2, self.in_shape[-1].value)

        # define a new layer_list
        self.layer_list = [
            layers.Conv2D(64,
                          (1, 1),
                          padding='same',
                          kernel_initializer='he_normal',
                          input_shape=self.input_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(64,
                          (3, 3),
                          padding='same',
                          kernel_initializer='he_normal'),
            Minibatch_std(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(64,
                          (3, 3),
                          padding='same',
                          kernel_initializer='he_normal'),
            Minibatch_std(),
            layers.LeakyReLU(alpha=0.2),
            layers.AveragePooling2D((2, 2),
                                    (2, 2))
        ]
        # layer for sampling down the input image
        self.downsample_input = layers.AveragePooling2D((2, 2),
                                                        (2, 2))

    @tf.function
    def call(self, x, old_model, input_layers=6):
        # x is the input which we need later
        y = x
        for layer in self.layer_list:
            y = layer(y)
        new_block = y
        # skip the input, 1x1 and activation for the old model
        for i in range(input_layers, len(old_model.layers)):
            y = old_model.layers[i](y)
        model1 = y
        # sample down the original input image
        downsample = self.downsample_input(x)
        # connect old input processing to downsampled new input
        old_block = old_model.layers[1](downsample)
        old_block = old_model.layers[2](old_block)
        # fade in output of old model input layer with new input
        y = WeightedSum()([old_block, new_block])
        # skip the input, 1x1 and activation for the old model
        for i in range(input_layers, len(old_model.layers)):
            y = old_model.layers[i](y)
        model2 = y
        return [model1, model2]


