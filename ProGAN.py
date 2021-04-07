import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend
from matplotlib import pyplot


class Weighted_Sum(layers.Add):
    """
    This layer combines the activations from two input layers, weighted with the variable alpha.
    This class extends tf.keras.layers.Add and uses the hyperparameter 'alpha'
    to control the contribution of each input.
    The layer assumes a list of 2 inputs: - the output of existing layers
                                          - the output of the new layers
    Alpha is defined as a backend variable meaning that it can be changed via changing
    the value of the variable.
    """

    def __init__(self, alpha=0.0):
        super(Weighted_Sum, self).__init__()
        self.alpha = backend.variable(alpha, name='ws_alpha')

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
    with the batch of real samples (to avoid mode collapse).
    The standard deviation of each feature across channels is calculated and then averaged over the minibatch.
    """

    def __init__(self):
        super(Minibatch_std, self).__init__()

        def call(self, inputs):
            mean = backend.mean(inputs, axis=0, keepdims=True)
            squ_diff = backend.square(inputs - mean)
            mean_squ_diff = backend.mean(squ_diff, axis=0, keepdims=True)
            mean_squ_diff += 1e-8
            std = backend.sqrt(mean_squ_diff)
            mean_pixel = backend.mean(std, keepdims=True)
            shape = backend.shape(inputs)
            minib_featuremap = backend.tile(mean_pixel, (shape[0], shape[1], shape[2], 1))
            output = backend.concatenate([inputs, minib_featuremap], axis=-1)
            return output


class Pixel_Normalization(layers.Layer):
    """
    This layer implements pixel-wise feature vector normalization which is used in the generator.
    It is used to normalize each pixel in the activation maps to unit length after each convolution layer.
    For formula look in documentation.
    """

    def __init__(self):
        super(Pixel_Normalization, self).__init__()

        def call(self, inputs):
            values = inputs ** 2
            mean = backend.mean(values, axis=-1, keepdims=True)
            mean += 1e-8
            l2 = backend.sqrt(mean)
            normalized = inputs / l2
            return normalized


def wgan_loss(y_true, y_pred):
    """
    Wasserstein loss function.
    :param y_true: is the real label
    :param y_pred: is the predicted label
    """

    return backend.mean(y_true * y_pred)


def define_discriminator(n_blocks, input_shape=(4, 4, 3)):
    """
    This function defines the discriminator.
    Weights are initialized with the He normal initializer and clipped from -0.01 to 0.01.
    :param n_blocks: is the number of layer blocks we want to add.
              Eg: n_blocks = 3 means the discriminator will grow from 4x4 to 8x8 and then 16x16
    :param input_shape: (default is (4,4,3)) defines the shape of the input image
    """
    init = tf.keras.initializers.he_normal()
    const = tf.keras.constraints.min_max_norm(-0.01, 0.01)
    # list to save models in later
    model_list = list()

    in_image = layers.Input(shape=input_shape)
    # input layer
    d = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # output block
    d = Minibatch_std()(d)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = Flatten()(d)
    out_class = Dense(1)(d)
    # define model
    model = Model(in_image, out_class)
    model.compile(loss=wgan_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without fade-in
        old_model = model_list[i-1][0]
        # create new model for next resolution
        models = add_disc_block(old_model)
        model_list.append(models)
    return model_list


def add_disc_block(old_model, n_input_layers=3):
    """
    This function adds a new discriminator block with the next resolution (double the shape).
    The models are weighted with the Weighted_Sum().
    Weights are initialized with the He normal initializer and clipped from -0.01 to 0.01.
    :param old_model: model with old input resolution and without faded in layer
    :param n_input_layers: (default = 3) is the number of input layers which are defined in order
                           to be able to skip them for the old model.
    :return: list of 2 models, one straight through and one faded-in version
    """
    init = tf.keras.initializers.he_normal()
    const = tf.keras.constraints.min_max_norm(-0.01, 0.01)
    in_shape = list(old_model.input.shape)
    input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)
    in_image = layers.Input(shape=input_shape)
    # input layer
    d = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # new layer block
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # average pooling layer to decrease size
    d = layers.AveragePooling2D()(d)
    new_block = d

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight through model
    model1 = Model(in_image, d)
    model1.compile(loss=wgan_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # downsample new larger input image
    downsample = layers.AveragePooling2D()(in_image)
    old_block = old_model.layers[1](downsample)
    old_block = old_model.layers[2](old_block)
    # fade in output of old model input layer with new input
    d = Weighted_Sum()([old_block, new_block])
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model2 = Model(in_image, d)
    model2.compile(loss=wgan_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]


def add_gen_block(old_model):
    """
    This function adds a new layer block to the generator with double the resolution.
    The models are weighted with the Weighted_Sum().
    Weights are initialized with the He normal initializer and clipped from -0.01 to 0.01.

    :param old_model: model with old input resolution and without faded in layer
    :return: list of 2 models, one straight through and one faded in
    """
    init = tf.keras.initializers.he_normal()
    const = tf.keras.constraints.min_max_norm(-0.01, 0.01)
    # get end of last block
    end_block = old_model.layers[-2].output
    # sample up and define new block
    upsampling = layers.UpSampling2D()(end_block)
    # new block
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = Pixel_Normalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = Pixel_Normalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # new output layer
    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    model1 = Model(old_model.input, out_image)
    old_out = old_model.layers[-1]
    out_image2 = old_out(upsampling)
    merge = Weighted_Sum()([out_image2, out_image])
    model2 = Model(old_model.input, merge)
    return [model1, model2]


def define_generator(latent_dim, n_blocks, in_dim=4):
    """
    This function defines the generator which takes a latent vector as input and generates images.
    To increase the resolution the output of the last layer block is upsampled with the nearest neighbor method.
    Weights are initialized with the He normal initializer and clipped from -0.01 to 0.01.
    :param latent_dim: dimension of latent space
    :param n_blocks: number of layer blocks we want to add
    :param in_dim: dimension of input, default 4 because we start with a 4x4 resolution
    :return: list of models
    """
    init = tf.keras.initializers.he_normal()
    const = tf.keras.constraints.min_max_norm(-0.01, 0.01)
    # list to safe models in later
    model_list = list()
    # base model latent input
    in_latent = layers.Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g = Dense(128*in_dim*in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = layers.Reshape((in_dim, in_dim, 128))(g)
    # input block
    g = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = Pixel_Normalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = Pixel_Normalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # output block
    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

    # define model
    model = Model(in_latent, out_image)
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        old_model = model_list[i-1][0]
        models = add_gen_block(old_model)
        model_list.append(models)
    return model_list


def define_composite(discs, gens):
    """
    Since the generator models are trained via the discriminator models, we need to combine them.
    This function creates a new model for each pair of models that stack the generator on top of the discriminator
    which in turn classifies these generated images as real or fake. The loss of this is then used to train the generator,
    so the weights of the discriminator are marked as not trainable, to ensure they are not changed while training the generator.
    :param discs: discriminator models
    :param gens: generator models
    :return: a list of 2 models, one straight-through and one faded in
    """
    model_list = list()
    for i in range(len(discs)):
        g_models, d_models = gens[i], discs[i]
        # straight-through model
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wgan_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # fade-in model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wgan_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # safe
        model_list.append([model1, model2])
    return model_list



def load_real_samples(filename):
    """
    This loads the dataset
    :param filename: same of the file where the data is stored
    :return: returns an array with scaled data
    """
    # load dataset
    data = np.load(filename)
    # extract numpy array
    X = data['arr_0']
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


def generate_real_samples(dataset, n_samples):
    """
    This function selects real samples.
    :param dataset:
    :param n_samples: number of samples
    :return: array of images, and labels for them
    """
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    """
    This function generates points in the latent space as input for the generator.
    :param latent_dim: dimension of the latent space
    :param n_samples: number of samples
    :return: batch of inputs for the generator
    """
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    """
    The function uses the generator to generate a number of fake examples with their class label.
    :param generator: generator model
    :param latent_dim: dimension of the latent space
    :param n_samples: number of samples
    :return: fake examples with their class label
    """
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = -np.ones((n_samples, 1))
    return X, y


def update_fadein(models, step, n_steps):
    """
    This function updates the alpha value on each instance of the Weighted_Sum().
    :param models: list of models for which the alpha value has to be calculated
    :param step: step size
    :param n_steps: number of steps
    """
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, Weighted_Sum):
                backend.set_value(layer.alpha, alpha)


# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    """
    This function trains a generator and a discriminator.
    :param g_model: generator model
    :param d_model: discriminator model
    :param gan_model: composite model of generators and discriminators
    :param dataset: the data to use
    :param n_epochs: number of training epochs
    :param n_batch: batchsize
    :param fadein: bool, default False, whether to use a faded-in version or not
    """
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # update alpha for all WeightedSum layers when fading in new blocks
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        # prepare real and fake samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update the generator via the discriminator's error
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, d_loss1, d_loss2, g_loss))


def scale_dataset(images, new_shape):
    """
    This function scales the images to the preferred size.
    :param images: input images
    :param new_shape: size of preferred shape
    :return: array of images with new shape
    """
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = np.resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)


def summarize_performance(status, g_model, latent_dim, n_samples=25):
    """
    This function summarizes the performance. It generates samples and saves them as a plot, as well as saves the model.
    :param status: 'tuned' or 'faded'
    :param g_model: generator model
    :param latent_dim: dimension of latent space
    :param n_samples: number of samples (default: 25)
    """
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i])
    # save plot to file
    filename1 = 'plot_%s.png' % (name)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%s.h5' % (name)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


def train(g_models, d_models, gan_models, dataset, latent_dim, epochs_norm, epochs_fadein, n_batch):
    """
    This function trains the generator and the discriminator.
    :param g_models: generator models
    :param d_models: discriminator models
    :param gan_models: composite model of generators and discriminators
    :param dataset:
    :param latent_dim: dimension of latent space
    :param epochs_norm: number of epochs for straight through models
    :param epochs_fadein: number of epochs for faded-in models
    :param n_batch: batchsize
    """
    # fit the baseline model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, epochs_norm[0], n_batch[0])
    summarize_performance('tuned', g_normal, latent_dim)
    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print('Scaled Data', scaled_data.shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, epochs_fadein[i], n_batch[i], True)
        summarize_performance('faded', g_fadein, latent_dim)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, epochs_norm[i], n_batch[i])
        summarize_performance('tuned', g_normal, latent_dim)


"""
Hyperparameters
"""
# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 6
# size of the latent space
latent_dim = 100
# define models
d_models = define_discriminator(n_blocks)
# define models
g_models = define_generator(latent_dim, n_blocks)
# define composite models
gan_models = define_composite(d_models, g_models)
# load image data
dataset = load_real_samples('img_align_celeba_128.npz')
print('Loaded', dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)





