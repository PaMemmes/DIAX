from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import initializers
import tensorflow as tf


def get_generator(config, num_features):
    generator = Sequential()
    generator.add(Dense(64, input_dim=num_features,
                  kernel_initializer=initializers.glorot_normal(seed=32)))
    generator.add(Activation('relu'))

    for _, layer in config['gen_layers'].items():
        print(Activation(config['gen_activation']))
        generator.add(Dense(layer))
        generator.add(Activation(config['gen_activation']))

    generator.add(Dense(num_features))
    generator.add(Activation('tanh'))

    optim = getattr(
        tf.optimizers.legacy,
        config['optimizer'])(
        learning_rate=config['learning_rate'],
        beta_1=config['momentum'])
    generator.compile(loss=config['loss'], optimizer=optim)

    return generator


def get_discriminator(config, num_features):

    discriminator = Sequential()

    discriminator.add(Dense(256, input_dim=num_features,
                      kernel_initializer=initializers.glorot_normal(seed=32)))

    for _, layer in config['dis_layers'].items():
        discriminator.add(Dense(layer))
        activation = getattr(tf.keras.layers, config['dis_activation'])()
        discriminator.add(activation)

    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    optim = getattr(
        tf.optimizers.legacy,
        config['optimizer'])(
        learning_rate=config['learning_rate'],
        beta_1=config['momentum'])
    discriminator.compile(loss=config['loss'], optimizer=optim)

    return discriminator


def make_gan_network(config, discriminator, generator, input_dim):
    discriminator.trainable = False
    gan_input = Input(shape=(input_dim,))
    print('gan_input', gan_input.shape)
    x = generator(gan_input)
    print('x', x.shape)
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    optim = getattr(
        tf.optimizers.legacy,
        config['optimizer'])(
        learning_rate=config['learning_rate'],
        beta_1=config['momentum'])
    gan.compile(loss='binary_crossentropy', optimizer=optim)

    return gan
