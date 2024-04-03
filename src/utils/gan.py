from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import initializers, layers
import tensorflow as tf
import keras_tuner
import numpy as np


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, num_features):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.num_features = num_features
        self.gen_loss_tracker = tf.keras.metrics.Mean(
            name="Mean Generator Loss")
        self.dis_loss_tracker = tf.keras.metrics.Mean(
            name="Mean Discriminator Loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]

    def compile(self, dis_optim, gen_optim, loss_function):
        super().compile()
        self.dis_optimizer = dis_optim
        self.gen_optimizer = gen_optim
        self.loss_function = loss_function

    def train_step(self, data):
        x, y = data

        batch_size = tf.shape(x)[0]
        data = tf.convert_to_tensor(x)
        noise = tf.random.normal(shape=(batch_size, self.num_features))

        generated_data = self.generator(noise)

        X = tf.concat([generated_data, data], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Adding noise for labels:
        # https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            preds = self.discriminator(X)
            dis_loss = self.loss_function(labels, preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))

        noise = tf.random.normal(shape=[batch_size, self.num_features])
        y_gen = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            preds = self.discriminator(self.generator(noise))
            gen_loss = self.loss_function(y_gen, preds)
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(gen_loss)
        self.dis_loss_tracker.update_state(dis_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.dis_loss_tracker.result()
        }

    def test_step(self, data):
        x, y = data
        preds = self.discriminator(x, training=False)
        loss = self.loss_function(y, preds)
        self.dis_loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}


class HyperGAN(keras_tuner.HyperModel):
    def __init__(self, num_features, config):
        super(HyperGAN, self).__init__()
        self.num_features = num_features
        self.config = config

    def get_discriminator(self, dropout):

        discriminator = Sequential()

        discriminator.add(
            Dense(
                256,
                input_dim=self.num_features,
                kernel_initializer=initializers.glorot_normal(
                    seed=32)))

        for _, layer in self.config['dis_layers'].items():
            discriminator.add(Dense(layer))
            activation = getattr(
                tf.keras.layers,
                self.config['dis_activation'])()
            discriminator.add(activation)

        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator

    def get_generator(self, activation_function):
        generator = Sequential()
        generator.add(
            Dense(
                64,
                input_dim=self.num_features,
                kernel_initializer=initializers.glorot_normal(
                    seed=32)))
        generator.add(activation_function)

        for _, layer in self.config['gen_layers'].items():
            generator.add(Dense(layer))
            generator.add(activation_function)

        generator.add(Dense(self.num_features))
        generator.add(activation_function)

        return generator

    def build(self, hp):
        drop_rate = hp.Float('Dropout', min_value=0, max_value=0.30)

        activation_dict = {
            'leaky_relu': layers.LeakyReLU(),
            'relu': layers.ReLU(),
            'tanh': Activation('tanh')
        }
        self.discriminator = self.get_discriminator(drop_rate)
        self.generator = self.get_generator(activation_dict['relu'])

        model_gan = GAN(self.discriminator, self.generator, self.num_features)

        optimizer = tf.keras.optimizers.legacy.Adam()
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
        model_gan.compile(optimizer, optimizer, binary_crossentropy)
        return model_gan

    def mean_bc_score(self, y, preds):
        results = []
        n_part = np.floor(len(y) / 10)
        for i in range(10):
            ix_start, ix_end = int(i * n_part), int(i * n_part + n_part)
            bc = tf.keras.losses.BinaryCrossentropy()
            kl_div = bc(y[ix_start:ix_end], preds[ix_start:ix_end]).numpy()
            results.append(kl_div)
        return np.mean(results)

    def mean_kl_score(self, y, preds):
        results = []
        n_part = np.floor(len(y) / 10)
        for i in range(10):
            ix_start, ix_end = int(i * n_part), int(i * n_part + n_part)
            kl = tf.keras.losses.KLDivergence()
            kl_div = kl(y[ix_start:ix_end], preds[ix_start:ix_end]).numpy()
            results.append(kl_div)
        return np.mean(results)

    def fit(self, hp, model, data, callbacks=None, **kwargs):
        x, y = data.x, data.y
        model.fit(x, y, batch_size=data.batch_size, **kwargs)

        return (model.dis_loss_tracker.result().numpy() +
                model.gen_loss_tracker.result().numpy()) / 2
