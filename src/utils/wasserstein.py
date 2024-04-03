from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Activation
from tensorflow.keras import initializers, layers
import tensorflow as tf
import keras_tuner
import numpy as np
from sklearn.metrics import accuracy_score


class WGAN(tf.keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            num_features,
            discriminator_extra_steps=3,
            gp_weight=5.0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.num_features = num_features
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.gen_loss_tracker = tf.keras.metrics.Mean(
            name="Mean Generator Loss")
        self.dis_loss_tracker = tf.keras.metrics.Mean(
            name="Mean Discriminator Loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]

    def compile(self, dis_optim, gen_optim, d_loss_function, g_loss_function):
        super().compile()
        self.dis_optimizer = dis_optim
        self.gen_optimizer = gen_optim
        self.g_loss_function = g_loss_function
        self.d_loss_function = d_loss_function

    def gradient_penalty(self, batch_size, real_data, fake_data):
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        x, y = data

        batch_size = tf.shape(x)[0]
        for i in range(self.d_steps):
            noise = tf.random.normal(shape=(batch_size, self.num_features))
            with tf.GradientTape() as tape:
                fake_x = self.generator(noise, training=True)
                fake_preds = self.discriminator(fake_x, training=True)
                real_preds = self.discriminator(x, training=True)

                d_cost = self.d_loss_function(
                    real_img=real_preds, fake_img=fake_preds)
                gp = self.gradient_penalty(batch_size, x, fake_x)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            self.dis_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        noise = tf.random.normal(shape=(batch_size, self.num_features))
        with tf.GradientTape() as tape:
            generated_data = self.generator(noise, training=True)
            gen_preds = self.discriminator(generated_data, training=True)
            g_loss = self.g_loss_function(gen_preds)

        gen_gradient = tape.gradient(
            g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        self.d_loss = d_loss
        self.g_loss = g_loss
        return {"d_loss": d_loss, "g_loss": g_loss}

    def test_step(self, data):
        x, y = data
        preds = self.discriminator(x, training=False)
        loss = self.d_loss_function(y, preds)
        self.dis_loss_tracker.update_state(loss)
        z = tf.py_function(
            func=calculate_accuracy, inp=[
                preds, y], Tout=tf.float32)
        return {m.name: m.result() for m in self.metrics}


def calculate_accuracy(preds, y):
    preds = preds.numpy()
    per = np.percentile(preds, 0.55 * 100)
    inds = preds > per
    inds_comp = preds <= per
    preds[inds] = 0
    preds[inds_comp] = 1
    acc = accuracy_score(preds, y)
    return acc


class HyperWGAN(keras_tuner.HyperModel):
    def __init__(
            self,
            num_features,
            config,
            discriminator_extra_steps,
            gp_weight):
        super(HyperWGAN, self).__init__()
        self.num_features = num_features
        self.config = config
        self.discriminator_extra_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def make_optimizers(self):
        self.dis_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

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
        generator.add(Activation('tanh'))

        return generator

    def build(self, hp):

        drop_rate = hp.Float('Dropout', min_value=0, max_value=0.30)

        activation_dict = {
            'leaky_relu': layers.LeakyReLU(),
            'relu': layers.ReLU(),
            'tanh': Activation('tanh')
        }
        self.discriminator = self.get_discriminator(drop_rate)
        self.generator = self.get_generator(activation_dict['leaky_relu'])

        model_gan = WGAN(
            self.discriminator,
            self.generator,
            self.num_features,
            self.discriminator_extra_steps,
            self.gp_weight)

        self.make_optimizers()
        model_gan.compile(
            self.dis_optimizer,
            self.gen_optimizer,
            self.discriminator_loss,
            self.generator_loss)
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
        preds = model.discriminator.predict(x)
        return (model.dis_loss_tracker.result().numpy() +
                model.gen_loss_tracker.result().numpy()) / 2
