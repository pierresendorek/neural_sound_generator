from src.gan.batch_generator import BatchGenerator
from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
import numpy as np
import tensorflow as tf
import src.init_cudnn # keep this import for CuDNN to work

window_len = 1024

discriminator = Discriminator()
generator = Generator(window_len)


def generate_batch(sound_prev, sound_curr):
    artificial = generator(sound_prev)
    natural = sound_curr
    return sound_prev, natural, artificial


def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def discriminator_loss(sound_prev, natural, artificial):
    d_natural = discriminator.call(sound_prev, natural)
    d_artificial = discriminator.call(sound_prev, artificial)
    loss = mean_square(d_natural) + \
           mean_square(d_artificial - tf.ones_like(d_artificial))
    return loss, (d_natural, d_artificial)

def generator_loss(sound_prev, artificial):
    return mean_square(discriminator.call(sound_prev, artificial))


generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

batch_generator = BatchGenerator()

nb_steps_train = 100

def train_step(nb_steps_train):

    for _ in range(nb_steps_train):
        with tf.GradientTape() as disc_tape:
            sound_prev, sound_curr = batch_generator.draw_batch(window_len, 10)
            sound_prev, natural, artificial = generate_batch(sound_prev, sound_curr)
            disc_loss, res = discriminator_loss(sound_prev, natural, artificial)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        print(res)

    for _ in range(nb_steps_train):
        with tf.GradientTape() as gen_tape:
          sound_prev, sound_curr = batch_generator.draw_batch(window_len, 10)
          sound_prev, natural, artificial = generate_batch(sound_prev, sound_curr)
          gen_loss = generator_loss(sound_prev, artificial)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))












for i in range(10**6):
    train_step(100)
    print(i)


