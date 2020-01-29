from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
import numpy as np
import tensorflow as tf


window_len = 1024

discriminator = Discriminator()
generator = Generator(window_len)


def generate_batch(batch_size, window_len):
    sound_prev = np.random.randn(batch_size, window_len, 1).astype(np.float32)
    artificial = generator(sound_prev)
    natural = np.random.randn(batch_size, window_len, 1).astype(np.float32)

    #labels = np.concatenate([np.zeros([batch_size, 1]), np.ones([batch_size, 1])], axis=0).astype(np.float32)
    #sounds = np.concatenate([natural, artificial], axis=0).astype(np.float32)

    return sound_prev, natural, artificial


def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def discriminator_loss(sound_prev, natural, artificial):
    return mean_square(discriminator.call(sound_prev, natural)) + \
           mean_square(discriminator.call(sound_prev, artificial) - 1.0)

def generator_loss(sound_prev, artificial):
    return mean_square(discriminator.call(sound_prev, artificial))


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def train_step():

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      sound_prev, natural, artificial = generate_batch(batch_size=10, window_len=1024)
      gen_loss = generator_loss(sound_prev, artificial)
      disc_loss = discriminator_loss(sound_prev, natural, artificial)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


for i in range(100):
    print(i)
    train_step()


