from src.gan.batch_generator import BatchGenerator
from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
import numpy as np
import tensorflow as tf
import src.init_cudnn # keep this import for CuDNN to work
from scipy.io import wavfile
from pathlib import Path
import os


window_len = 1024



class GanTrainer:
    def __init__(self):
        self.discriminator = Discriminator()
        self.generator = Generator(window_len)
        self.batch_generator = BatchGenerator()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

    def generate_batch(self, sound_prev, sound_curr):
        artificial = self.generator(sound_prev)
        natural = sound_curr
        return sound_prev, natural, artificial

    @staticmethod
    def mean_square(x):
        return tf.reduce_mean(tf.square(x))

    def discriminator_loss(self, sound_prev, natural, artificial):
        d_natural = self.discriminator.call(sound_prev, natural)
        d_artificial = self.discriminator.call(sound_prev, artificial)
        loss = self.mean_square(d_natural) + \
               self.mean_square(d_artificial - tf.ones_like(d_artificial))
        return loss, (d_natural, d_artificial)

    def generator_loss(self, sound_prev, artificial):
        return self.mean_square(self.discriminator.call(sound_prev, artificial))


    def train(self):
        for i in range(10**5):
            self.train_step(100)
            print("global step ", i)

            self.generate_example_sound()

    def generate_example_sound(self):
        sound_prev, _ = self.batch_generator.draw_batch(1024, 1)
        sound_list = []
        for i in range(1000):
            sound = self.generator.call(sound_prev)
            sound_list.append(sound)
            sound_prev = sound
        long_sound = np.concatenate(sound_list, axis=1)[0, :, 0]
        long_sound = long_sound / np.max(np.abs(long_sound))
        home = str(Path.home())
        wavfile.write(os.path.join(home, "out.wav"), rate=44100, data=long_sound)

    def train_step(self, nb_steps_train):

        for _ in range(nb_steps_train):
            with tf.GradientTape() as disc_tape:
                sound_prev, sound_curr = self.batch_generator.draw_batch(window_len, 10)
                sound_prev, natural, artificial = self.generate_batch(sound_prev, sound_curr)
                disc_loss, disc_outputs = self.discriminator_loss(sound_prev, natural, artificial)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            if np.sqrt(disc_loss.numpy()) < 0.1:
                print("Stopping learning discriminator.")
                break
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


        for _ in range(nb_steps_train):
            with tf.GradientTape() as gen_tape:
              sound_prev, sound_curr = self.batch_generator.draw_batch(window_len, 10)
              sound_prev, natural, artificial = self.generate_batch(sound_prev, sound_curr)
              gen_loss = self.generator_loss(sound_prev, artificial)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

            disc_loss, disc_outputs = self.discriminator_loss(sound_prev, natural, artificial)
            print(disc_outputs)
            print(disc_loss)
            if disc_loss.numpy() > 0.4:
                print("Stopping learning generator.")
                break


            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))



if __name__ == "__main__":
    trainer = GanTrainer()
    trainer.train()

