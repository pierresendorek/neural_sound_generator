
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten, GlobalMaxPool1D, Dense

from src.custom_layers import ConcatNoise, CenteredConv1D, ConcatFourier, SkipConnection
from src.fourier import get_fourier_basis_as_rows, get_half_period_fourier_basis_as_rows

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.sound_discriminator = Sequential([ConcatFourier(),
                                     CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=1),
                                     ConcatFourier(),
                                     CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=4),
                                     ConcatFourier(),
                                     CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=16),
                                     ConcatFourier(),
                                     CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=64),
                                     ConcatFourier(),
                                     CenteredConv1D(filters=30, kernel_size=10, activation="sigmoid", dilation_rate=256),
                                     Flatten(),
                                     Dense(30, activation="tanh"),
                                     Dense(10, activation="tanh"),
                                     Dense(1, activation="sigmoid")])


    def call(self, prev_window, curr_window):
        sound_artefact = tf.concat([prev_window, curr_window], axis=1)
        return self.sound_discriminator(sound_artefact)




