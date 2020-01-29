import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten

from src.custom_layers import ConcatNoise, CenteredConv1D, ConcatFourier, SkipConnection
from src.fourier import get_fourier_basis_as_rows, get_half_period_fourier_basis_as_rows


class Generator(tf.keras.Model):

    def __init__(self, window_len):
        super(Generator, self).__init__()
        self.window_len = window_len

        # prev window feature extractor
        self.prev_window_fe = Sequential([Flatten()])

        self.sound_gen = Sequential([  ConcatFourier(),
                                       CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=1),
                                       ConcatNoise(),
                                       ConcatFourier(),
                                       CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=4),
                                       ConcatNoise(),
                                       ConcatFourier(),
                                       CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=16),
                                       ConcatNoise(),
                                       ConcatFourier(),
                                       CenteredConv1D(filters=256, kernel_size=10, activation="tanh", dilation_rate=64),
                                       ConcatNoise(),
                                       ConcatFourier(),
                                       CenteredConv1D(filters=1, kernel_size=5, activation=None, dilation_rate=256),
                                       ])


    def call(self, prev_window):
        prev_window_features = self.prev_window_fe(prev_window)
        tiled_features = tf.tile(tf.expand_dims(prev_window_features, axis=1), [1, self.window_len,1])
        return self.sound_gen(tiled_features)


        return prev_window_features
        #prev_window_replicated = tf.tile(tf.expand_dims(prev_window_features, axis=1), [1, self.window_len, 1])



if __name__ == "__main__":
    import numpy as np
    win_len = 1024

    x = np.zeros([3, 1024, 1]).astype(np.float32)

    generator = Generator(win_len)

    res = generator(x)

    print(res.shape)
