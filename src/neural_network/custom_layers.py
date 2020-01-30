import tensorflow as tf

from src.utils.fourier import get_half_period_fourier_basis_as_rows


class CenteredConv1D(tf.keras.layers.Layer):
    """
    Ajoute du padding de part du Layer. Permet de retrouver la réponse impulsionnelle de a convolution
    centrée sur l'input

    Permet de faire des convolutions avec des filtres de kernel_size * dilation_rate arbitraire et garantit que
    l'opération ne retourne pas un vecteur null
    """
    def __init__(self, filters, kernel_size, activation, dilation_rate=1):
        super(CenteredConv1D, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, dilation_rate=dilation_rate)
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def call(self, x):
        equiv_kernel_size = (self.kernel_size - 1) * self.dilation_rate
        pad_pre = equiv_kernel_size // 2
        if (equiv_kernel_size % 2) == 0:
            pad_post = equiv_kernel_size // 2
        else:
            pad_post = equiv_kernel_size // 2 + 1

        x_pad = tf.pad(x, [[0, 0], [pad_pre, pad_post], [0, 0]])
        x_conv = self.conv(x_pad)
        return x_conv


class SkipConnection(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(SkipConnection, self).__init__()
        self.layer = layer

    def call(self, x):
        return tf.concat([x, self.layer(x)], axis=-1)

    def get_config(self, *args, **kwargs):
        return {}


class ConcatNoise(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatNoise, self).__init__()

    def call(self, x):
        g = tf.random.normal(
            x.shape,
            mean=0.0,
            stddev=1.0,
            dtype=tf.dtypes.float32,
            seed=None,
            name=None)

        return tf.concat([g, x], axis=-1)

    def get_config(self, *args, **kwargs):
        return {}


class ConcatFourier(tf.keras.layers.Layer):
    def __init__(self, nb_vectors):
        super(ConcatFourier, self).__init__()
        self.basis_initialized = False
        self.nb_vectors = nb_vectors

    def call(self, x):
        if not self.basis_initialized:
            window_len = x.shape[1]
            self.fourier_basis = tf.expand_dims(tf.constant(get_half_period_fourier_basis_as_rows(window_len, self.nb_vectors), dtype=tf.float32), 0)
            self.basis_initialized = True

        tiled_fourier = tf.tile(self.fourier_basis, multiples=(x.shape[0], 1, 1))
        return tf.concat([x, tiled_fourier], axis=-1)


class ParallelLayers(tf.keras.layers.Layer):
    def __init__(self, layer_list):
        super(ParallelLayers, self).__init__()
        self.layer_list = layer_list

    def call(self, x):
        return tf.concat( [layer(x) for layer in self.layer_list], axis=-1)


    def get_config(self, *args, **kwargs):
        return {}



if __name__ == "__main__":
    import numpy as np
    concat_fourier = ConcatFourier(window_len=1024, nb_vectors=10)
    x = np.zeros([2, 1024, 1]).astype(np.float32)
    res = concat_fourier(x)


