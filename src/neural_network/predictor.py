
import tensorflow as tf

from src.utils.batch_generator import BatchGenerator
from src.neural_network.custom_layers import CenteredConv1D, ConcatFourier


class Predictor(tf.keras.Model):

    def __init__(self):
        super(Predictor, self).__init__()
        self.conv_layers = [CenteredConv1D(filters=30, kernel_size=10, activation="tanh", dilation_rate=2**i) for i in range(9)]

        self.delta_phase_mu = CenteredConv1D(filters=2, kernel_size=10, activation=None, dilation_rate=1)
        self.log_delta_phase_sigma2 = CenteredConv1D(filters=2, kernel_size=10, activation=None, dilation_rate=1)

        self.amplitude_mu = CenteredConv1D(filters=1, kernel_size=10, activation="sigmoid", dilation_rate=1)
        self.log_amplitude_sigma2 = CenteredConv1D(filters=1, kernel_size=10, activation=None, dilation_rate=1)

        self.concat_fourier = ConcatFourier(nb_vectors=4)

    def call(self, x, **kwargs):
        prev_out = self.concat_fourier(x)

        for i, conv_layer in enumerate(self.conv_layers):
            conv_out = conv_layer(prev_out)
            prev_out = tf.concat([prev_out, conv_out], axis=2)

        return self.amplitude_mu(prev_out), self.log_amplitude_sigma2(prev_out), self.delta_phase_mu(prev_out), self.log_delta_phase_sigma2(prev_out)


if __name__ == "__main__":
    p = Predictor()

    b = BatchGenerator()
    _, features, _, _ = b.draw_batch(window_len=1024, batch_size=13)

    p(features)


