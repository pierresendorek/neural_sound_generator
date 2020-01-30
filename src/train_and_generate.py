import tensorflow as tf
from src.batch_generator import BatchGenerator
from src.complex_numbers_np import normalize_to_unit, mult
from src.feature_engineering import reconstruct, expand_dim_1_32_bits
from src.predictor import Predictor
from math import pi
import src.init_cudnn
import numpy as np
from scipy.io import wavfile


class Trainer:

    @staticmethod
    def mean_square(x):
        return tf.reduce_mean(tf.square(x))

    @staticmethod
    def neg_log_likelihood(x, mu, log_sigma2):
        # N(x;mu, sigma2) = 1/sqrt(2 pi sigma2) exp( - 1/2 (x - mu)**2 / sigma2 )
        # - log( ... ) =  + 0.5 log(2 * pi * sigma2) + 0.5 (x - mu)**2 / sigma2
        return 0.5 * tf.reduce_mean(tf.math.log(2 * pi) + log_sigma2 + tf.square(x - mu) * tf.exp(-log_sigma2))

    def train(self):

        predictor = Predictor()
        batch_generator = BatchGenerator()
        optimizer = tf.keras.optimizers.Adam(1e-4)
        window_len = 1024

        for i_step in range(10**5):
            with tf.GradientTape() as tape:
                phase_list, features, target_amp, target_delta_phase = batch_generator.draw_batch(window_len, 10)
                amplitude_mu, log_amplitude_sigma2, delta_phase_mu, log_delta_phase_sigma2 = predictor(features)
                loss = self.neg_log_likelihood(target_amp, amplitude_mu, log_amplitude_sigma2) + \
                       0.1 * self.neg_log_likelihood(target_delta_phase, delta_phase_mu, log_delta_phase_sigma2)

            gradients = tape.gradient(loss, predictor.trainable_variables)
            optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))
            print(i_step, loss)
            if (i_step % 20) == 0:
                generate_sound(200, phase_list, features, predictor, window_len)

###########



def generate_sound(nb_steps, phase_list, features, predictor, window_len):
    # initialization phase
    phase = phase_list[0]

    amp_prev_1 = features[0, :, 0]


    y = np.zeros([(nb_steps + 1) * window_len])

    for i_step in range(nb_steps):
        amplitude_mu, log_amplitude_sigma2, delta_phase_mu, log_delta_phase_sigma2 = predictor(features)

        amp = generate_amplitude(amplitude_mu, log_amplitude_sigma2)
        delta_phase = generate_delta_phase(delta_phase_mu, log_delta_phase_sigma2)

        sound = reconstruct(phase, delta_phase, amp)

        phase = mult(phase, delta_phase)
        amp_prev_2 = amp_prev_1
        amp_prev_1 = amp

        features = np.concatenate([np.expand_dims(expand_dim_1_32_bits(a), axis=0) for a in [amp_prev_1, amp_prev_2, *delta_phase]], axis=2)

        y[(window_len//2) * i_step: window_len//2 * i_step + window_len] += sound

    y = y / np.max(np.abs(y))

    wavfile.write("../data/out.wav", 44100, y)

def generate_amplitude(amplitude_mu, log_amplitude_sigma2):
    return amplitude_mu[0, :, 0].numpy() + np.random.randn(1024) * np.exp(log_amplitude_sigma2[0, :, 0].numpy())


def generate_delta_phase(delta_phase_mu, log_delta_phase_sigma2):
    delta_phase = delta_phase_mu.numpy() + np.random.randn(1024, 2) * np.exp(log_delta_phase_sigma2.numpy())
    delta_phase_complex = (delta_phase[0, :, 0], delta_phase[0, :, 1])
    delta_phase_complex = normalize_to_unit(delta_phase_complex)
    return delta_phase_complex


Trainer().train()