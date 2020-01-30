import numpy as np
from scipy.io import wavfile
import os

from src.complex_numbers_np import div_phases, normalize_to_unit, modulus
from src.feature_engineering import get_fourier_transform_on_apodized, expand_dim_1_32_bits


class BatchGenerator:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sampling_frequency, sound = wavfile.read(dir_path + "/../data/claude-debussy-clair-de-lune.wav")
        #print("len sound", len(sound))
        normalized_sound = 0.5 * (sound - np.mean(sound)) / np.std(sound)
        cut_samples = len(normalized_sound) // 100
        self.sound = normalized_sound[cut_samples:-cut_samples]




    def get_phase(self, real, imag):
        norm = np.sqrt(real**2 + imag**2)
        return real/norm, imag/norm

    def draw_batch(self, window_len, batch_size):
        sample_starts = np.random.randint(len(self.sound) - 2*window_len, size=batch_size)
        features_list = []
        target_amp_list = []
        target_delta_phase_list = []
        phase_list = []
        for sample_start in sample_starts:
            sound_prev_2 = self.sound[sample_start:sample_start+window_len]
            sound_prev_1 = self.sound[sample_start+ window_len //2: sample_start + window_len + window_len // 2]
            sound_curr = self.sound[sample_start + window_len: sample_start + 2 * window_len]

            fft_prev_1 = get_fourier_transform_on_apodized(sound_prev_1)
            fft_prev_2 = get_fourier_transform_on_apodized(sound_prev_2)
            fft_curr = get_fourier_transform_on_apodized(sound_curr)

            phi_curr = normalize_to_unit(fft_curr)
            phi_prev_1 = normalize_to_unit(fft_prev_1)
            phi_prev_2 = normalize_to_unit(fft_prev_2)

            delta_phase_curr = div_phases(phi_curr, phi_prev_1)
            delta_phase_prev = div_phases(phi_prev_1, phi_prev_2)

            amp_curr = modulus(fft_curr)
            amp_prev_1 = modulus(fft_prev_1)
            amp_prev_2 = modulus(fft_prev_2)

            features = np.concatenate([expand_dim_1_32_bits(a) for a in (amp_prev_1, amp_prev_2, *delta_phase_prev)], axis=1)
            target_amp = expand_dim_1_32_bits(amp_curr)
            target_delta_phase = np.concatenate([expand_dim_1_32_bits(delta_phase) for delta_phase in delta_phase_curr], axis=1)

            phase_list.append(phi_curr)

            features_list.append(features)
            target_amp_list.append(target_amp)
            target_delta_phase_list.append(target_delta_phase)

        return phase_list, np.stack(features_list), np.stack(target_amp_list), np.stack(target_delta_phase_list)


if __name__ == "__main__":
    bg = BatchGenerator()

    phase_list, features, target_amp, target_delta_phase = bg.draw_batch(window_len=1024, batch_size=1)


    print(features.shape)
    print(target_delta_phase.shape)