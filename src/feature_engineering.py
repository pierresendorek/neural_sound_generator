from src.complex_numbers_np import normalize_to_unit, mult
import numpy as np


def get_fourier_transform_on_apodized(sound):
    apodisation_window = np.sin(np.linspace(0, np.pi, len(sound)))
    sound_fft = np.fft.ifft(sound * apodisation_window)
    sound_fft_real = np.real(sound_fft)
    sound_fft_imag = np.imag(sound_fft)
    return sound_fft_real, sound_fft_imag


def expand_dim_1_32_bits(sound):
    return np.expand_dims(np.array(sound), axis=1).astype(np.float32)


def reconstruct(phase_prev, delta_phase, amp):
    apodisation_window = np.sin(np.linspace(0, np.pi, len(amp)))
    delta_phase = normalize_to_unit(delta_phase)
    reconstructed_phase = mult(phase_prev, delta_phase)
    ifft_re, ifft_im = (amp * reconstructed_phase[0], amp * reconstructed_phase[1])
    ifft = ifft_re + 1.0j * ifft_im
    return np.real(np.fft.fft(ifft)) * apodisation_window







