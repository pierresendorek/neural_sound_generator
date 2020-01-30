from scipy.linalg import dft
import numpy as np


def get_half_period_fourier_basis_as_rows(nb_points, nb_vectors):
    f = dft(2*nb_points)
    f_real = np.real(f)
    f_imag = np.imag(f)
    return np.concatenate([f_real[:nb_points, :nb_vectors], f_imag[:nb_points, :nb_vectors]], axis=1).astype(np.float32)


