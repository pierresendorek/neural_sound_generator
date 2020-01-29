from scipy.linalg import dft
import numpy as np

def get_fourier_basis_as_rows(nb_points):
    if (nb_points % 2) == 1:
        raise ValueError("Please provide an even value for nb_points")
    f = dft(nb_points)
    f_real = np.real(f)
    f_imag = np.imag(f)
    return np.concatenate([f_real[:nb_points//2, :], f_imag[:nb_points//2, :]], axis=0).astype(np.float32)


def get_half_period_fourier_basis_as_rows(nb_points):
    f = dft(2*nb_points)
    f_real = np.real(f)
    f_imag = np.imag(f)
    return np.concatenate([f_real[:nb_points, :nb_points//8], f_imag[:nb_points, :nb_points//8]], axis=1).astype(np.float32)


