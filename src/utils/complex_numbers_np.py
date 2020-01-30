import numpy as np

def mult(z1, z2):
    a, b = z1
    c, d = z2
    real = a * c - b * d
    imag = a * d + b * c
    return (real, imag)


def conj(z):
    re, im = z
    return (re, -im)


def normalize_to_unit(z):
    a, b = z
    norm = modulus(z) + 1E-14
    return (a/norm, b/norm)


def modulus(z):
    a, b = z
    return np.sqrt(a**2 + b**2)


def div_phases(z1, z2):
    return mult(z1, conj(z2))

