import numpy as np


def gauss(x, w, k):
    return np.exp(-1 * (x ** 2) * (w ** 2) / (8 * (k ** 2)))


def gabor2d(n1, n2, w, theta):
    p = 1.5
    k = np.sqrt(2 * np.log(2)) * (2 ** p + 1) / (2 ** p - 1)
    radian = np.radians(theta)
    rotation = [[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]]
    kernel = np.zeros((n1, n1), dtype=np.complex64)

    for ii in range(n2):
        for jj in range(n1):
            xy = [[jj + 1 - (n1 + 1) / 2], [ii + 1 - (n2 + 1) / 2]]
            u = np.dot(rotation, xy).ravel()
            kernel[ii, jj] = (
                (w / (np.sqrt(2 * np.pi) * k))
                * gauss(2 * u[1], w, k)
                * gauss(u[0], w, k)
                * (np.cos(w * u[1]) - np.exp(-(k ** 2) / 2))
            )

    kernel_real = (-1 * kernel).real
    return kernel_real - np.mean(kernel_real)
