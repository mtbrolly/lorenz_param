"""
Module for simulation of the Lorenz '63 model with stochastic forcing.
"""

import numpy as np
import numba
from numba import jit, prange


@jit(nopython=True, cache=True)
def l63_tendency(X, sigma, rho, beta):
    dXdt = np.zeros_like(X)
    dXdt[..., 0] = sigma * (X[..., 1] - X[..., 0])
    dXdt[..., 1] = X[..., 0] * (rho - X[..., 2]) - X[..., 1]
    dXdt[..., 2] = X[..., 0] * X[..., 1] - beta * X[..., 2]
    return dXdt


@jit(nopython=True, cache=True, locals={
    'X0': numba.float64[:, :],
    'dt': numba.float64,
    'n_steps': numba.int64,
    'sigma': numba.float64,
    'rho': numba.float64,
    'beta': numba.float64,
    'n_weathers': numba.int64,
    'X_series': numba.float64[:, :, :]})
def integrate_l63(X0, dt, n_steps, sigma=10., rho=28., beta=8/3):
    """
    Args:
        X0: (n_weathers, 3)

    Returns:
        X_series: (n_weathers, n_steps, 3)
    """

    X_series = np.zeros((X0.shape[0], n_steps, X0.shape[1]))
    X = X0.copy()
    X_series[:, 0] = X0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)

    for n in range(1, n_steps):
        k1_dXdt[:] = l63_tendency(X, sigma, rho, beta)
        k2_dXdt[:] = l63_tendency(X + k1_dXdt * dt / 2, sigma, rho, beta)
        k3_dXdt[:] = l63_tendency(X + k2_dXdt * dt / 2, sigma, rho, beta)
        k4_dXdt[:] = l63_tendency(X + k3_dXdt * dt, sigma, rho, beta)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt

        X_series[:, n] = X
    return X_series


@jit(nopython=True, cache=True, locals={
    'X0': numba.float64[:, :, :],
    'dt': numba.float64,
    'n_steps': numba.int64,
    'phi_1': numba.float64,
    'phi_2': numba.float64,
    'C': numba.float64[:, :],
    'n_samples': numba.int64,
    'sigma': numba.float64,
    'rho': numba.float64,
    'beta': numba.float64,
    'n_weathers': numba.int64,
    'X_series': numba.float64[:, :, :, :]})
def integrate_l63_with_additive_noise(
        X0, dt, n_steps,
        phi_1=0., phi_2=0.,
        C=np.zeros((3, 3)),
        sigma=10., rho=28., beta=8/3,
        seed=1996):
    """
    Args:
        X0: (n_weathers, n_samples, 3)
        randn: (n_weathers, n_samples, n_steps, 3)

    Returns:
        X_series: (n_weathers, n_samples, n_steps, 3)
        M_series: (n_weathers, n_samples, n_steps, 3)
    """
    n_samples = X0.shape[1]
    X_series = np.zeros((X0.shape[0], n_samples, n_steps, 3))
    X_series[..., 0, :] = X0
    X = X0.copy()

    M_series = np.zeros((X0.shape[0], n_samples, n_steps, 3))
    n_weathers, _, _ = X0.shape
    noise_shape = (n_weathers, n_samples, 3)
    np.random.seed(seed)
    scale_tril = np.linalg.cholesky(C)
    prefactor = (1 - phi_2) / (
        (1 + phi_2) * (1 - phi_1 - phi_2) * (1 + phi_1 - phi_2))
    noise = np.random.randn(*noise_shape)
    M = np.sqrt(prefactor) * matmul(
        scale_tril, noise, n_weathers, n_samples)
    M_series[..., 0, :] = M

    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)

    for n in range(1, n_steps):
        k1_dXdt[:] = l63_tendency(X, sigma, rho, beta)
        k2_dXdt[:] = l63_tendency(X + k1_dXdt * dt / 2, sigma, rho, beta)
        k3_dXdt[:] = l63_tendency(X + k2_dXdt * dt / 2, sigma, rho, beta)
        k4_dXdt[:] = l63_tendency(X + k3_dXdt * dt, sigma, rho, beta)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt

        noise = np.random.randn(*noise_shape)
        if phi_2:
            if n == 1:
                M = phi_1 / (1 - phi_2) * M + np.sqrt(
                    1 / (1 - phi_2 ** 2)) * matmul(
                        scale_tril, noise, n_weathers, n_samples)
            else:
                M = (phi_1 * M + phi_2 * M_series[..., n - 2, :]
                     + matmul(
                    scale_tril, noise, n_weathers, n_samples))
        else:
            M = phi_1 * M + matmul(scale_tril, noise, n_weathers, n_samples)
        X += M
        M_series[..., n, :] = M
        X_series[..., n, :] = X
    return X_series, M_series


@ jit(nopython=True, cache=True, locals={
    'X': numba.float64[:, :],
    'Y': numba.float64[:, :, :],
    'prod': numba.float64[:, :, :]})
def matmul(X, Y, n_weathers, n_samples):
    """For arrays X with shape (3, 3) and Y with shape
    (n_weathers, n_samples, 3), compute vectorised matrix multiplication of
    shapes (3, 3) and (3, 1).

    Returns an array with shape (n_weathers, n_samples, 3).
    """
    prod = np.zeros((n_weathers, n_samples, 3))
    for i in prange(n_weathers):
        for j in prange(n_samples):
            for k in range(3):
                for l in range(3):  # noqa
                    prod[i, j, k] += X[k, l] * Y[i, j, l]
    return prod


@jit(nopython=True, cache=True, locals={
    'X0': numba.float64[:, :, :],
    'dt': numba.float64,
    'n_steps': numba.int64,
    'phi_1': numba.float64,
    'phi_2': numba.float64,
    'C': numba.float64[:, :],
    'n_samples': numba.int64,
    'sigma': numba.float64,
    'rho': numba.float64,
    'beta': numba.float64,
    'n_weathers': numba.int64,
    'X_series': numba.float64[:, :, :, :]})
def integrate_l63_with_multiplicative_noise(
        X0, dt, n_steps, phi_1=0., phi_2=0.,
        C=np.zeros((3, 3)),
        sigma=10., rho=28., beta=8/3):
    """
    Args:
        X0: (n_weathers, n_samples, 3)
        randn: (n_weathers, n_samples, n_steps, 3)

    Returns:
        X_series: (n_weathers, n_samples, n_steps, 3)
        M_series: (n_weathers, n_samples, n_steps, 3)
    """
    n_samples = X0.shape[1]
    X_series = np.zeros((X0.shape[0], n_samples, n_steps, 3))
    X_series[..., 0, :] = X0
    X = X0.copy()

    M_series = np.zeros((X0.shape[0], n_samples, n_steps, 3))
    n_weathers, _, _ = X0.shape
    noise_shape = (n_weathers, n_samples, 3)
    np.random.seed(1996)
    scale_tril = np.linalg.cholesky(C)
    prefactor = (1 - phi_2) / (
        (1 + phi_2) * (1 - phi_1 - phi_2) * (1 + phi_1 - phi_2))
    noise = np.random.randn(*noise_shape)
    M = np.sqrt(prefactor) * matmul(
        scale_tril, noise, n_weathers, n_samples)
    M_series[..., 0, :] = M

    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)

    for n in range(1, n_steps):
        k1_dXdt[:] = l63_tendency(X, sigma, rho, beta)
        k2_dXdt[:] = l63_tendency(X + k1_dXdt * dt / 2, sigma, rho, beta)
        k3_dXdt[:] = l63_tendency(X + k2_dXdt * dt / 2, sigma, rho, beta)
        k4_dXdt[:] = l63_tendency(X + k3_dXdt * dt, sigma, rho, beta)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt

        noise = np.random.randn(*noise_shape)
        if phi_2:
            if n == 1:
                M = phi_1 / (1 - phi_2) * M + np.sqrt(
                    1 / (1 - phi_2 ** 2)) * matmul(
                        scale_tril, noise, n_weathers, n_samples)
            else:
                M = (phi_1 * M + phi_2 * M_series[..., n - 2, :]
                     + matmul(
                    scale_tril, noise, n_weathers, n_samples))
        else:
            M = phi_1 * M + matmul(scale_tril, noise, n_weathers, n_samples)
        X += M * X
        M_series[..., n, :] = M
        X_series[..., n, :] = X
    return X_series, M_series
