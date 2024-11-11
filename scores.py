"""
Scores for assessment of parameterisations.
"""

import numpy as np
import scipy
import warnings
from numba import jit


def weather_score(X_true, X_ensemble, score=None, skip=1):
    """
    Compute a score for ensemble forecasts.

    Args:
        X_true (ndarray) [N_weathers, N_steps, dimension]: true weathers
        X_ensemble (ndarray) [N_weathers, N_samples, N_steps, dimension]:
            ensemble predictions
        score (function): score for ensemble forecasts
    """
    X_true = X_true[:, ::skip]
    X_ensemble = X_ensemble[:, :, ::skip]
    N_steps = X_true.shape[1]
    scores = np.zeros((N_steps, ), dtype=X_true.dtype)
    for i in range(N_steps):
        scores[i] = score(X_true[:, i], X_ensemble[:, :, i])
    return scores


def energy(X_true, X_ensemble):
    return (np.mean(np.linalg.norm(
        X_true[:, None, ...] - X_ensemble, ord=2, axis=-1))
        - 0.5 * np.mean(np.linalg.norm(
            X_ensemble[:, :, None] - X_ensemble[:, None], ord=2, axis=-1)))


def relative_entropy_via_kde(reference, parameterised):
    ref_kde = scipy.stats.gaussian_kde(reference)
    param_kde = scipy.stats.gaussian_kde(parameterised)

    def relative_entropy_integrand(x):
        return ref_kde(x) * (
            np.log(ref_kde(x)) - np.log(param_kde(x)))

    integral, abs_error_estimate = scipy.integrate.quad(
        relative_entropy_integrand, reference.min(), reference.max())
    if abs_error_estimate > 1e-2 * integral:
        warnings.warn(
            "Relative entropy estimate may be inaccurate due to quadrature.")
    return integral


def hellinger_distance_via_kde(reference, parameterised):
    ref_kde = scipy.stats.gaussian_kde(reference)
    param_kde = scipy.stats.gaussian_kde(parameterised)

    def relative_entropy_integrand(x):
        return (np.sqrt(ref_kde(x)) - np.sqrt(param_kde(x))) ** 2

    integral, abs_error_estimate = scipy.integrate.quad(
        relative_entropy_integrand, reference.min(), reference.max())
    if abs_error_estimate > 1e-2 * integral:
        warnings.warn(
            "Hellinger distance estimate may be inaccurate due to quadrature.")
    return (0.5 * integral) ** 0.5


@jit(nopython=True, cache=True)
def numba_mean(X, axis=None):
    """Because numba is stupid.
    """
    return X.sum(axis=axis) / X.shape[axis]


def temporal_spectrum(data, time_axis=0):
    return np.abs(np.fft.rfft(data, axis=time_axis)
                  ) ** 2 / data.shape[time_axis]


def temporal_autocovariance(data, time_axis=0):
    return np.fft.irfft(
        temporal_spectrum(data, time_axis=time_axis), axis=time_axis)
