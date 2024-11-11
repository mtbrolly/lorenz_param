"""
Module for simulation of the Lorenz '96 model with/without parameterisation.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # noqa
import numba
import numpy as np
import tensorflow as tf
from numba import jit
tf.keras.backend.set_floatx("float64")


@jit(nopython=True, cache=True, parallel=True)
def l96_tendency(X, Y, h=1., F=20., b=10., c=10.):
    """
    Calculate tendencies for the X and Y variables of the Lorenz '96 model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments,
        dYdt (1D ndarray): Array of Y increments
    """

    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = (- X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F
                   - h * c / b * np.sum(Y[k * J: (k + 1) * J]))
    for j in range(J * K):
        dYdt[j] = (
            - c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1])
            - c * Y[j] + h * c / b * X[int(j / J)])
    return dXdt, dYdt


@jit(nopython=True, cache=True, parallel=True)
def l96_reduced_tendency(X, F=20.):
    """
    Calculate tendencies for the X variables of the reduced Lorenz '96 model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        F (float): Forcing term

    Returns:
        dXdt (1D ndarray): Array of X increments
    """

    K = X.size
    dXdt = np.zeros(X.shape)
    for k in range(K):
        dXdt[k] = - X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F
    return dXdt


@jit(nopython=True, cache=True)
def l96_reduced_tendency_v(X, F=20.):
    """
    Calculate tendencies for the X variables of the reduced Lorenz '96 model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        F (float): Forcing term

    Returns:
        dXdt (1D ndarray): Array of X increments
    """

    K = X.shape[-1]
    dXdt = np.zeros(X.shape)
    for k in range(K):
        dXdt[..., k] = - X[..., k - 1] * (
            X[..., k - 2] - X[..., (k + 1) % K]) - X[..., k] + F
    return dXdt


@jit(nopython=True, cache=True, parallel=True)
def integrate_l96(X_0, Y0, dt, n_steps, h=1., F=20., b=10., c=10.):
    """
    Integrate the Lorenz '96 model with RK4.

    Args:
        X_0 (1D ndarray): Initial X values.
        Y0 (1D ndarray): Initial Y values.
        dt (float): Size of the integration time step in MTU
        n_steps (int): Number of time steps integrated forward.
        h (float): Coupling constant.
        F (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        X_series [number of timesteps, X size]: X values at each time step,
        Y_series [number of timesteps, Y size]: Y values at each time step
    """

    X_series = np.zeros((n_steps + 1, X_0.size))
    Y_series = np.zeros((n_steps + 1, Y0.size))
    T0_series = np.zeros((n_steps, X_0.size))
    M_series = np.zeros((n_steps, X_0.size))
    X = np.zeros(X_0.shape)
    Y = np.zeros(Y0.shape)
    X[:] = X_0
    Y[:] = Y0
    X_series[0] = X_0
    Y_series[0] = Y0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)
    k1_dYdt = np.zeros(Y.shape)
    k2_dYdt = np.zeros(Y.shape)
    k3_dYdt = np.zeros(Y.shape)
    k4_dYdt = np.zeros(Y.shape)

    # Reduced tendencies.
    k1_dXdt_r = np.zeros(X.shape)
    k2_dXdt_r = np.zeros(X.shape)
    k3_dXdt_r = np.zeros(X.shape)
    k4_dXdt_r = np.zeros(X.shape)

    for n in range(1, n_steps + 1):

        # This bit computes and records T0(X_nm1).
        k1_dXdt_r[:] = l96_reduced_tendency(X, F=F)
        k2_dXdt_r[:] = l96_reduced_tendency(X + k1_dXdt * dt / 2, F=F)
        k3_dXdt_r[:] = l96_reduced_tendency(X + k2_dXdt * dt / 2, F=F)
        k4_dXdt_r[:] = l96_reduced_tendency(X + k3_dXdt * dt, F=F)

        T0_series[n - 1] = X + (
            k1_dXdt_r + 2 * k2_dXdt_r + 2 * k3_dXdt_r + k4_dXdt_r) / 6 * dt
        # End of bit.

        # This bit computes and records X_n.
        k1_dXdt[:], k1_dYdt[:] = l96_tendency(X, Y, h=h, F=F, b=b, c=c)
        k2_dXdt[:], k2_dYdt[:] = l96_tendency(X + k1_dXdt * dt / 2,
                                              Y + k1_dYdt * dt / 2,
                                              h=h, F=F, b=b, c=c)
        k3_dXdt[:], k3_dYdt[:] = l96_tendency(X + k2_dXdt * dt / 2,
                                              Y + k2_dYdt * dt / 2,
                                              h=h, F=F, b=b, c=c)
        k4_dXdt[:], k4_dYdt[:] = l96_tendency(X + k3_dXdt * dt,
                                              Y + k3_dYdt * dt,
                                              h=h, F=F, b=b, c=c)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt
        Y += (k1_dYdt + 2 * k2_dYdt + 2 * k3_dYdt + k4_dYdt) / 6 * dt
        X_series[n] = X
        Y_series[n] = Y
        # End of bit.

    # Compute M_n given X_np1 and T0(X_n).
    M_series[:] = X_series[1:] - T0_series

    return X_series, Y_series, T0_series, M_series


@jit(nopython=True, cache=True, parallel=True)
def integrate_reduced_l96(X_0, dt, n_steps, F=20.):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X_0 (1D ndarray): Initial X values.
        dt (float): Size of the integration time step in MTU
        n_steps (int): Number of time steps integrated forward.
        F (float): Forcing term.

    Returns:
        X_series [number of timesteps, X size]: X values at each time step
    """

    X_series = np.zeros((n_steps, X_0.size))
    X = np.zeros(X_0.shape)
    X[:] = X_0
    X_series[0] = X_0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)

    for n in range(1, n_steps):
        k1_dXdt[:] = l96_reduced_tendency(X, F=F)
        k2_dXdt[:] = l96_reduced_tendency(X + k1_dXdt * dt / 2, F=F)
        k3_dXdt[:] = l96_reduced_tendency(X + k2_dXdt * dt / 2, F=F)
        k4_dXdt[:] = l96_reduced_tendency(X + k3_dXdt * dt, F=F)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt

        X_series[n] = X
    return X_series


@tf.function
def integrate_reduced_l96_tf(X_0, dt, n_steps, param, param_type=None, F=20.,
                             M_m1=None):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X_0 (ndarray): Initial X values, shape (N_s, K).
        dt (float): Size of the integration time step in MTU.
        n_steps (int): Number of time steps integrated forward.
        param (TensorFlow Function): Evaluates parameterisation.
        F (float): Forcing term.
        M_m1 (ndarray): M at time -1

    Returns:
        X_series [n_steps, N_s, K]: X values at each time step
    """

    @tf.function
    def l96_reduced_tendency_tf(X, F):
        """
        Calculate tendencies for the X variables of the reduced Lorenz '96
        model.

        Args:
            X (ndarray): Values of X variables at the current time step.
            F (float): Forcing term.

        Returns:
            dXdt (ndarray): Array of X increments, shape (N_s, K).
        """

        K = X.shape[-1]
        dXdt = tf.TensorArray(tf.float64, size=K)
        for k in tf.range(K):
            dXdt_k = (- X[:, k - 1] * (X[:, k - 2] - X[:, (k + 1) % K])
                      - X[:, k] + F)
            dXdt = dXdt.write(k, dXdt_k)
        return tf.transpose(dXdt.stack())

    X_series = tf.TensorArray(dtype=tf.float64, size=n_steps + 1)
    M_series = tf.TensorArray(dtype=tf.float64, size=n_steps)
    X_series = X_series.write(0, X_0)
    X_n = tf.identity(X_0)
    if M_m1:
        M_nm1 = M_m1
    else:
        M_nm1 = 0. * X_0  # Not used.

    for n in tf.range(1, n_steps + 1):
        if n % 1000 == 0:
            tf.print(n)

        # This bit computes M_nm1.
        if param_type == 'MSD':
            M_nm1 = param(X_n)
        elif param_type == 'Markov':
            M_nm1 = param(X_n, M_nm1)
        M_series = M_series.write(n - 1, M_nm1)
        # End of bit.

        # This bit computes T0(X_nm1).
        k1_dXdt = l96_reduced_tendency_tf(X_n, F)
        k2_dXdt = l96_reduced_tendency_tf(X_n + k1_dXdt * dt / 2, F)
        k3_dXdt = l96_reduced_tendency_tf(X_n + k2_dXdt * dt / 2, F)
        k4_dXdt = l96_reduced_tendency_tf(X_n + k3_dXdt * dt, F)
        T0_nm1 = X_n + dt * (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6
        # End of bit.

        # This bit increments X_n
        X_n = T0_nm1 + M_nm1
        # End of bit.

        X_series = X_series.write(n, X_n)

    return (tf.transpose(X_series.stack(), (1, 0, 2)),
            tf.transpose(M_series.stack(), (1, 0, 2)))


@tf.function
def integrate_reduced_l96_tf(X_0, dt, n_steps, param,
                                 param_type=None, F=20., M_m1=None, t_p=1):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X_0 (ndarray): Initial X values, shape (N_s, K).
        dt (float): Size of the integration time step in MTU.
        t_p (int): Number of integration time steps per parameterisation time
                   step.
        n_steps (int): Number of time steps integrated forward.
        param (TensorFlow Function): Evaluates parameterisation.
        F (float): Forcing term.
        M_m1 (ndarray): M at time -1

    Returns:
        X_series [n_steps, N_s, K]: X values at each time step
    """

    @tf.function
    def l96_reduced_tendency_tf(X, F):
        """
        Calculate tendencies for the X variables of the reduced Lorenz '96
        model.

        Args:
            X (ndarray): Values of X variables at the current time step.
            F (float): Forcing term.

        Returns:
            dXdt (ndarray): Array of X increments, shape (N_s, K).
        """

        K = X.shape[-1]
        dXdt = tf.TensorArray(tf.float64, size=K)
        for k in tf.range(K):
            dXdt_k = (- X[:, k - 1] * (X[:, k - 2] - X[:, (k + 1) % K])
                      - X[:, k] + F)
            dXdt = dXdt.write(k, dXdt_k)
        return tf.transpose(dXdt.stack())

    X_series = tf.TensorArray(dtype=tf.float64, size=n_steps + 1)
    M_series = tf.TensorArray(dtype=tf.float64, size=n_steps)
    X_series = X_series.write(0, X_0)
    X_n = tf.identity(X_0)
    if tf.is_tensor(M_m1):
        M_nm1 = M_m1
    else:
        M_nm1 = 0. * X_n

    for n in tf.range(1, n_steps + 1):
        if n % 1000 == 0:
            tf.print("n =", n)

        # This bit computes M_nm1.
        if (n - 1) % t_p == 0:
            # Only update M every t_p time steps.
            if param_type == 'MSD':
                M_nm1 = param(X_n)
            elif param_type == 'Markov':
                M_nm1 = param(X_n, M_nm1)
            else:
                M_nm1 = 0. * X_n
        M_series = M_series.write(n - 1, M_nm1)
        # End of bit.

        # This bit computes T0(X_nm1).
        k1_dXdt = l96_reduced_tendency_tf(X_n, F)
        k2_dXdt = l96_reduced_tendency_tf(X_n + k1_dXdt * dt / 2, F)
        k3_dXdt = l96_reduced_tendency_tf(X_n + k2_dXdt * dt / 2, F)
        k4_dXdt = l96_reduced_tendency_tf(X_n + k3_dXdt * dt, F)
        T0_nm1 = X_n + dt * (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6
        # End of bit.

        # This bit increments X_n
        X_n = T0_nm1 + M_nm1
        # End of bit.

        X_series = X_series.write(n, X_n)

    return (tf.transpose(X_series.stack(), (1, 0, 2)),
            tf.transpose(M_series.stack(), (1, 0, 2)))


@tf.function
def integrate_reduced_l96_tf_weather(X_0, dt, n_steps, param,
                                   param_type=None, F=20., M_m1=None, t_p=1):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X_0 (ndarray): Initial X values, shape (N_w, N_s, K).
        dt (float): Size of the integration time step in MTU.
        t_p (int): Number of integration time steps per parameterisation time
                   step.
        n_steps (int): Number of time steps integrated forward.
        param (TensorFlow Function): Evaluates parameterisation.
        F (float): Forcing term.
        M_m1 (ndarray): M at time -1

    Returns:
        X_series [n_steps, N_s, K]: X values at each time step
    """

    @tf.function
    def l96_reduced_tendency_tf(X, F):
        """
        Calculate tendencies for the X variables of the reduced Lorenz '96
        model.

        Args:
            X (ndarray): Values of X variables at the current time step.
            F (float): Forcing term.

        Returns:
            dXdt (ndarray): Array of X increments, shape (N_w, N_s, K).
        """

        K = X.shape[-1]
        dXdt = tf.TensorArray(tf.float64, size=K)
        for k in tf.range(K):
            dXdt_k = (- X[..., k - 1] * (X[..., k - 2] - X[..., (k + 1) % K])
                      - X[..., k] + F)
            dXdt = dXdt.write(k, dXdt_k)
        return tf.transpose(dXdt.stack(), (1, 2, 0))

    X_series = tf.TensorArray(dtype=tf.float64, size=n_steps + 1)
    M_series = tf.TensorArray(dtype=tf.float64, size=n_steps)
    X_series = X_series.write(0, X_0)
    X_n = tf.identity(X_0)
    if tf.is_tensor(M_m1):
        M_nm1 = M_m1
    else:
        M_nm1 = 0. * X_n

    for n in tf.range(1, n_steps + 1):
        if n % 1000 == 0:
            tf.print("n =", n)

        # This bit computes M_nm1.
        if (n - 1) % t_p == 0:
            # Only update M every t_p time steps.
            if param_type == 'MSD':
                M_nm1 = param(X_n)
            elif param_type == 'Markov':
                M_nm1 = param(X_n, M_nm1)
            else:
                M_nm1 = 0. * X_n
        M_series = M_series.write(n - 1, M_nm1)
        # End of bit.

        # This bit computes T0(X_nm1).
        k1_dXdt = l96_reduced_tendency_tf(X_n, F)
        k2_dXdt = l96_reduced_tendency_tf(X_n + k1_dXdt * dt / 2, F)
        k3_dXdt = l96_reduced_tendency_tf(X_n + k2_dXdt * dt / 2, F)
        k4_dXdt = l96_reduced_tendency_tf(X_n + k3_dXdt * dt, F)
        T0_nm1 = X_n + dt * (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6
        # End of bit.

        # This bit increments X_n
        X_n = T0_nm1 + M_nm1
        # End of bit.

        X_series = X_series.write(n, X_n)

    # return X_series, M_series
    return (tf.transpose(X_series.stack(), (1, 2, 0, 3)),
            tf.transpose(M_series.stack(), (1, 2, 0, 3)))


@jit(nopython=True, cache=True)
def polyval(x, p):
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[-(i + 1)]
    return y


@jit(nopython=True, cache=True)
def l96_polyar(X, M_noise_m1, poly_c, phi, sigma, F=20.):
    M_det = polyval(X, poly_c)
    M_noise = phi * M_noise_m1 + sigma * np.random.randn(X.size)
    return M_det, M_noise


@jit(nopython=True, cache=True)
def l96_polyar_v(X, M_noise_m1, poly_c, phi, sigma, F=20.):
    M_det = polyval(X, poly_c)
    M_noise = phi * M_noise_m1 + sigma * np.random.randn(*X.shape)
    return M_det, M_noise


@jit(nopython=True, cache=True)
def integrate_reduced_l96_polyar(X0, dt, n_steps, poly_c, phi, sigma, F=20.,
                                 t_p=1):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X0 (1D ndarray): Initial X values.
        dt (float): Size of the integration time step in MTU.
        n_steps (int): Number of time steps integrated forward.
        poly_c (1D ndarray): Coefficients of polynomial.
        F (float): Forcing term.
        parameterisation: Parameterisation (a function of X variables and the
                                            reduced tendency).

    Returns:
        X_series [number of timesteps, X size]: X values at each time step
    """

    X_series = np.zeros((n_steps + 1, X0.size))
    M_series = np.zeros((n_steps, X0.size))
    X = np.zeros(X0.shape)
    X[:] = X0
    X_series[0] = X0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)
    M_noise = np.random.randn(X0.size) * sigma ** 2 / (1 - phi ** 2)

    for n in range(1, n_steps + 1):
        if (n - 1) % t_p == 0:
            M_det, M_noise = l96_polyar(X, M_noise, poly_c, phi, sigma, F=F)
            M = M_det + M_noise
        k1_dXdt[:] = l96_reduced_tendency(X, F=F)
        k2_dXdt[:] = l96_reduced_tendency(X + k1_dXdt * dt / 2, F=F)
        k3_dXdt[:] = l96_reduced_tendency(X + k2_dXdt * dt / 2, F=F)
        k4_dXdt[:] = l96_reduced_tendency(X + k3_dXdt * dt, F=F)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt + M

        X_series[n] = X
        M_series[n - 1] = M
    return X_series, M_series


@jit(nopython=True, cache=True)
def integrate_reduced_l96_polyar_weather(X0, dt, n_steps, poly_c, phi, sigma, F=20.,
                                   t_p=1):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X0 (ndarray): Initial X values, shape (N_w, N_s, K).
        dt (float): Size of the integration time step in MTU.
        n_steps (int): Number of time steps integrated forward.
        poly_c (1D ndarray): Coefficients of polynomial.
        F (float): Forcing term.
        parameterisation: Parameterisation (a function of X variables and the
                                            reduced tendency).

    Returns:
        X_series [number of timesteps, X size]: X values at each time step
    """

    X_series = np.zeros(X0.shape[:2] + (n_steps + 1, ) + X0.shape[-1:])
    M_series = np.zeros(X0.shape[:2] + (n_steps, ) + X0.shape[-1:])
    X = np.zeros(X0.shape)
    X[:] = X0
    X_series[:, :, 0] = X0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)
    M_noise = np.random.randn(X0.size).reshape(
        X0.shape) * sigma ** 2 / (1 - phi ** 2)

    for n in range(1, n_steps + 1):
        if (n - 1) % t_p == 0:
            M_det, M_noise = l96_polyar_v(X, M_noise, poly_c, phi, sigma, F=F)
            M = M_det + M_noise
        k1_dXdt[:] = l96_reduced_tendency_v(X, F=F)
        k2_dXdt[:] = l96_reduced_tendency_v(X + k1_dXdt * dt / 2, F=F)
        k3_dXdt[:] = l96_reduced_tendency_v(X + k2_dXdt * dt / 2, F=F)
        k4_dXdt[:] = l96_reduced_tendency_v(X + k3_dXdt * dt, F=F)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt + M

        X_series[:, :, n] = X
    M_series[:, :, n - 1] = M
    return X_series, M_series
