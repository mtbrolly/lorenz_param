"""
Produce weather data for non-Markovian forcing experiment.
"""

import numpy as np
from l63.l63 import (integrate_l63,
                     integrate_l63_with_additive_noise,
                     integrate_l63_with_multiplicative_noise)
from time import time
from pathlib import Path

# Setup.
dt = 1e-3
lyapunov_time = 1. / 0.9056
n_steps = int(1e5)  # Duration of chunks from climate run.
forecast_steps = int(3. * lyapunov_time / dt)  # Duration of forecasts.
T = dt * n_steps
t_series = np.arange(0, dt * n_steps, dt)
n_samples = 100

nonlocality = 'strong'
amplitude = 'very_very_small'
decorrelation_time = 'moderate'
noise_type = 'multiplicative'

data_dir = (f"data/l63/climate/{noise_type}_{nonlocality}_nonlocality/"
            + f"{amplitude}/" + f"{decorrelation_time}_decorrelation/")
weather_data_dir = (f"data/l63/weather/{noise_type}_{nonlocality}_nonlocality/"
                    + f"{amplitude}/" + f"{decorrelation_time}_decorrelation/")
if not Path(weather_data_dir).exists():
    Path(weather_data_dir).mkdir(parents=True)

if noise_type == 'additive':
    integrate_with_noise_fn = integrate_l63_with_additive_noise
else:
    integrate_with_noise_fn = integrate_l63_with_multiplicative_noise

if nonlocality == 'moderate':
    alpha = -0.45
elif nonlocality == 'strong':
    alpha = 0.9
elif nonlocality == 'mild':
    alpha = -0.1
elif nonlocality == 'very_strong':
    alpha = 0.9999

if amplitude == 'small':
    var_AR = 1e-4
elif amplitude == 'moderate':
    var_AR = 1e-3
elif amplitude == 'very_small':
    var_AR = 1e-5
elif amplitude == 'large':
    var_AR = 1e-2
elif amplitude == 'smaller':
    var_AR = 4e-4
elif amplitude == 'very_very_small':
    var_AR = 1e-7

if decorrelation_time == 'moderate':
    d_time = lyapunov_time
elif decorrelation_time == 'short':
    d_time = 0.1 * lyapunov_time

gamma = 1 / d_time
phi = 1 - gamma * dt

# phi = 1. - (1. / 0.1) * dt
# gamma = (1 - phi) / dt
# decorrelation_time = 1 / gamma

epsilon_AR1 = np.sqrt(var_AR * (1 - phi ** 2))
C = alpha * np.ones((3, 3))
np.fill_diagonal(C, 1.)
C *= epsilon_AR1 ** 2
C_diag = np.zeros((3, 3))
np.fill_diagonal(C_diag, epsilon_AR1 ** 2)


# ------------- Split VAR1 and unforced climate run into weathers. ------------
X_VAR1 = np.load(data_dir + "X_VAR1_nonlocal.npy")
X_VAR1 = X_VAR1.reshape(1000, 100000, 3)
np.save(weather_data_dir + "X_VAR1_nonlocal.npy", X_VAR1)
X = np.load(data_dir + "X_VAR1_nonlocal.npy")
X = X.reshape(1000, 100000, 3)
np.save(weather_data_dir + "X.npy", X)

X0 = X_VAR1[:, 0].copy()
X0_ensemble = np.tile(X0[:, np.newaxis, :], (1, n_samples, 1))


# ------------- Simulate weather ensembles for forced versions. ---------------

# VAR1 local
T0 = time()
X_VAR1_local, _ = integrate_with_noise_fn(
    X0_ensemble, dt, forecast_steps, phi_1=phi, C=C_diag)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X_VAR1_local.npy", X_VAR1_local.squeeze())
del X_VAR1_local

# VAR1 nonlocal
T0 = time()
X_VAR1s, _ = integrate_with_noise_fn(
    X0_ensemble, dt, forecast_steps, phi_1=phi, C=C)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X_VAR1_nonlocals.npy", X_VAR1s.squeeze())
del X_VAR1s
