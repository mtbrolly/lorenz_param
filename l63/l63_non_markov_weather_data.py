"""
Produce weather data for non-Markovian forcing experiment.
"""

import numpy as np
from l63.l63 import integrate_l63, integrate_l63_with_additive_noise
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

non_Markovianity = 'strong'
amplitude = 'small'

data_dir = (f"data/l63/climate/{non_Markovianity}_non_markovianity/"
            + f"{amplitude}_amplitude/")
weather_data_dir = (f"data/l63/weather/{non_Markovianity}_non_markovianity/"
                    + f"{amplitude}_amplitude/")
if not Path(weather_data_dir).exists():
    Path(weather_data_dir).mkdir(parents=True)

if non_Markovianity == 'moderate':
    phi_1 = 0.45
    phi_2 = 0.5
elif non_Markovianity == 'strong':
    phi_1 = 0.045
    phi_2 = 0.95
elif non_Markovianity == 'very_strong':
    phi_1 = 0.00001
    phi_2 = 0.99998

if amplitude == 'small':
    var_AR = 1e-4
elif amplitude == 'moderate':
    var_AR = 1e-3
elif amplitude == 'very_small':
    var_AR = 1e-5

phi = phi_1 / (1 - phi_2)  # Correct one-step autocorrelation.
phi_plus = 0.5 * (phi_1 + np.sqrt(phi_1 ** 2 + 4 * phi_2))  # Better fit.

epsilon_AR2 = np.sqrt(
    var_AR * (1 + phi_2) * (1 - phi_1 - phi_2) *
    (1 + phi_1 - phi_2) / (1 - phi_2))

epsilon_AR1 = np.sqrt(var_AR * (1 - phi ** 2))
epsilon_AR1_plus = np.sqrt(var_AR * (1 - phi_plus ** 2))

C_diag_AR1 = np.zeros((3, 3))
np.fill_diagonal(C_diag_AR1, epsilon_AR1 ** 2)

C_diag_AR1_plus = np.zeros((3, 3))
np.fill_diagonal(C_diag_AR1_plus, epsilon_AR1_plus ** 2)

C_diag_AR2 = np.zeros((3, 3))
np.fill_diagonal(C_diag_AR2, epsilon_AR2 ** 2)


# ------------------- Split AR2 climate run into weathers. --------------------
X_AR2 = np.load(data_dir + "X_AR2.npy")
X_AR2 = X_AR2.reshape(1000, 100000, 3)


X0 = X_AR2[:, 0].copy()


# Repeat AR2 weathers from same initial conditions to ensure stationarity of M.
T0 = time()
X_AR2, M_AR2 = integrate_l63_with_additive_noise(
    X0[:, None], dt, forecast_steps, phi_1=phi_1, phi_2=phi_2,
    C=C_diag_AR2, seed=1)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X_AR2.npy", X_AR2)


X0_ensemble = np.tile(X0[:, np.newaxis, :], (1, n_samples, 1))
del X_AR2, M_AR2


# --------- Simulate weather ensembles for other model variations. ------------

# AR1
T0 = time()
X_AR1, M_AR1 = integrate_l63_with_additive_noise(
    X0_ensemble, dt, forecast_steps, phi_1=phi, C=C_diag_AR1)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X_AR1.npy", X_AR1)
del X_AR1, M_AR1


# AR1 plus
T0 = time()
X_AR1_plus, M_AR1_plus = integrate_l63_with_additive_noise(
    X0_ensemble, dt, forecast_steps, phi_1=phi_plus, C=C_diag_AR1_plus)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X_AR1_plus.npy", X_AR1_plus)
del X_AR1_plus, M_AR1_plus


# No forcing
T0 = time()
X = integrate_l63(X0, dt, forecast_steps)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X.npy", X)
del X


# AR2 ensemble
T0 = time()
X_AR2s, M_AR2s = integrate_l63_with_additive_noise(
    X0_ensemble, dt, forecast_steps, phi_1=phi_1, phi_2=phi_2, C=C_diag_AR2)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(weather_data_dir + "X_AR2s.npy", X_AR2s)
del X_AR2s, M_AR2s
