"""
Produce climate data for non-Markovian forcing experiment.
"""

import numpy as np
from l63.l63 import integrate_l63, integrate_l63_with_additive_noise
from time import time
from pathlib import Path


# ------------ Produce a climate run of each model variation.------------------

# Setup.
# X0 = np.array([0.05, 0.05, 23.55])  # Average of an orbit.
X0 = np.array([10.53345065, 15.26303085, 22.85340481])  # 1e5 steps from avg.

dt = 1e-3
n_steps = int(1e8)
T = dt * n_steps
t_series = np.arange(0, dt * n_steps, dt)
n_samples = 1

non_Markovianity = 'moderate'
amplitude = 'small'

data_dir = (f"data/l63/climate/{non_Markovianity}_non_markovianity/"
            + f"{amplitude}_amplitude/")
if not Path(data_dir).exists():
    Path(data_dir).mkdir(parents=True)

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
t_corr = dt / (1 - phi)
t_corr_plus = dt / (1 - phi_plus)

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

n_samples = 1
X0_ensemble = np.tile(X0[np.newaxis, np.newaxis, :], (1, n_samples, 1))

# AR1
T0 = time()
X_AR1, M_AR1 = integrate_l63_with_additive_noise(
    X0_ensemble, dt, n_steps, phi_1=phi, C=C_diag_AR1)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(data_dir + "X_AR1.npy", X_AR1.squeeze())
np.save(data_dir + "M_AR1.npy", M_AR1.squeeze())
del X_AR1, M_AR1

# AR1 plus
T0 = time()
X_AR1_plus, M_AR1_plus = integrate_l63_with_additive_noise(
    X0_ensemble, dt, n_steps, phi_1=phi_plus, C=C_diag_AR1_plus)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(data_dir + "X_AR1_plus.npy", X_AR1_plus.squeeze())
np.save(data_dir + "M_AR1_plus.npy", M_AR1_plus.squeeze())
del X_AR1_plus, M_AR1_plus

# AR2
T0 = time()
X_AR2, M_AR2 = integrate_l63_with_additive_noise(
    X0_ensemble, dt, n_steps, phi_1=phi_1, phi_2=phi_2, C=C_diag_AR2)
T0 = time() - T0
print(f"Integration took {T0:.0f} seconds.")
np.save(data_dir + "X_AR2.npy", X_AR2.squeeze())
np.save(data_dir + "M_AR2.npy", M_AR2.squeeze())
del X_AR2, M_AR2

# No forcing
T0 = time()
X = integrate_l63(X0[None, :], dt, int(n_steps))
T0 = time() - T0
print(f"Integration took {T0:.2f} seconds.")
np.save(data_dir + "X.npy", X)
del X
