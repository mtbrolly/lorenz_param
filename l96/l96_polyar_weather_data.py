"""
Simulate Lorenz '96 with poly-AR parameterisation.
"""
import pickle
import numpy as np
from time import time
from l96 import integrate_reduced_l96_polyar_weather

tp = 100
c = 4
MODEL_NAME = f"l96_polyar_c{c:.0f}_tp{tp:.0f}"
MODEL_DIR = "models/" + MODEL_NAME + "/"

with open(MODEL_DIR + r"experiment.pkl", "rb") as file:
    experiment = pickle.load(file)

c = experiment['c']
t_p = experiment['t_p']
phi = experiment['M_res_phi']
sigma = experiment['M_res_sigma']

# ------------------------ Load weather data ----------------------------------

X_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_X_n.npy")[:, 1:]
M_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy")[:, 1:]
M_nm1_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy")[:, :-1]
dt = 1e-3

# Load parameterisation.

poly_c = np.load(MODEL_DIR + "poly_coeff.npy")

# -------------------------- Do ensemble simulations --------------------------

weather_indices = np.arange(0, 1000, 10)
N_weathers = len(weather_indices)

X_true = X_true[weather_indices]
M_true = M_true[weather_indices]
M_nm1_true = M_nm1_true[weather_indices]

X0 = X_true[:, 0]
del X_true, M_true, M_nm1_true

N_realisations = 100
L_realisations = 3500

X0 = np.tile(X0[:, None], (1, N_realisations, 1))

t0 = time()
X_poly, M_poly = integrate_reduced_l96_polyar_weather(
    X0, 1e-3, L_realisations, poly_c, phi, sigma, t_p=tp)
T0 = time() - t0

np.save(MODEL_DIR + f"X_weather_N{N_realisations}_L{L_realisations}.npy",
        X_poly)
np.save(MODEL_DIR + f"M_weather_N{N_realisations}_L{L_realisations}.npy",
        M_poly)
