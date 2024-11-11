"""
Simulate Lorenz '96 with poly-AR parameterisation.
"""

import pickle
import numpy as np
from time import time
from l96 import integrate_reduced_l96_polyar

np.random.seed(1)

MODEL_NAME = "l96_polyar_c10_tp20"
MODEL_DIR = "models/" + MODEL_NAME + "/"

with open(MODEL_DIR + r"experiment.pkl", "rb") as file:
    experiment = pickle.load(file)

c = experiment['c']
t_p = experiment['t_p']
phi = experiment['M_res_phi']
sigma = experiment['M_res_sigma']


# Load climate data.

data = np.load(f"data/l96/climate_rk4_c{c:.0f}_without_y.npz")

X_n = data['X'][:-1]
M_n = data['M'][:]


# Load parameterisation.

poly_c = np.load(MODEL_DIR + "poly_coeff.npy")

# Run simulation.

T0 = time()
burnin = int(1e5)
X0 = X_n[0 + burnin]
N_realisations = 1
L_realisations = int(1e7)
X_polyar = np.zeros((N_realisations, L_realisations + 1, X_n.shape[-1]))
M_polyar = np.zeros((N_realisations, L_realisations, X_n.shape[-1]))

for i in range(N_realisations):
    X_polyar[i], M_polyar[i] = integrate_reduced_l96_polyar(
        X0, 1e-3, L_realisations, poly_c, phi, sigma, t_p=t_p)
T0 = time() - T0


np.save(MODEL_DIR + "X.npy",
        X_polyar.squeeze()[:-1])
np.save(MODEL_DIR + "M.npy",
        M_polyar.squeeze())
