"""
Generate reduced Lorenz '96 climate data.
"""

import numpy as np
from time import time
from l96 import integrate_reduced_l96

np.random.seed(1)
c = 4

MODEL_NAME = f"l96_reduced_c{c:.0f}"
MODEL_DIR = "models/" + MODEL_NAME + "/"

# Load climate data.

data = np.load(f"data/l96/climate_rk4_c{c:.0f}_without_y.npz")

X_n = data['X'][:-1]
M_n = data['M'][:]


# Run simulation.

T0 = time()
burnin = int(1e5)
X0 = X_n[0 + burnin]
L_realisations = int(1e7)
X_reduced = integrate_reduced_l96(X0, 1e-3, L_realisations)
T0 = time() - T0


np.save(MODEL_DIR + "X.npy", X_reduced)
