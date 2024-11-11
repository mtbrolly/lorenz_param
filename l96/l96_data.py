"""
Produce Lorenz '96 data.
"""

import numpy as np
from l96 import integrate_l96_rk4
from time import time


# Initial state.
K = 8
J = 32
rng = np.random.default_rng(seed=1)
X0 = rng.normal(size=K)
Y0 = rng.normal(size=J * K)

F = 20.
cs = [4., 10.]

dt = 1e-3
n_steps = int(1e7)
L_weathers = int(1e4)
N_weathers = n_steps // L_weathers


for i in range(len(cs)):
    c = cs[i]

    t0 = time()
    X, Y, T0, M = integrate_l96_rk4(X0, Y0, dt, n_steps, F=F, c=c)
    t0 = time() - t0
    print(f"Integration with c={c:.0f} took {t0:.0f} seconds.")

    t = np.arange(0, dt * n_steps, dt)

    # np.savez_compressed(f"data/l96/climate_rk4_c{c:.0f}.npz",
    #                     t=t, X=X, Y=Y, T0=T0, M=M)
    np.savez_compressed(f"data/l96/climate_rk4_c{c:.0f}_without_y.npz",
                        t=t, X=X, T0=T0, M=M)

    Xw = np.zeros((N_weathers, L_weathers, K))
    Mw = np.zeros((N_weathers, L_weathers, K))
    for i in range(N_weathers):
        Xw[i] = X[i * L_weathers: (i + 1) * L_weathers]
        Mw[i] = M[i * L_weathers: (i + 1) * L_weathers]

    np.save(f"data/l96/weather_rk4_c{c:.0f}_X_n.npy", Xw)
    np.save(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy", Mw)
    print(f"Finished saving data with c={c:.0f}.")

    del X, Y, T0, M
