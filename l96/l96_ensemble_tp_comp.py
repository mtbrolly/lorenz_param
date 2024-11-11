import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')
plt.ioff()

c = 10


# ------------------------ Load weather data ----------------------------------

X_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_X_n.npy")[:, 1:]
M_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy")[:, 1:]
M_nm1_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy")[:, :-1]
dt = 1e-3


# ------------------------- Load ensemble simulations -------------------------

weather_indices = np.arange(0, 1000, 10)
N_weathers = len(weather_indices)

X_true = X_true[weather_indices]
M_true = M_true[weather_indices]
M_nm1_true = M_nm1_true[weather_indices]


N_realisations = 100
L_realisations = 3500


model_dir = f"models/l96_MDN_nonlocal_MSD_c{c}_nc32_npl128/"

t_p = [1, 20]
X_NNs = [np.load(model_dir + f"tp_{tp}/X_weather_tp{tp:.0f}_N100_L3500.npy")
         for tp in t_p]

weather_index_index = 0
weather_index = weather_indices[weather_index_index]

# Plot realisations vs truth.
k = 0
ts0 = np.arange(0, X_true.shape[1] * dt, dt)

fig, ax = plt.subplots(2, 1, figsize=(6, 4))
ax[0].plot(ts0, X_true[weather_index_index, :, k], 'k', label="Full system",
           zorder=10)
for j in range(3):
    if j == 0:
        label = r"Ensemble mean $\pm 3\, \mathrm{std}$"
    else:
        label = "_"
    ax[0].fill_between(
        ts0[:L_realisations + 1],
        np.mean(X_NNs[0][weather_index_index, :, :, k], axis=0)
        - (j + 1) * np.std(X_NNs[0][weather_index_index, :, :, k], axis=0),
        np.mean(X_NNs[0][weather_index_index, :, :, k], axis=0)
        + (j + 1) * np.std(X_NNs[0][weather_index_index, :, :, k], axis=0),
        color='#386641', alpha=0.3 - 0.05 * j, edgecolor=None,
        label=label, zorder=4 - j)
ax[0].set_ylabel(rf"$t_p=$ {t_p[0]}")
ax[0].set_xlim(0, dt * X_NNs[0].shape[2])
ax[0].grid(True)

ax[1].plot(ts0, X_true[weather_index_index, :, k], 'k', label="Full system",
           zorder=10)
for j in range(3):
    if j == 0:
        label = r"Ensemble mean $\pm 3\, \mathrm{std}$"
    else:
        label = "_"
    ax[1].fill_between(
        ts0[:L_realisations + 1],
        np.mean(X_NNs[1][weather_index_index, :, :, k], axis=0)
        - (j + 1) * np.std(X_NNs[1][weather_index_index, :, :, k], axis=0),
        np.mean(X_NNs[1][weather_index_index, :, :, k], axis=0)
        + (j + 1) * np.std(X_NNs[1][weather_index_index, :, :, k], axis=0),
        color='#386641', alpha=0.3 - 0.05 * j, edgecolor=None,
        label=label, zorder=4 - j)
ax[1].set_xlabel(r"$t$")
ax[1].set_ylabel(rf"$t_p=$ {t_p[1]}")
ax[1].grid(True)
ax[1].set_xlim(0, dt * X_NNs[0].shape[2])
fig.tight_layout()
plt.savefig("figures/nonlocal_ensemble_tp_comp.png", dpi=576)
plt.close()
