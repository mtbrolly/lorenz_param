"""
Compute weather scores for spatially-correlated forcing experiment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scores
plt.style.use('./paper.mplstyle')
plt.ioff()


# Setup.
dt = 1e-3
lyapunov_time = 1. / 0.9056
n_steps = int(1e5)  # Duration of chunks from climate run.
forecast_steps = int(3. * lyapunov_time / dt)  # Duration of forecasts.
T = dt * n_steps
t_series = np.arange(0, dt * n_steps, dt)
tl_series = t_series / lyapunov_time
n_samples = 100

nonlocality = 'moderate'
amplitude = 'very_very_small'
decorrelation_time = 'moderate'
noise_type = 'multiplicative'

data_dir = (f"data/l63/climate/{noise_type}_{nonlocality}_nonlocality/"
            + f"{amplitude}/" + f"{decorrelation_time}_decorrelation/")
weather_data_dir = (f"data/l63/weather/{noise_type}_{nonlocality}_nonlocality/"
                    + f"{amplitude}/" + f"{decorrelation_time}_decorrelation/")

# Load true weathers.
X_VAR1 = np.load(weather_data_dir + "X_VAR1_nonlocal.npy").squeeze()

# Load forecast ensembles.
X_VAR1s = np.load(weather_data_dir + "X_VAR1_nonlocals.npy").squeeze()
X_VAR1_local = np.load(weather_data_dir + "X_VAR1_local.npy").squeeze()
X = np.load(weather_data_dir + "X.npy")[:, None]


# Compute and plot weather scores.

score_log = pd.DataFrame(columns=['VAR1_nonlocal', 'VAR1_local', 'unforced'],
                         index=['energy', 'spread_error_ratio', 'log_score'])

ensemble = [X_VAR1s, X_VAR1_local, X]
ensemble_labels = [r"$\mathrm{VAR}(1)$ nonlocal", r"$\mathrm{VAR}(1)$ local",
                   r"Unforced"]
color = ['#36382E', '#F06449', 'grey']
ls = ['-', '-.', '--']

# Plot ensembles.
coord = 0
weather = -1

N = n_samples
fig, ax = plt.subplots(2, 1, figsize=(6, 2.5))
ax[1].plot(tl_series[:forecast_steps],
           X_VAR1[weather, :forecast_steps, coord], 'k', zorder=100)
for i in range(N):
    ax[1].plot(tl_series[:forecast_steps], X_VAR1s[weather, i, :, coord],
               color='grey', alpha=0.1)

ax[0].plot(tl_series[:forecast_steps],
           X_VAR1[weather, :forecast_steps, coord], 'k', zorder=100)
for i in range(N):
    ax[0].plot(tl_series[:forecast_steps], X_VAR1_local[weather, i, :, coord],
               color=color[1], alpha=0.1)
ax[1].set_xlabel(r"$t_{\lambda}$")
for i in range(2):
    ax[i].set_xlim(0, tl_series[forecast_steps // 1])
for i in range(len(ensemble[:-1])):
    ax[i].set_ylabel(ensemble_labels[1 - i])
plt.tight_layout()
plt.savefig(weather_data_dir + "ensembles" + ".png")
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(6, 2.5))
ax[1].plot(tl_series[:forecast_steps],
           X_VAR1[weather, :forecast_steps, coord], 'k', zorder=100)
for i in range(3):
    ax[1].fill_between(
        tl_series[:forecast_steps],
        X_VAR1s[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_VAR1s[weather, :, :, coord].std(axis=0),
        X_VAR1s[weather, :, :, coord].mean(axis=0)
        + (i+1) * X_VAR1s[weather, :, :, coord].std(axis=0),
        color='grey', alpha=0.4 - i * 0.05, edgecolor=None, zorder=-i)

ax[0].plot(tl_series[:forecast_steps],
           X_VAR1[weather, :forecast_steps, coord], 'k', zorder=100)
for i in range(3):
    ax[0].fill_between(
        tl_series[:forecast_steps],
        X_VAR1_local[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_VAR1_local[weather, :, :, coord].std(axis=0),
        X_VAR1_local[weather, :, :, coord].mean(axis=0)
        + (i + 1) * X_VAR1_local[weather, :, :, coord].std(axis=0),
        color=color[1], alpha=0.4 - i * 0.05, edgecolor=None, zorder=-i)
ax[1].set_xlabel(r"$t_{\lambda}$")

for i in range(2):
    ax[i].set_xlim(0, tl_series[forecast_steps // 1])
for i in range(len(ensemble[:-1])):
    ax[i].set_ylabel(ensemble_labels[1 - i])
plt.tight_layout()
plt.savefig(weather_data_dir + "ensembles_fb" + ".png")
plt.close()


# ---------------------
skip = 100

energy = [scores.weather_score(X_VAR1[:, :forecast_steps], i,
                               score=scores.energy, skip=skip)
          for i in ensemble[:-1]]

rmseemx = [scores.weather_score(X_VAR1[:, :forecast_steps], i,
                                score=scores.rmse_of_ensemble_meanx, skip=skip)
           for i in ensemble]

msex = [scores.weather_score(X_VAR1[:, :forecast_steps], i,
                             score=scores.msex, skip=skip)
        for i in ensemble]

stdx = [scores.weather_score(X_VAR1[:, :forecast_steps], i,
                             score=scores.forecast_stdx, skip=skip)
        for i in ensemble[:-1]]

varx = [scores.weather_score(X_VAR1[:, :forecast_steps], i,
                             score=scores.forecast_varx, skip=skip)
        for i in ensemble[:-1]]

spread_error_ratiox = [stdx[i] / rmseemx[i]
                       for i in range(len(ensemble[:-1]))]
spread_error_ratiox2 = [varx[i] / msex[i]
                        for i in range(len(ensemble[:-1]))]

scores_ = [energy, rmseemx, msex, stdx, varx,
           spread_error_ratiox, spread_error_ratiox2]
score_labels = [r"Energy score", r"RMSE of ensemble mean", r"RMSE",
                r"Ensemble standard deviation", r"Ensemble variance",
                r"Spread error ratio", "Spread error ratio"]
score_file_names = ["energy", "rmseemx", "msex", "stdx", "varx",
                    "spread_error_ratiox", "spread_error_ratiox2"]

for i in range(len(scores_)):
    plt.figure()
    for j in range(len(scores_[i])):
        plt.plot(tl_series[:forecast_steps: skip], scores_[i][j],
                 ['--', '-'][j % 2],
                 label=ensemble_labels[j])
    plt.ylabel(score_labels[i])
    plt.xlabel(r"$t_{\lambda}$")
    plt.ylim(0, None)
    plt.xlim(0, tl_series[forecast_steps])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(weather_data_dir + score_file_names[i] + ".png")
    plt.close()
