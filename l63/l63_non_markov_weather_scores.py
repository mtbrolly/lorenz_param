"""
Compute weather scores for non-Markovian forcing experiment.
"""

import numpy as np
from pathlib import Path
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

non_Markovianity = 'moderate'
amplitude = 'small'

data_dir = (f"data/l63/climate/{non_Markovianity}_non_markovianity/"
            + f"{amplitude}_amplitude/")
weather_data_dir = (f"data/l63/weather/{non_Markovianity}_non_markovianity/"
                    + f"{amplitude}_amplitude/")
if not Path(weather_data_dir).exists():
    Path(weather_data_dir).mkdir(parents=True)

# Load true weathers.
X_AR2 = np.load(weather_data_dir + "X_AR2.npy").squeeze()

# Load forecast ensembles.
X_AR1 = np.load(weather_data_dir + "X_AR1.npy")
X_AR1_plus = np.load(weather_data_dir + "X_AR1_plus.npy")
X_AR2s = np.load(weather_data_dir + "X_AR2s.npy")
X = np.load(weather_data_dir + "X.npy")[:, None]


# Compute and plot weather scores.

# score_log = pd.DataFrame(columns=['AR1', 'AR1_plus', 'unforced'],
#                          index=['energy', 'spread_error_ratio', 'log_score'])

ensemble = [X_AR1, X_AR1_plus, X_AR2s, X]
ensemble_labels = [r"$\mathrm{AR}(1)$", r"$\mathrm{AR}(1)^{+}$",
                   r"$\mathrm{AR}(2)$", r"Unforced"]
color = ['#1f78b4', '#fc8d62', 'k', 'grey']
ls = ['-.', ':', '-', '--']

# Plot ensembles.
coord = 0
weather = -1

N = n_samples
fig, ax = plt.subplots(3, 1, figsize=(5, 3))
ax[0].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(N):
    ax[0].plot(tl_series[:forecast_steps], X_AR1[weather, i, :, coord],
               color=color[0], alpha=0.1)
ax[1].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(N):
    ax[1].plot(tl_series[:forecast_steps], X_AR1_plus[weather, i, :, coord],
               color=color[1], alpha=0.1)
ax[2].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(N):
    ax[2].plot(tl_series[:forecast_steps], X_AR2s[weather, i, :, coord],
               color='grey', alpha=0.1)
ax[2].set_xlabel(r"$t_{\lambda}$")
for i in range(3):
    ax[i].set_xlim(0, tl_series[forecast_steps // 1])
for i in range(len(ensemble[:-1])):
    ax[i].set_ylabel(ensemble_labels[i])
plt.tight_layout()
plt.savefig(weather_data_dir + "ensembles" + ".png")
plt.close()


fig, ax = plt.subplots(3, 1, figsize=(6, 3))
ax[0].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(3):
    ax[0].fill_between(
        tl_series[:forecast_steps],
        X_AR1[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_AR1[weather, :, :, coord].std(axis=0),
        X_AR1[weather, :, :, coord].mean(axis=0)
        + (i+1) * X_AR1[weather, :, :, coord].std(axis=0),
        color=color[0], alpha=0.7 - i * 0.1, edgecolor=None, zorder=-i)
ax[1].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(3):
    ax[1].fill_between(
        tl_series[:forecast_steps],
        X_AR1_plus[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_AR1_plus[weather, :, :, coord].std(axis=0),
        X_AR1_plus[weather, :, :, coord].mean(axis=0)
        + (i + 1) * X_AR1_plus[weather, :, :, coord].std(axis=0),
        color=color[1], alpha=0.7 - i * 0.1, edgecolor=None, zorder=-i)
ax[2].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(3):
    ax[2].fill_between(
        tl_series[:forecast_steps],
        X_AR2s[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_AR2s[weather, :, :, coord].std(axis=0),
        X_AR2s[weather, :, :, coord].mean(axis=0)
        + (i + 1) * X_AR2s[weather, :, :, coord].std(axis=0),
        color=color[2], alpha=0.2 - i * 0.01, edgecolor=None, zorder=-i)
ax[2].set_xlabel(r"$t_{\lambda}$")
for i in range(3):
    ax[i].set_xlim(0, tl_series[forecast_steps // 1])
for i in range(len(ensemble[:-1])):
    ax[i].set_ylabel(ensemble_labels[i])
plt.tight_layout()
plt.savefig(weather_data_dir + "ensembles_fb" + ".png")
plt.close()


# Produce rank histograms at a fixed lead time.

lead_tn = int(.25 * lyapunov_time / dt)
ranks = [scores.rank(X_AR2[:, :forecast_steps], i, lead_tn)
         for i in ensemble[:-1]]

plt.figure(figsize=(3, 3))
bin_width = 5
for i in range(len(ensemble) - 1):
    plt.hist(ranks[i].flatten(),
             bins=np.arange(0, ensemble[i].shape[1] + 1, bin_width),
             histtype='step', label=ensemble_labels[i], density=True,
             color=color[i])
plt.plot(np.linspace(0, ensemble[i].shape[1], 10),
         np.ones(10) * 0.01, 'k--', zorder=-1)
plt.xlabel('rank')
plt.ylabel(r'$p(\mathrm{rank})$')
plt.legend()
plt.tight_layout()
plt.savefig(weather_data_dir + "rank_histogram_new.png")
plt.close()

skip = 100

energy = [scores.weather_score(X_AR2[:, :forecast_steps], i,
                               score=scores.energy, skip=skip)
          for i in ensemble[:-1]]

rmseemx = [scores.weather_score(X_AR2[:, :forecast_steps], i,
                                score=scores.rmse_of_ensemble_meanx, skip=skip)
           for i in ensemble]

msex = [scores.weather_score(X_AR2[:, :forecast_steps], i,
                             score=scores.msex, skip=skip)
        for i in ensemble]

stdx = [scores.weather_score(X_AR2[:, :forecast_steps], i,
                             score=scores.forecast_stdx, skip=skip)
        for i in ensemble[:-1]]

varx = [scores.weather_score(X_AR2[:, :forecast_steps], i,
                             score=scores.forecast_varx, skip=skip)
        for i in ensemble[:-1]]

spread_error_ratiox = [stdx[i] / rmseemx[i]
                       for i in range(len(ensemble[:-1]))]
spread_error_ratiox2 = [varx[i] / msex[i]
                        for i in range(len(ensemble[:-1]))]

scores_ = [energy, rmseemx, msex, stdx, varx,
           spread_error_ratiox, spread_error_ratiox2]
score_labels = [r"Energy score", r"RMSE of ensemble mean", r"MSE",
                r"Ensemble standard deviation", r"Ensemble variance",
                r"Spread error ratio", "Spread error ratio"]
score_file_names = ["energy", "rmseemx", "msex", "stdx", "varx",
                    "spread_error_ratiox", "spread_error_ratiox2"]

ls = ['-.', ':', '-', '--']

for i in range(len(scores_)):
    plt.figure(figsize=(3, 2.5))
    for j in range(len(scores_[i])):
        plt.plot(tl_series[:forecast_steps: skip], scores_[i][j],
                 ls=ls[j],
                 label=ensemble_labels[j],
                 color=color[j], zorder=100-j)
    plt.ylabel(score_labels[i])
    plt.xlabel(r"$t_{\lambda}$")
    plt.ylim(0, None)
    plt.xlim(0, tl_series[forecast_steps])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(weather_data_dir + score_file_names[i] + ".png")
    plt.close()


# Figure with ensembles and energy score combined

# fig = plt.figure(constrained_layout=True, figsize=(6, 3))
fig = plt.figure(figsize=(6, 2.5))
gs = fig.add_gridspec(9, 10)
ax1 = fig.add_subplot(gs[:2, :5])
ax2 = fig.add_subplot(gs[3: 5, :5])
ax3 = fig.add_subplot(gs[6: 8, :5])
ax4 = fig.add_subplot(gs[1:7, 6:])
ax = (ax1, ax2, ax3)
ax[0].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(3):
    ax[0].fill_between(
        tl_series[:forecast_steps],
        X_AR1[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_AR1[weather, :, :, coord].std(axis=0),
        X_AR1[weather, :, :, coord].mean(axis=0)
        + (i+1) * X_AR1[weather, :, :, coord].std(axis=0),
        color=color[0], alpha=0.7 - i * 0.1, edgecolor=None, zorder=-i)
ax[1].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
ax[1].sharey(ax[0])
for i in range(3):
    ax[1].fill_between(
        tl_series[:forecast_steps],
        X_AR1_plus[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_AR1_plus[weather, :, :, coord].std(axis=0),
        X_AR1_plus[weather, :, :, coord].mean(axis=0)
        + (i + 1) * X_AR1_plus[weather, :, :, coord].std(axis=0),
        color=color[1], alpha=0.7 - i * 0.1, edgecolor=None, zorder=-i)
ax[2].plot(tl_series[:forecast_steps],
           X_AR2[weather, :, coord], 'k', zorder=100)
for i in range(3):
    ax[2].fill_between(
        tl_series[:forecast_steps],
        X_AR2s[weather, :, :, coord].mean(axis=0)
        - (i + 1) * X_AR2s[weather, :, :, coord].std(axis=0),
        X_AR2s[weather, :, :, coord].mean(axis=0)
        + (i + 1) * X_AR2s[weather, :, :, coord].std(axis=0),
        color=color[2], alpha=0.2 - i * 0.01, edgecolor=None, zorder=-i)
ax[2].set_xlabel(r"$t_{\lambda}$")
for i in range(3):
    ax[i].set_xlim(0, tl_series[forecast_steps // 1])
for i in range(len(ensemble[:-1])):
    ax[i].set_ylabel(ensemble_labels[i])


for j in range(len(energy)):
    ax4.plot(tl_series[:forecast_steps: skip], energy[j],
             ls=ls[j],
             label=ensemble_labels[j],
             color=color[j], zorder=100-j)
ax4.set_ylabel(r"Energy score")
ax4.set_xlabel(r"$t_{\lambda}$")
ax4.set_ylim(0, None)
ax4.set_xlim(0, tl_series[forecast_steps])
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig(weather_data_dir + "combined_figure" + ".png")
plt.close()
