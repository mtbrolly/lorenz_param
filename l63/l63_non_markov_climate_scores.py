"""
Compute climate scores for non-Markovian forcing experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scores
plt.style.use('./paper.mplstyle')
plt.ioff()


# Setup.
dt = 1e-3
lyapunov_time = 1. / 0.9056
n_steps = int(1e8)
T = dt * n_steps
t_series = np.arange(0, dt * n_steps, dt)
tl_series = t_series / lyapunov_time

non_Markovianity = 'strong'
amplitude = 'small'

data_dir = (f"data/l63/climate/{non_Markovianity}_non_markovianity/"
            + f"{amplitude}_amplitude/")

# Load true climate data.
X_AR2 = np.load(data_dir + "X_AR2.npy").squeeze()

# Load other climate data.
X_AR1 = np.load(data_dir + "X_AR1.npy").squeeze()
X_AR1_plus = np.load(data_dir + "X_AR1_plus.npy").squeeze()
X = np.load(data_dir + "X.npy").squeeze()


# Compute climate scores.

score_log = pd.DataFrame(columns=['AR1', 'AR1_plus', 'unforced'],
                         index=['KL', 'Hellinger', 'r_dist'])

# score_log = pd.read_csv(data_dir + "climate_scores.csv", index_col=0)

models = [X_AR1, X_AR1_plus, X]
model_labels = [r"$\mathrm{AR}(1)$", r"$\mathrm{AR}(1)^{+}$", r"Unforced"]
color = ['k', '#1f78b4', '#fc8d62', 'grey']
ls = ['-', '-.', ':', '--']

skip = 100
score_log.loc['Hellinger', 'unforced'] = scores.hellinger_distance_via_kde(
    X_AR2[::skip, 0], X[::skip, 0])
score_log.loc['Hellinger', 'AR1'] = scores.hellinger_distance_via_kde(
    X_AR2[::skip, 0], X_AR1[::skip, 0])
score_log.loc['Hellinger', 'AR1_plus'] = scores.hellinger_distance_via_kde(
    X_AR2[::skip, 0], X_AR1_plus[::skip, 0])

score_log.loc['KL', 'unforced'] = scores.relative_entropy_via_kde(
    X_AR2[::skip, 0], X[::skip, 0])
score_log.loc['KL', 'AR1'] = scores.relative_entropy_via_kde(
    X_AR2[::skip, 0], X_AR1[::skip, 0])
score_log.loc['KL', 'AR1_plus'] = scores.relative_entropy_via_kde(
    X_AR2[::skip, 0], X_AR1_plus[::skip, 0])


# Compare temporal autocovariances.

r = scores.temporal_autocovariance(X[:, 0])
r_AR2 = scores.temporal_autocovariance(X_AR2[:, 0])
r_AR1 = scores.temporal_autocovariance(X_AR1[:, 0])
r_AR1_plus = scores.temporal_autocovariance(X_AR1_plus[:, 0])

# Compute L2 distance of autocovariance functions normalised by integral of
# true autocovariance.

max_tn = int(10. * lyapunov_time / dt) // 2

r_AR2_norm = np.mean(r_AR2[:max_tn])

score_log.loc['r_dist', 'unforced'] = np.sqrt(
    np.mean((r_AR2[:max_tn] - r[:max_tn]) ** 2)) / r_AR2_norm
score_log.loc['r_dist', 'AR1'] = np.sqrt(
    np.mean((r_AR2[:max_tn] - r_AR1[:max_tn]) ** 2)) / r_AR2_norm
score_log.loc['r_dist', 'AR1_plus'] = np.sqrt(
    np.mean((r_AR2[:max_tn] - r_AR1_plus[:max_tn]) ** 2)) / r_AR2_norm


score_log.to_csv(data_dir + "climate_scores.csv")
