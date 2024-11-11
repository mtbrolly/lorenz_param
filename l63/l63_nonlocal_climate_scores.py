"""
Compute climate scores for spatially-correlated forcing experiment.
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

nonlocality = 'moderate'
amplitude = 'very_very_small'
decorrelation_time = 'moderate'
noise_type = 'multiplicative'

data_dir = (f"data/l63/climate/{noise_type}_{nonlocality}_nonlocality/"
            + f"{amplitude}/" + f"{decorrelation_time}_decorrelation/")

# Load true climate data.
X_VAR1 = np.load(data_dir + "X_VAR1_nonlocal.npy").squeeze()

# Load other climate data.
X_VAR1_local = np.load(data_dir + "X_VAR1_local.npy").squeeze()
X = np.load(data_dir + "X.npy").squeeze()


# Compute climate scores.

score_log = pd.DataFrame(columns=['VAR1_local', 'unforced'],
                         index=['KL', 'Hellinger', 'r_dist'])

# score_log = pd.read_csv(data_dir + "climate_scores.csv", index_col=0)

models = [X_VAR1, X_VAR1_local, X]
model_labels = [r"$\mathrm{VAR}(1)$ nonlocal", r"$\mathrm{VAR}(1)$ local",
                r"Unforced"]
color = ['#36382E', '#F06449', 'grey']
ls = ['-', '-.', '--']

skip = 100
score_log.loc['Hellinger', 'unforced'] = scores.hellinger_distance_via_kde(
    X_VAR1[::skip, 0], X[::skip, 0])
score_log.loc['Hellinger', 'VAR1_local'] = scores.hellinger_distance_via_kde(
    X_VAR1[::skip, 0], X_VAR1_local[::skip, 0])

score_log.loc['KL', 'unforced'] = scores.relative_entropy_via_kde(
    X_VAR1[::skip, 0], X[::skip, 0])
score_log.loc['KL', 'VAR1_local'] = scores.relative_entropy_via_kde(
    X_VAR1[::skip, 0], X_VAR1_local[::skip, 0])


# Compare temporal autocovariances.

r = scores.temporal_autocovariance(X[:, 0])
r_VAR1 = scores.temporal_autocovariance(X_VAR1[:, 0])
r_VAR1_local = scores.temporal_autocovariance(X_VAR1_local[:, 0])

# Compute L2 distance of autocovariance functions normalised by integral of
# true autocovariance.

max_tn = int(10. * lyapunov_time / dt)
max_tn_plt = max_tn // 2

r_VAR1_norm = np.mean(r_VAR1[:max_tn])

score_log.loc['r_dist', 'unforced'] = np.sqrt(
    np.mean((r_VAR1[:max_tn] - r[:max_tn]) ** 2)) / r_VAR1_norm
score_log.loc['r_dist', 'VAR1_local'] = np.sqrt(
    np.mean((r_VAR1[:max_tn] - r_VAR1_local[:max_tn]) ** 2)) / r_VAR1_norm

score_log.to_csv(data_dir + "climate_scores.csv")
