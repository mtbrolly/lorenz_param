import numpy as np
import pandas as pd
import scores as scores_module
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')
plt.ioff()

c = 10
t_p = [1, 10, 20, 30, 50, 100]
dt = 1e-3

scores = [pd.read_csv(
    f"models/l96_polyar_c{c:.0f}_tp{i}/climate_scores_with_diag.csv",
    index_col=0) for i in t_p]

score_names = ['KL', 'r_dist']
score_names_plt = ['KL divergence',
                   'Temporal autocorrelation error']
model_names = ['MDN_nonlocal', "MDN_nonlocal_diag", 'MDN_local',
               'Poly_AR1']
model_names_plt = ["MDN nonlocal", "MDN weakly local", "MDN strongly local",
                   r"Poly-AR$(1)$"]

co = ['#386641', 'grey', '#0077b6', '#d62828']
ls = ['--', '-.', (5, (10, 3)), ':']

# ----------

# --- nonlocal

locality = 'nonlocal'
model_dir = f"models/l96_MDN_{locality}_MSD_c{c:.0f}_nc32_npl128/"
X_NNs = [np.load(model_dir + f"tp_{tp}/X_weather_tp{tp:.0f}_N100_L3500.npy")
         for tp in t_p]

weather_indices = np.arange(0, 1000, 10)
X_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_X_n.npy"
                 )[weather_indices, 1:1 + X_NNs[0].shape[2]]

energy_nl = [scores_module.energy(
    X_true[:, 1000], p[:, :, 1000]) for p in X_NNs]

# --- local

locality = 'local'
model_dir = f"models/l96_MDN_{locality}_MSD_c{c:.0f}_nc32_npl128/"
X_NNs = [np.load(model_dir + f"tp_{tp}/X_weather_tp{tp:.0f}_N100_L3500.npy")
         for tp in t_p[1:]]
X_NNs = [np.load(
    model_dir + f"tp_{t_p[0]}/X_weather_tp{t_p[0]}_N50_L3500.npy"), ] + X_NNs

energy_l = [scores_module.energy(
    X_true[:, 1000], p[:, :, 1000]) for p in X_NNs]

# --- polyar

X_NNs = [np.load(f"models/l96_polyar_c{c}_tp{tp}/X_weather_N100_L3500.npy")
         for tp in t_p]
energy_p = [scores_module.energy(
    X_true[:, 1000], p[:, :, 1000]) for p in X_NNs]

# --- diag
locality = 'nonlocal_diag'
model_dir = f"models/l96_MDN_{locality}_MSD_c{c:.0f}_nc32_npl128/"
X_NNs = [np.load(model_dir + f"tp_{tp}/X_weather_tp{tp:.0f}_N100_L3500.npy")
         for tp in t_p]
energy_diag = [scores_module.energy(
    X_true[:, 1000], p[:, :, 1000]) for p in X_NNs]

# ---

energy = [energy_nl, energy_diag, energy_l, energy_p]

# ----------

fig, ax = plt.subplots(3, 1, figsize=(4, 8))
for i in range(len(score_names)):
    for j in range(len(model_names)):
        score_tp = [sl[model_names[j]][score_names[i]] for sl in scores]
        ax[i].plot(t_p, score_tp, 'o', label=model_names_plt[j],
                   c=co[j], ls=ls[j], zorder=10 - j)
    ax[i].set_ylabel(score_names_plt[i])
    ax[i].set_ylim(0, None)
    ax[i].set_xlim(0, 102)
    if i == 0:
        ax[i].legend()
    ax[i].grid()

for j in range(len(model_names)):
    ax[2].plot(t_p, energy[j], 'o', label=model_names_plt[j],
               c=co[j], ls=ls[j], zorder=10 - j)
ax[2].grid()
ax[2].set_xlabel(r'$t_p$')
ax[2].set_ylabel(r'Energy score')
ax[2].set_xlim(0, 102)

plt.tight_layout()
plt.savefig(f"figures/l96_c{c:.0f}_combined_with_diag.png")
plt.close()
