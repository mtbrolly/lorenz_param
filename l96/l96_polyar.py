import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from pathlib import Path
plt.style.use('./paper.mplstyle')
plt.ioff()

c = 4
t_p = 100

MODEL_DIR = f"models/l96_polyar_c{c:.0f}_tp{t_p:.0f}/"

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)

data = np.load(f"data/l96/climate_rk4_c{c:.0f}_without_y.npz")

X_n = data["X"][:-1]
M_n = data["M"][:]


# Discard a burn-in period.
X_n = X_n[10000:]
M_n = M_n[10000:]


# --- BUILD MODEL ---


# Fit polynomial.
sample_rate = 100
poly_c = np.polynomial.polynomial.polyfit(
    X_n[::sample_rate].flatten(), M_n[::sample_rate].flatten(), deg=3
)
np.save(MODEL_DIR + "poly_coeff.npy", poly_c)


@jit(nopython=True, cache=True)
def polyval(x, p):
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[-(i + 1)]
    return y


plt.figure()
plt.scatter(X_n[::sample_rate].flatten(),
            -1e3 * M_n[::sample_rate].flatten(), s=1.,
            marker='o', color='w', edgecolors='grey', linewidths=0.25)
x = np.linspace(X_n.min(), X_n.max(), 1000)
plt.plot(x, -1e3 * polyval(x, poly_c), "k", zorder=100)
plt.plot(X_n[:5000, 0], -1e3 * M_n[:5000, 0], 'g', lw=0.5)
# plt.plot(
#     x, -0.00235 * x**3 - 0.0136 * x**2 + 1.3 * x + 0.341, "r", zorder=101
# )
# plt.plot(
#     x, -0.000223 * x**3 - 0.00550 * x**2 + 0.575 * x + 0.198, "r", zorder=101
# )
plt.xlabel(r'$X$')
plt.ylabel(r'$M$')
plt.tight_layout()
plt.savefig(MODEL_DIR + "M_det.png")
plt.close()


# Fit AR(1) noise model.

M_poly = polyval(X_n, poly_c)
M_res = M_poly - M_n

M_res_phi = np.mean(M_res[:-t_p] * M_res[t_p:]) / np.var(M_res)
M_res_sigma = (np.var(M_res) * (1 - M_res_phi ** 2)) ** 0.5

# Short range autocorrelation
a_res = [
    np.var(M_res),
]

max_n = 10 * t_p
skip = t_p
tns = np.arange(0, max_n + 1, skip)
for i in range(1, len(tns)):
    a_res.append(np.mean(M_res[:-tns[i]] * M_res[tns[i]:]))
a_res = np.array(a_res) / np.var(M_res)

dt = 1e-3

plt.figure()
ts = tns * dt
plt.plot(ts, a_res, 'kv-')
a_ar1 = M_res_phi ** (tns // t_p)
plt.plot(ts, a_ar1, '--')
plt.xlabel(r'$\delta t$')
plt.ylabel(r'Temporal autocorrelation of $M_{\mathrm{res}}$')
plt.xlim(0, ts.max())
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR + "M_res_autocorrelation_short.png")
plt.close()

# Longer range autocorrelation
a_res = [
    np.var(M_res),
]

max_n = 1000
skip = t_p
tns = np.arange(0, max_n + 1, skip)
for i in range(1, len(tns)):
    a_res.append(np.mean(M_res[:-tns[i]] * M_res[tns[i]:]))
a_res = np.array(a_res) / np.var(M_res)

dt = 1e-3

plt.figure()
ts = tns * dt
plt.plot(ts, a_res, 'k-')
a_ar1 = M_res_phi ** (tns // t_p)
plt.plot(ts, a_ar1, '--')
plt.xlabel(r'$\delta t$')
plt.ylabel(r'Temporal autocorrelation of $M_{\mathrm{res}}$')
plt.xlim(0, ts.max())
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR + "M_res_autocorrelation.png")
plt.close()


# Plot time series of residual.
plt.figure()
plt.plot(np.arange(5000) * dt, M_res[:5000, 0], 'k')
plt.xlim(0, None)
plt.xlabel(r'$t$')
plt.ylabel(r'$M_{\mathrm{res}}(t)$')
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR + "M_res_series.png")


# Save experiment parameters
experiment = {
    "c": c,
    "t_p": t_p,
    "sample_rate": sample_rate,
    "M_res_phi": M_res_phi,
    "M_res_sigma": M_res_sigma
}

with open(MODEL_DIR + r"experiment.pkl", "wb") as file:
    pickle.dump(experiment, file)

with open(MODEL_DIR + "experiment.csv", "w") as f:
    w = csv.writer(f)
    w.writerows(experiment.items())
