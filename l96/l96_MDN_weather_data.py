"""
Simulate Lorenz '96 with MDN Markovian parameterisation.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # noqa
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from time import time
from l96 import integrate_reduced_l96_tf_weather
from utils import load_multivariate_MDN, load_independent_MDN
import matplotlib.pyplot as plt
plt.ioff()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
multiple_gpu = len(tf.config.list_physical_devices('GPU')) > 1

tf.random.set_seed(1)

param_type = 'MSD'
tp = 1
MODEL_NAME = "l96_MDN_local_" + param_type + "_c10_nc32_npl128"


MODEL_DIR = "models/" + MODEL_NAME + "/"
MODEL_TP_DIR = MODEL_DIR + f"tp_{tp:.0f}/"
if not Path(MODEL_TP_DIR).exists():
    Path(MODEL_TP_DIR).mkdir(parents=True)

with open(MODEL_DIR + r"experiment.pkl", "rb") as file:
    experiment = pickle.load(file)

c = experiment['c']
local = experiment['local']
activation_fn = experiment['activation_fn']
n_hidden_layers = experiment['n_hidden_layers']
n_neurons_per_layer = experiment['n_neurons_per_layer']
n_components = experiment['n_components']

with open(MODEL_DIR + "Xscaler.pkl", "rb") as file:
    Xscaler = pickle.load(file)

with open(MODEL_DIR + "Yscaler.pkl", "rb") as file:
    Yscaler = pickle.load(file)


# ------------------------ Load weather data ----------------------------------

X_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_X_n.npy")[:, 1:]
M_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy")[:, 1:]
M_nm1_true = np.load(f"data/l96/weather_rk4_c{c:.0f}_M_n.npy")[:, :-1]
dt = 1e-3


# ------------------------ Load parameterisation ------------------------------

if param_type == 'MSD':
    if local:
        model = load_independent_MDN(
            activation_fn=activation_fn,
            n_hidden_layers=n_hidden_layers,
            n_neurons_per_layer=n_neurons_per_layer,
            n_components=n_components,
            model_name=MODEL_NAME,
            multiple_gpu=multiple_gpu
        )
    else:
        model = load_multivariate_MDN(
            output_size=X_true.shape[-1],
            activation_fn=activation_fn,
            n_hidden_layers=n_hidden_layers,
            n_neurons_per_layer=n_neurons_per_layer,
            n_components=n_components,
            model_name=MODEL_NAME,
            multiple_gpu=multiple_gpu
        )

    if local:
        @tf.function
        def eval_param(X_n):
            return tf.squeeze(
                Yscaler.invert_standardisation(model(
                    Xscaler.standardise(X_n[..., None])).sample()))
    else:
        @tf.function
        def eval_param(X_n):
            if len(X_n.shape) < 2:
                X_n = X_n[None, :]
            return Yscaler.invert_standardisation(model(
                Xscaler.standardise(X_n)).sample())

elif param_type == 'Markov':
    if local:
        model = load_independent_MDN(
            activation_fn=activation_fn,
            n_hidden_layers=n_hidden_layers,
            n_neurons_per_layer=n_neurons_per_layer,
            n_components=n_components,
            model_name=MODEL_NAME,
            multiple_gpu=multiple_gpu
        )
    else:
        model = load_multivariate_MDN(
            output_size=X_true.shape[-1],
            activation_fn=activation_fn,
            n_hidden_layers=n_hidden_layers,
            n_neurons_per_layer=n_neurons_per_layer,
            n_components=n_components,
            model_name=MODEL_NAME,
            multiple_gpu=multiple_gpu
        )

    if local:
        @tf.function
        def eval_param(X_n, M_nm1):
            return tf.squeeze(
                Yscaler.invert_standardisation(model(
                    Xscaler.standardise(
                        tf.concat((X_n[..., None], M_nm1[..., None]), axis=-1))
                ).sample()))
    else:
        @tf.function
        def eval_param(X_n, M_nm1):
            if len(X_n.shape) < 2:
                X_n = X_n[None, :]
                M_nm1 = M_nm1[None, :]
            return Yscaler.invert_standardisation(model(
                Xscaler.standardise(
                    tf.concat(
                        (X_n, M_nm1), axis=-1))).sample())


# -------------------------- Do ensemble simulations --------------------------

weather_indices = np.arange(0, 1000, 10)
N_weathers = len(weather_indices)

X_true = X_true[weather_indices]
M_true = M_true[weather_indices]
M_nm1_true = M_nm1_true[weather_indices]

X0 = X_true[:, 0]
M_m1 = M_nm1_true[:, 0]
X0_tf = tf.constant(X0)
M_m1_tf = tf.constant(M_m1)

N_realisations = 100  # // 2  # !!!
L_realisations = 3500

X0_vec = tf.tile(
    tf.reshape(X0_tf, (N_weathers, 1, -1)), (1, N_realisations, 1))
M_m1_vec = tf.tile(
    tf.reshape(M_m1_tf, (N_weathers, 1, -1)), (1, N_realisations, 1))

t0 = time()
X_NN, M_NN = integrate_reduced_l96_tf_weather(
    X0_vec, 1e-3, L_realisations, eval_param, param_type=param_type, t_p=tp)
T0 = time() - t0
print(f"Took {T0:.0f} seconds.")

np.save(MODEL_TP_DIR
        + f"X_weather_tp{tp:.0f}_N{N_realisations}_L{L_realisations}.npy",
        X_NN)
np.save(MODEL_TP_DIR
        + f"M_weather_tp{tp:.0f}_N{N_realisations}_L{L_realisations}.npy",
        M_NN)
