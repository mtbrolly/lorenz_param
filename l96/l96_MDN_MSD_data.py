"""
Simulate Lorenz '96 with MDN Markovian parameterisation.
"""

import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from time import time
from l96_compat import integrate_reduced_l96_tf
from utils import load_multivariate_MDN, load_independent_MDN

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
multiple_gpu = len(tf.config.list_physical_devices('GPU')) > 1

tf.random.set_seed(1)

locality = 'local'
c = 4
tp = 100
print((locality, f"c = {c:.0f}", f"tp = {tp:.0f}"))

param_type = 'MSD'

MODEL_NAME = "l96_MDN_" + locality + "_" + \
    param_type + f"_c{c:.0f}_nc32_npl128"

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


# Load climate data.

data = np.load(f"data/l96/climate_rk4_c{c:.0f}_without_y.npz")

X_n = data['X'][:-1]
M_n = data['M'][:]


# Load parameterisation.

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
            output_size=X_n.shape[-1],
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
            return Yscaler.invert_standardisation(model(
                Xscaler.standardise(X_n[..., None])).sample()[..., 0])
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
            output_size=X_n.shape[-1],
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
            return Yscaler.invert_standardisation(
                model(Xscaler.standardise(
                    tf.concat((X_n[..., None], M_nm1[..., None]), axis=-1))
                ).sample()[..., 0])
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


# Run simulation.

burnin = int(1e5)
X0 = X_n[0 + burnin]
L_realisations = int(1e4)  # 7)

T0 = time()
X_NN, M_NN = integrate_reduced_l96_tf(
    tf.constant(X0[None, :]), 1e-3, L_realisations, eval_param,
    param_type='MSD', t_p=tp)
T0 = time() - T0


np.save(MODEL_TP_DIR + "X.npy",
        X_NN.numpy().squeeze()[:-1])
np.save(MODEL_TP_DIR + "M.npy",
        M_NN.numpy().squeeze())
