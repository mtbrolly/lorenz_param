"""
A multivariate mixture density network modelling M_n given X_n.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # noqa
import pickle
import csv
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path
from utils import load_multivariate_MDN_diag
from preprocessing import Scaler

tfkl = tf.keras.layers
tfpl = tfp.layers
cb = tf.keras.callbacks
tf.keras.backend.set_floatx("float64")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
multiple_gpu = len(tf.config.list_physical_devices('GPU')) > 1

c = 10.

local = False
activation_fn = 'tanh'
n_hidden_layers = 6
n_neurons_per_layer = 128
n_components = 32

MODEL_DIR = ("models/l96_MDN_nonlocal_diag_MSD_"
             + f"c{c:.0f}_nc{n_components:.0f}_npl{n_neurons_per_layer}/")

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)

data = np.load(f"data/l96/climate_rk4_c{c:.0f}_without_y.npz")

X_n = data['X'][:-1]
M_n = data['M'][:]

sample_rate = 10

X = X_n[::sample_rate]
Y = M_n[::sample_rate]

Xscaler = Scaler(X)
Yscaler = Scaler(Y)

with open(MODEL_DIR + r"Xscaler.pkl", "wb") as file:
    pickle.dump(Xscaler, file)

with open(MODEL_DIR + r"Yscaler.pkl", "wb") as file:
    pickle.dump(Yscaler, file)

X_ = Xscaler.standardise(X)
Y_ = Yscaler.standardise(Y)
del X, Y, X_n, M_n


# --- BUILD MODEL ---

model = load_multivariate_MDN_diag(output_size=Y_.shape[-1],
                                   activation_fn=activation_fn,
                                   n_hidden_layers=n_hidden_layers,
                                   n_neurons_per_layer=n_neurons_per_layer,
                                   n_components=n_components,
                                   multiple_gpu=multiple_gpu)


# --- TRAIN MODEL ---

LOG_FILE = "log.csv"
CHECKPOINT_FILE = "checkpoint_epoch_{epoch:02d}/weights"
TRAINED_FILE = "trained/weights"

# Training configuration


def nll(data_point, tf_distribution):
    """Negative log likelihood."""
    return -tf_distribution.log_prob(data_point)


LOSS = nll
BATCH_SIZE = 8192
LEARNING_RATE = 5e-4
EPOCHS = 100000
OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
VALIDATION_SPLIT = 0.2


# Callbacks
CSV_LOGGER = cb.CSVLogger(MODEL_DIR + LOG_FILE)
PATIENCE = 10
EARLY_STOPPING = cb.EarlyStopping(monitor="val_loss", patience=PATIENCE)
CALLBACKS = [CSV_LOGGER, EARLY_STOPPING]


# Check model coded properly
model(X_[:10])


# Model compilation and training
model.compile(loss=LOSS, optimizer=OPTIMISER)

History = model.fit(
    X_,
    Y_,
    epochs=EPOCHS,
    callbacks=CALLBACKS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=VALIDATION_SPLIT,
    verbose=1,  # !!!
)

model.save_weights(MODEL_DIR + TRAINED_FILE)

# Save experiment parameters
experiment = {
    'c': c,
    'local': local,
    'activation_fn': activation_fn,
    'n_hidden_layers': n_hidden_layers,
    'n_neurons_per_layer': n_neurons_per_layer,
    'n_components': n_components,
    'sample_rate': sample_rate,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'epochs': EPOCHS,
    'validation_split': VALIDATION_SPLIT,
    'patience': PATIENCE
}

with open(MODEL_DIR + r"experiment.pkl", "wb") as file:
    pickle.dump(experiment, file)

with open(MODEL_DIR + "experiment.csv", 'w') as f:
    w = csv.writer(f)
    w.writerows(experiment.items())
