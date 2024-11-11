import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # noqa
import tensorflow as tf
import tensorflow_probability as tfp
from contextlib import nullcontext
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors
tf.keras.backend.set_floatx("float64")


def load_multivariate_MDN(
        output_size,
        activation_fn='tanh',
        n_hidden_layers=4,
        n_neurons_per_layer=32,
        n_components=1,
        model_name=None,
        multiple_gpu=False):
    """
    Loads multivariate MDN model for Lorenz '96 parameterisation.
    """

    if multiple_gpu:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope() if multiple_gpu else nullcontext():
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(
                tfkl.Dense(n_neurons_per_layer,
                           activation=activation_fn))
        model = tf.keras.Sequential(
            hidden_layers
            + [tfkl.Dense(
                (1 + tfpl.MultivariateNormalTriL.params_size(
                    event_size=output_size)) * n_components,
                activation=None),
                tfpl.MixtureSameFamily(
                    n_components,
                    tfpl.MultivariateNormalTriL(output_size))])

        if model_name:
            model.load_weights(
                "models/" + model_name
                + "/trained/weights").expect_partial()
    return model


def load_multivariate_MDN_diag(
        output_size,
        activation_fn='tanh',
        n_hidden_layers=4,
        n_neurons_per_layer=32,
        n_components=1,
        model_name=None,
        multiple_gpu=False):
    """
    Loads multivariate MDN model with diagonal covariance strucutre for Lorenz
    '96 parameterisation.
    """

    if multiple_gpu:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope() if multiple_gpu else nullcontext():
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(
                tfkl.Dense(n_neurons_per_layer,
                           activation=activation_fn))
        params_size = (1 + output_size * 2) * n_components

        def make_dist_lambda(params): return tfd.MultivariateNormalDiag(
            loc=params[..., ::2],
            scale_diag=tf.exp(params[..., 1::2]))
        model = tf.keras.Sequential(
            hidden_layers
            + [tfkl.Dense(params_size, activation=None),
               tfpl.MixtureSameFamily(
                   n_components,
                   tfpl.DistributionLambda(
                       make_distribution_fn=make_dist_lambda))])

        if model_name:
            model.load_weights(
                "models/" + model_name
                + "/trained/weights").expect_partial()
    return model


def load_independent_MDN(
        activation_fn='tanh',
        n_hidden_layers=4,
        n_neurons_per_layer=32,
        n_components=1,
        model_name=None,
        multiple_gpu=False):
    """
    Loads MDN model for Lorenz '96 parameterisation, where components of the
    error are modelled as independent and identically distributed.
    """

    if multiple_gpu:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope() if multiple_gpu else nullcontext():
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(
                tfkl.Dense(n_neurons_per_layer,
                           activation=activation_fn))
        model = tf.keras.Sequential(
            hidden_layers
            + [tfkl.Dense(
                tfpl.MixtureNormal.params_size(n_components),
                activation=None),
                tfpl.MixtureNormal(n_components,
                                   event_shape=(1,))])

        if model_name:
            model.load_weights(
                "models/" + model_name
                + "/trained/weights").expect_partial()
    return model


def load_CVAE(
        x_dim, y_dim, latent_dim,
        activation_fn='tanh',
        n_hidden_layers=4,
        n_neurons_per_layer=32,
        model_name=None):

    concatenated_shape = (x_dim + y_dim, )
    encoded_dimension = latent_dim

    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(encoded_dimension, dtype="float64"), scale=1),
        reinterpreted_batch_ndims=1)

    def build_encoder(n_hidden_layers, n_neurons_per_layer, activation_fn):
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(
                tfkl.Dense(n_neurons_per_layer,
                           activation=activation_fn))
        encoder = tf.keras.Sequential(
            hidden_layers + [tfkl.Dense(
                tfpl.MultivariateNormalTriL.params_size(encoded_dimension),
                activation=None),
                tfpl.MultivariateNormalTriL(
                    encoded_dimension,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(
                        prior, weight=0.5))])
        return encoder

    encoder = build_encoder(
        n_hidden_layers, n_neurons_per_layer, activation_fn)

    def build_decoder(n_hidden_layers, n_neurons_per_layer, activation_fn):
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(
                tfkl.Dense(n_neurons_per_layer,
                           activation=activation_fn))
        decoder = tf.keras.Sequential(
            hidden_layers + [tfkl.Dense(
                tfpl.MultivariateNormalTriL.params_size(y_dim),
                activation=None),
                tfpl.MultivariateNormalTriL(y_dim)])
        return decoder

    decoder = build_decoder(
        n_hidden_layers, n_neurons_per_layer, activation_fn)

    inputs = tf.keras.Input(shape=concatenated_shape)
    encoded = encoder(inputs)
    input_x, _ = tf.split(inputs, 2, axis=-1)
    decoder_inputs = tf.concat((input_x, encoded), -1)
    decoded = decoder(decoder_inputs)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=decoded)

    if model_name:
        autoencoder.load_weights(
            "models/" + model_name
            + "/trained/weights").expect_partial()
    return autoencoder
