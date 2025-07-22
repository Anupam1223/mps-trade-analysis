# src/mps_model.py
import tensorflow as tf
import tensornetwork as tn
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Input, Layer
from tensorflow.keras.models import Model

from .quantile_loss import quantile_loss

# Set TensorNetwork backend
tn.set_default_backend("tensorflow")


class MPSLayer(Layer):
    """
    Matrix Product State (MPS) layer for time-series inputs.

    This layer implements a Matrix Product State, a tensor network model
    often used in quantum physics, adapted here for machine learning. It's
    particularly effective at capturing long-range correlations in sequential
    data.

    Input shape: (batch_size, time_steps, num_features)
    Output shape: (batch_size, output_dim)
    """

    def __init__(self, output_dim=1, bond_dim=40, l2_lambda=1e-5, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim  # Internal dimension of the MPS chain
        self.l2_lambda = l2_lambda

    def build(self, input_shape):
        """Initializes the MPS tensors (weights) of the layer."""
        num_sites = input_shape[1]  # Corresponds to time_steps
        self.num_features = input_shape[2]
        reg = regularizers.l2(self.l2_lambda)
        init = tf.keras.initializers.RandomNormal(stddev=0.01)

        # The 'label_site' is where the final output is extracted from the MPS chain.
        # Placing it in the middle is a common and effective choice.
        self.label_site = num_sites // 2
        self.mps_tensors = []

        for i in range(num_sites):
            if i == 0:
                # First tensor in the chain
                shape = (self.num_features, self.bond_dim)
            elif i == self.label_site:
                # The central tensor, which includes the output dimension
                shape = (
                    self.bond_dim,
                    self.num_features,
                    self.bond_dim,
                    self.output_dim,
                )
            elif i == num_sites - 1:
                # Last tensor in the chain
                shape = (self.bond_dim, self.num_features)
            else:
                # Intermediate tensors (the "bulk" of the chain)
                shape = (self.bond_dim, self.num_features, self.bond_dim)

            self.mps_tensors.append(
                self.add_weight(
                    shape=shape, initializer=init, regularizer=reg, name=f"mps_tensor_{i}"
                )
            )
        super(MPSLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Defines the forward pass, which contracts the MPS with the input data.
        """
        inputs = tf.cast(inputs, tf.float32)

        def contract_sample(sample):
            """Contracts the MPS tensors with a single input sample."""
            num_sites = sample.shape[0]

            # Contract from the left end to the center (label_site)
            left = tn.ncon([self.mps_tensors[0], sample[0]], [[1, -1], [1]])
            for i in range(1, self.label_site):
                left = tn.ncon(
                    [left, self.mps_tensors[i], sample[i]], [[1], [1, 2, -1], [2]]
                )

            # Contract from the right end to the center
            right = tn.ncon([self.mps_tensors[-1], sample[-1]], [[-1, 1], [1]])
            for i in range(num_sites - 2, self.label_site, -1):
                right = tn.ncon(
                    [right, self.mps_tensors[i], sample[i]], [[1], [-1, 2, 1], [2]]
                )

            # Final contraction at the center to get the output
            output = tn.ncon(
                [
                    left,
                    self.mps_tensors[self.label_site],
                    right,
                    sample[self.label_site],
                ],
                [[1], [1, 2, 3, -1], [3], [2]],
            )
            return output

        # Use vectorized_map for efficient batch processing
        logits = tf.vectorized_map(contract_sample, inputs)
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_quantile_mps_model(
    input_shape, quantile, bond_dim=40, learning_rate=1e-4, l2_lambda=1e-5
):
    """
    Builds a quantile regression model using the MPSLayer.
    """
    inputs = Input(shape=input_shape)

    mps_output = MPSLayer(
        output_dim=1, bond_dim=bond_dim, l2_lambda=l2_lambda, name="mps_layer"
    )(inputs)

    # --- KEY CHANGE ---
    # Add a BatchNormalization layer to stabilize the output of the MPSLayer.
    # The complex tensor contractions in the MPS can lead to outputs with
    # high variance or unstable scaling, which hinders training.
    # BatchNormalization rescales and centers the output, creating a more
    # stable learning environment for the optimizer.
    output = BatchNormalization(name="batch_norm")(mps_output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=quantile_loss(quantile),
        metrics=["mean_absolute_error"],
    )

    return model
