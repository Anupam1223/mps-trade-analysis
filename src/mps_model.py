# src/mps_model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense  # Removed unused Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensornetwork as tn
import numpy as np
from .quantile_loss import quantile_loss

# Set TensorNetwork backend
tn.set_default_backend("tensorflow")


class MPSLayer(Layer):
    """
    Matrix Product State layer for time-series inputs.
    Expects input shape: (batch_size, time_steps, num_features)
    """
    def __init__(self, output_dim=1, bond_dim=8, l2_lambda=1e-5, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.l2_lambda = l2_lambda

    def build(self, input_shape):
        num_sites = input_shape[1]
        self.num_features = input_shape[2]
        reg = regularizers.l2(self.l2_lambda)
        init = tf.keras.initializers.RandomNormal(stddev=1e-2)  # Small init for stable contractions

        self.label_site = num_sites // 2
        self.mps_tensors = []

        for i in range(num_sites):
            if i == 0:
                shape = (self.num_features, self.bond_dim)
            elif i == self.label_site:
                shape = (self.bond_dim, self.num_features, self.bond_dim, self.output_dim)
            elif i == num_sites - 1:
                shape = (self.bond_dim, self.num_features)
            else:
                shape = (self.bond_dim, self.num_features, self.bond_dim)
            self.mps_tensors.append(self.add_weight(
                shape=shape,
                initializer=init,
                regularizer=reg,
                name=f'mps_{i}'
            ))
        super(MPSLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        def contract_sample(sample):
            num_sites = sample.shape[0]
            left = tn.ncon([self.mps_tensors[0], sample[0]], [[1, -1], [1]])

            for i in range(1, self.label_site):
                left = tn.ncon(
                    [left, self.mps_tensors[i], sample[i]],
                    [[1], [1, 2, -1], [2]]
                )

            right = tn.ncon([self.mps_tensors[-1], sample[-1]], [[-1, 1], [1]])

            for i in range(num_sites - 2, self.label_site, -1):
                right = tn.ncon(
                    [right, self.mps_tensors[i], sample[i]],
                    [[1], [-1, 2, 1], [2]]
                )

            output = tn.ncon(
                [left, self.mps_tensors[self.label_site], right, sample[self.label_site]],
                [[1], [1, 2, 3, -1], [3], [2]]
            )

            return output

        # Vectorized contraction
        logits = tf.vectorized_map(contract_sample, inputs)
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_quantile_mps_model(input_shape, quantile, bond_dim=8, learning_rate=1e-4, l2_lambda=1e-5):
    """
    Builds a quantile regression model using MPSLayer.
    """
    inputs = Input(shape=input_shape)

    # MPS Layer
    mps_output = MPSLayer(
        output_dim=1,
        bond_dim=bond_dim,
        l2_lambda=l2_lambda,
        name="mps_layer"
    )(inputs)

    # Optional stabilization layer
    output = Dense(1, activation="linear", name="output_dense")(mps_output)

    # Optional: Add batch normalization (commented for now)
    # from tensorflow.keras.layers import BatchNormalization
    # output = BatchNormalization()(output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=quantile_loss(quantile),
        metrics=['mean_absolute_error']
    )

    return model
