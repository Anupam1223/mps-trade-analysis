import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensornetwork as tn
import numpy as np

# Set the backend for TensorNetwork to TensorFlow
tn.set_default_backend("tensorflow")

class MPSLayer(Layer):
    """
    A Matrix Product State layer designed for multi-feature time-series classification.

    This layer now correctly processes input of shape (batch, lookback_period, num_features).
    The MPS learns correlations along the 'lookback_period' (time) dimension,
    with each tensor in the chain processing a vector of 'num_features'.
    """
    def __init__(self, output_dim, bond_dim=10, l2_lambda=1e-5, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.l2_lambda = l2_lambda

    def build(self, input_shape):
        """
        Build the layer's weights (the MPS tensors).
        The physical dimension of the tensors now corresponds to the number of features.
        """
        # input_shape: (batch_size, num_sites, num_features)
        num_sites = input_shape[1]
        self.num_features = input_shape[2]
        reg = regularizers.l2(self.l2_lambda)

        self.label_site = num_sites // 2

        self.mps_tensors = []
        for i in range(num_sites):
            if i == 0:
                # Shape: (num_features, bond_dim)
                shape = (self.num_features, self.bond_dim)
            elif i == self.label_site:
                # Shape: (bond_dim, num_features, bond_dim, output_dim)
                shape = (self.bond_dim, self.num_features, self.bond_dim, self.output_dim)
            elif i == num_sites - 1:
                # Shape: (bond_dim, num_features)
                shape = (self.bond_dim, self.num_features)
            else:
                # Shape: (bond_dim, num_features, bond_dim)
                shape = (self.bond_dim, self.num_features, self.bond_dim)

            self.mps_tensors.append(self.add_weight(
                shape=shape, initializer='glorot_uniform', regularizer=reg, name=f'mps_{i}'
            ))
        super(MPSLayer, self).build(input_shape)

    def call(self, inputs):
        """
        The forward pass performs the tensor network contraction.
        It directly uses the feature vectors without a feature map.
        """
        # --- FIX: The tf.squeeze line was removed from here. ---
        # The input data is now correctly handled as (batch, sites, features).
        data = tf.cast(inputs, tf.float32)

        def contract_one_sample(sample_data):
            """
            Defines the contraction for a single sample (a window of feature vectors).
            """
            num_sites = sample_data.shape[0]

            # --- Contract the left side ---
            left_vector = tn.ncon([self.mps_tensors[0], sample_data[0]], [[1, -1], [1]])
            for i in range(1, self.label_site):
                left_vector = tn.ncon(
                    [left_vector, self.mps_tensors[i], sample_data[i]],
                    [[1], [1, 2, -1], [2]]
                )

            # --- Contract the right side ---
            right_vector = tn.ncon([self.mps_tensors[-1], sample_data[-1]], [[-1, 1], [1]])
            for i in range(num_sites - 2, self.label_site, -1):
                right_vector = tn.ncon(
                    [right_vector, self.mps_tensors[i], sample_data[i]],
                    [[1], [-1, 2, 1], [2]]
                )

            # --- Final contraction at the label site ---
            final_result = tn.ncon(
                [left_vector, self.mps_tensors[self.label_site], right_vector, sample_data[self.label_site]],
                [[1], [1, 2, 3, -1], [3], [2]]
            )
            return final_result

        logits = tf.map_fn(contract_one_sample, data, fn_output_signature=tf.float32)
        return logits

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)

# ----------------------------------------------------------------------------------

def build_model(input_shape, num_classes, bond_dim=10, learning_rate=1e-4, l2_lambda=1e-5):
    """
    Builds the Keras model using the multi-feature MPSLayer.
    """
    inputs = Input(shape=input_shape)

    # The MPS layer produces the raw logits
    mps_output = MPSLayer(
        output_dim=num_classes,
        bond_dim=bond_dim,
        l2_lambda=l2_lambda,
        name='mps_layer'
    )(inputs)

    # A Softmax layer converts logits to probabilities.
    outputs = Softmax(name='softmax_output')(mps_output)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['sparse_categorical_accuracy']
    )
    return model
