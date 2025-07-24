# src/mps_model.py
import tensorflow as tf
import tensornetwork as tn
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dense, Input, Layer, TimeDistributed)
from tensorflow.keras.models import Model
from src.helper import quantile_loss

# Set TensorNetwork backend
tn.set_default_backend("tensorflow")


class MPSLayer(Layer):
    """
    Matrix Product State layer for time-series inputs.
    Performs a causal, sequential contraction.
    Expects input shape: (batch_size, time_steps, num_features)
    """

    def __init__(self, output_dim=1, bond_dim=8, l2_lambda=1e-4, init_stddev=0.1, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.l2_lambda = l2_lambda
        self.init_stddev = init_stddev

    def build(self, input_shape):
        self.num_sites = input_shape[1]      # time_steps
        self.num_features = input_shape[2]   # feature dimension

        reg = regularizers.l2(self.l2_lambda)
        init = tf.keras.initializers.RandomNormal(stddev=self.init_stddev)

        self.mps_tensors = []
        for i in range(self.num_sites):
            if i == 0:
                # First tensor contracts with input, outputs a bond_dim vector
                shape = (self.num_features, self.bond_dim)
            elif i == self.num_sites - 1:
                # Last tensor takes a bond_dim vector and input, produces the final output
                shape = (self.bond_dim, self.num_features, self.output_dim)
            else:
                # Core tensors update the bond_dim vector
                shape = (self.bond_dim, self.num_features, self.bond_dim)
            
            tensor = self.add_weight(
                shape=shape, initializer=init, regularizer=reg, name=f"mps_{i}"
            )
            self.mps_tensors.append(tensor)
        
        # A final bias term can be helpful
        self.bias = self.add_weight(shape=(self.output_dim,), initializer="zeros", name="output_bias")

        super(MPSLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, num_features)
        
        # This function processes a single sample from the batch
        def contract_sample(sample):
            # sample shape: (time_steps, num_features)
            
            # Start with the first tensor and first time step
            # M_0 [phys, bond_out] * x_0 [phys] -> v_0 [bond_out]
            contracted_vec = tn.ncon([self.mps_tensors[0], sample[0]], [[1, -1], [1]])

            # Sequentially contract through the middle of the chain
            for i in range(1, self.num_sites - 1):
                # v_{t-1} [bond_in] * M_t [bond_in, phys, bond_out] -> [phys, bond_out]
                temp = tn.ncon([contracted_vec, self.mps_tensors[i]], [[1], [1, -1, -2]])
                # [phys, bond_out] * x_t [phys] -> v_t [bond_out]
                contracted_vec = tn.ncon([temp, sample[i]], [[1, -1], [1]])

            # Contract with the final tensor to get the output
            # v_{n-2} [bond_in] * M_{n-1} [bond_in, phys, out] -> [phys, out]
            final_tensor = tn.ncon([contracted_vec, self.mps_tensors[-1]], [[1], [1, -1, -2]])
            # [phys, out] * x_{n-1} [phys] -> [out]
            output = tn.ncon([final_tensor, sample[-1]], [[1, -1], [1]])
            
            return output

        # Use tf.map_fn to apply the contraction to each sample in the batch
        logits = tf.map_fn(contract_sample, inputs)
        logits += self.bias # Apply bias
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_quantile_mps_model(
    input_shape, quantile, bond_dim=8, learning_rate=1e-4, l2_lambda=1e-4,
    add_dense_output=True, embedding_dim=16
):
    """ 
    Builds a quantile regression model using MPSLayer.
    """
    inputs = Input(shape=input_shape)

    # Embed features at each time step
    embedded_inputs = TimeDistributed(Dense(embedding_dim, activation="tanh"), name="feature_embedding")(inputs)

    # MPS Layer
    mps_output = MPSLayer(
        output_dim=1, bond_dim=bond_dim, l2_lambda=l2_lambda, name="mps_layer"
    )(embedded_inputs)

    # Post-contraction processing with nonlinear layers
    x = Dense(32, activation="relu", name="post_mps_dense1")(mps_output)
    x = Dense(16, activation="relu", name="post_mps_dense2")(x)
    output = Dense(1, activation="linear", name="output_dense")(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=quantile_loss(quantile),
        metrics=["mean_absolute_error"],
    )

    return model
