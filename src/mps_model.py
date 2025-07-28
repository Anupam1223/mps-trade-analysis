import tensorflow as tf
import tensornetwork as tn
import os
import datetime
from tensorflow.keras import regularizers
# --- IMPROVEMENT: Import LayerNormalization ---
from tensorflow.keras.layers import (Dense, Input, Layer, TimeDistributed, Dropout, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from src.helper import quantile_loss

# Set TensorNetwork backend
tn.set_default_backend("tensorflow")


class MPSWeightLogger(TensorBoard):
    """
    A custom Keras callback that extends TensorBoard to also log the
    L2 norm of each MPS tensor at the end of each epoch.
    """
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        try:
            mps_layer = self.model.get_layer('mps_layer')
        except ValueError:
            return
        for i, tensor in enumerate(mps_layer.mps_tensors):
            tf.summary.scalar(f'mps_tensor_{i}_norm', tf.norm(tensor), step=epoch)


def get_mps_callbacks(log_dir_base="logs/fit"):
    """
    Creates a list of callbacks for monitoring MPS training.
    """
    log_dir = os.path.join(log_dir_base, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    mps_logger_callback = MPSWeightLogger(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    return [mps_logger_callback]


class MPSLayer(Layer):
    """
    Matrix Product State layer for time-series inputs.
    Includes Layer Normalization for improved training stability.
    """
    def __init__(self, output_dim=1, bond_dim=10, l2_lambda=1e-4, init_stddev=0.1, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.l2_lambda = l2_lambda
        self.init_stddev = init_stddev

    def build(self, input_shape):
        self.num_sites = input_shape[1]
        self.num_features = input_shape[2]
        reg = regularizers.l2(self.l2_lambda)
        
        # --- IMPROVEMENT: Use Orthogonal initialization for core tensors for stability ---
        self.mps_tensors = []
        for i in range(self.num_sites):
            if i == 0:
                shape = (self.num_features, self.bond_dim)
                init = tf.keras.initializers.RandomNormal(stddev=self.init_stddev)
            elif i == self.num_sites - 1:
                shape = (self.bond_dim, self.num_features, self.output_dim)
                init = tf.keras.initializers.RandomNormal(stddev=self.init_stddev)
            else:
                shape = (self.bond_dim, self.num_features, self.bond_dim)
                init = tf.keras.initializers.Orthogonal(gain=1.0) # Orthogonal for core
            
            tensor = self.add_weight(shape=shape, initializer=init, regularizer=reg, name=f"mps_{i}")
            self.mps_tensors.append(tensor)
        
        self.bias = self.add_weight(shape=(self.output_dim,), initializer="zeros", name="output_bias")
        
        # --- IMPROVEMENT: Add LayerNormalization for the state vector ---
        self.layer_norm = LayerNormalization()
        
        super(MPSLayer, self).build(input_shape)

    def call(self, inputs):
        @tf.function
        def contract_sample(sample):
            contracted_vec = tn.ncon([self.mps_tensors[0], sample[0]], [[1, -1], [1]])
            
            for i in range(1, self.num_sites - 1):
                temp = tn.ncon([contracted_vec, self.mps_tensors[i]], [[1], [1, -1, -2]])
                contracted_vec = tn.ncon([temp, sample[i]], [[1, -1], [1]])
                # --- IMPROVEMENT: Apply Layer Normalization to the state vector ---
                contracted_vec = self.layer_norm(contracted_vec)

            final_tensor = tn.ncon([contracted_vec, self.mps_tensors[-1]], [[1], [1, -1, -2]])
            output = tn.ncon([final_tensor, sample[-1]], [[1, -1], [1]])
            return output

        logits = tf.vectorized_map(contract_sample, inputs)
        logits += self.bias
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_quantile_mps_model(
    input_shape,
    quantile,
    bond_dim=10,
    learning_rate=1e-4,
    l2_lambda=1e-4,
    clipnorm=1.0,
    embedding_dim=16
):
    """ 
    Builds a quantile regression model using the stabilized MPSLayer.
    """
    inputs = Input(shape=input_shape)
    embedded_inputs = TimeDistributed(Dense(embedding_dim, activation="tanh"), name="feature_embedding")(inputs)
    mps_output = MPSLayer(
        output_dim=1, bond_dim=bond_dim, l2_lambda=l2_lambda, name="mps_layer"
    )(embedded_inputs)
    x = Dropout(0.1, name="dropout_after_mps")(mps_output)
    x = Dense(32, activation="relu", name="post_mps_dense1")(x)
    x = Dropout(0.1, name="dropout_dense1")(x)
    x = Dense(16, activation="relu", name="post_mps_dense2")(x)
    output = Dense(1, activation="linear", name="output_dense")(x)
    model = Model(inputs=inputs, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    
    model.compile(
        optimizer=optimizer,
        loss=quantile_loss(quantile),
        metrics=["mean_absolute_error"],
    )
    return model
