import tensorflow as tf
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensornetwork as tn
import numpy as np

# Set the backend for TensorNetwork
tn.set_default_backend("tensorflow")

class MPSLayer(Layer):
    """
    An MPS layer for classification using ncon for the contraction.
    This version is compatible with the DMRG optimizer.
    """
    def __init__(self, output_dim, bond_dim=20, l2_lambda=1e-4, feature_map_type='linear', **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.l2_lambda = l2_lambda
        self.feature_map_type = feature_map_type

    def build(self, input_shape):
        num_sites = input_shape[1]
        self.feature_map_dim = 2 
        reg = regularizers.l2(self.l2_lambda)
        
        # Place the label site in the middle of the chain
        self.label_site = num_sites // 2

        self.mps_tensors = []
        for i in range(num_sites):
            # Define shape for each tensor based on its position
            if i == 0:
                # Left-most tensor: [feature_dim, right_bond_dim]
                shape = (self.feature_map_dim, self.bond_dim)
            elif i == self.label_site:
                # Label tensor: [left_bond_dim, feature_dim, right_bond_dim, output_dim]
                shape = (self.bond_dim, self.feature_map_dim, self.bond_dim, self.output_dim)
            elif i == num_sites - 1:
                # Right-most tensor: [left_bond_dim, feature_dim]
                shape = (self.bond_dim, self.feature_map_dim)
            else:
                # Bulk tensor: [left_bond_dim, feature_dim, right_bond_dim]
                shape = (self.bond_dim, self.feature_map_dim, self.bond_dim)
            
            # Add each tensor as a trainable weight
            self.mps_tensors.append(self.add_weight(
                shape=shape, initializer='glorot_uniform', regularizer=reg, name=f'mps_{i}', trainable=True
            ))
        super(MPSLayer, self).build(input_shape)

    def _feature_map(self, inputs):
        """
        Applies a feature map to encode the input data into vectors.
        'linear': Φ(p) = [1-p, p]
        'angle':  Φ(p) = [cos(p * π/2), sin(p * π/2)]
        """
        if self.feature_map_type == 'angle':
            # Non-linear angle encoding feature map
            return tf.stack([tf.cos(inputs * np.pi / 2.0), tf.sin(inputs * np.pi / 2.0)], axis=-1)
        # Default to linear encoding
        return tf.stack([1.0 - inputs, inputs], axis=-1)

    def call(self, inputs):
        """
        Defines the forward pass for inference/evaluation.
        The DMRG training uses its own contraction function.
        """
        squeezed_inputs = tf.squeeze(inputs, axis=-1)
        feature_vectors = self._feature_map(squeezed_inputs)
        
        # The contraction logic is now centralized in the DMRG optimizer
        # to facilitate JIT compilation and manual gradient control.
        logits = contract_network_for_batch(feature_vectors, self.mps_tensors, self.label_site)
        return logits

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)

# ----------------------------------------------------------------------------------

def build_model(input_shape, num_classes, bond_dim=8, learning_rate=1e-3, l2_lambda=1e-4, feature_map_type='linear'):
    """
    Builds the Keras model using the data-encoding MPSLayer.
    """
    inputs = Input(shape=input_shape)
    # --- MODIFICATION: Added name='mps_layer' to easily access it later ---
    outputs = MPSLayer(
        output_dim=num_classes, 
        bond_dim=bond_dim, 
        l2_lambda=l2_lambda,
        feature_map_type=feature_map_type,
        name='mps_layer' # Important for the DMRG optimizer
    )(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    
    # The model still needs to be compiled to define the optimizer and loss function
    # that the DMRG training loop will use.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    return model
