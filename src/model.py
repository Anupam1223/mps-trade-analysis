import tensorflow as tf
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensornetwork as tn

# Set the backend for TensorNetwork
tn.set_default_backend("tensorflow")

class MPSLayer(Layer):
    """
    An MPS layer for classification using ncon for the contraction.

    This layer implements the methodology from arXiv:1906.06329, where
    the inner product between encoded data and a trainable MPS is computed
    using tensornetwork.ncon.
    """
    def __init__(self, output_dim, bond_dim=8, l2_lambda=1e-4, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.output_dim = output_dim  # Number of classes
        self.bond_dim = bond_dim
        self.l2_lambda = l2_lambda

    def build(self, input_shape):
        num_sites = input_shape[1]
        self.feature_map_dim = 2 # d=2 for the feature map Φ(p)=[1-p, p]
        reg = regularizers.l2(self.l2_lambda)
        
        self.label_site = num_sites // 2

        self.mps_tensors = []
        for i in range(num_sites):
            if i == 0:
                shape = (self.feature_map_dim, self.bond_dim)
            elif i == self.label_site:
                shape = (self.bond_dim, self.feature_map_dim, self.bond_dim, self.output_dim)
            elif i == num_sites - 1:
                shape = (self.bond_dim, self.feature_map_dim)
            else:
                shape = (self.bond_dim, self.feature_map_dim, self.bond_dim)
            
            self.mps_tensors.append(self.add_weight(
                shape=shape, initializer='glorot_uniform', regularizer=reg, name=f'mps_{i}'
            ))
        super(MPSLayer, self).build(input_shape)

    def _feature_map(self, inputs):
        """Applies the linear feature map Φ(p) = [1-p, p]."""
        return tf.stack([1.0 - inputs, inputs], axis=-1)

    def call(self, inputs):
        squeezed_inputs = tf.squeeze(inputs, axis=-1)
        data = self._feature_map(squeezed_inputs)
        
        def contract_one_sample(sample_data):
            """Defines the full network contraction for one sample using ncon."""
            num_sites = sample_data.shape[0]
            
            tensors_to_contract = []
            network_indices = []
            
            for i in range(num_sites):
                feature_idx = i + 1
                tensors_to_contract.append(sample_data[i])
                network_indices.append([feature_idx])
                
                tensors_to_contract.append(self.mps_tensors[i])
                if i == 0:
                    mps_indices = [feature_idx, 101]
                elif i == self.label_site:
                    mps_indices = [100 + i, feature_idx, 101 + i, -1]
                elif i == num_sites - 1:
                    mps_indices = [100 + i, feature_idx]
                else:
                    mps_indices = [100 + i, feature_idx, 101 + i]
                network_indices.append(mps_indices)
            
            return tn.ncon(tensors_to_contract, network_indices)

        logits = tf.map_fn(contract_one_sample, data, fn_output_signature=tf.float32)
        return logits

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)

# ----------------------------------------------------------------------------------

def build_model(input_shape, num_classes, bond_dim=8):
    """
    Builds the Keras model using the data-encoding MPSLayer.
    """
    inputs = Input(shape=input_shape)
    outputs = MPSLayer(output_dim=num_classes, bond_dim=bond_dim)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    
    # --- THIS IS THE FIX ---
    # Use SparseCategoricalCrossentropy because the labels are integers (0, 1),
    # not one-hot encoded vectors. The metric is also updated accordingly.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    return model
