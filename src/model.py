import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensornetwork as tn

# A custom Keras Layer for a simple MPS contraction
class MPSLayer(tf.keras.layers.Layer):
    def __init__(self, bond_dim, output_dim=1, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.bond_dim = bond_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        # input_shape is (batch_size, num_sites, feature_dim)
        num_sites = input_shape[1]
        feature_dim = input_shape[2]
        
        # Create the MPS tensors as trainable weights
        self.mps_tensors = []
        # First tensor
        self.mps_tensors.append(self.add_weight(shape=(feature_dim, self.bond_dim), initializer='glorot_uniform', name='mps_0'))
        # Middle tensors
        for i in range(1, num_sites - 1):
            self.mps_tensors.append(self.add_weight(shape=(self.bond_dim, feature_dim, self.bond_dim), initializer='glorot_uniform', name=f'mps_{i}'))
        # Last tensor
        self.mps_tensors.append(self.add_weight(shape=(self.bond_dim, feature_dim), initializer='glorot_uniform', name=f'mps_{num_sites-1}'))

    def call(self, inputs):
        # inputs is the batch of time series data
        batch_size = tf.shape(inputs)[0]
        
        # This function processes one sample at a time
        def contract_one_sample(sample):
            # sample shape: (num_sites, feature_dim)
            nodes = [tn.Node(t) for t in self.mps_tensors]
            
            # Contract input features into the MPS
            # Start with the first site
            result = tf.tensordot(sample[0], nodes[0].tensor, axes=[[0], [0]])
            
            # Contract the rest of the sites
            for i in range(1, len(nodes)):
                result = tf.tensordot(result, nodes[i].tensor, axes=[[-1], [0]])
                result = tf.tensordot(result, sample[i], axes=[[-1], [0]])

            return result

        # Use tf.map_fn to apply the contraction to each sample in the batch
        batch_results = tf.map_fn(contract_one_sample, inputs, dtype=tf.float32)
        return batch_results

def build_model(input_shape, bond_dim=8):
    """Builds and compiles the Keras model with the MPS layer."""
    inputs = Input(shape=input_shape)
    # The MPS layer contracts the time series down to a single feature vector
    mps_output = MPSLayer(bond_dim=bond_dim)(inputs)
    # A final dense layer for regression
    outputs = Dense(1)(mps_output)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"Model built with bond dimension {bond_dim}.")
    model.summary()
    return model