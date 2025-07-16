import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensornetwork as tn

# Although we are not using ncon in the final version, setting the backend is still
# good practice if you use any other tensornetwork features.
tn.set_default_backend("tensorflow")


def directional_accuracy(y_true, y_pred):
    """Calculates directional accuracy for a classification problem."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    correct_direction = tf.equal(tf.sign(y_true - 0.5), tf.sign(y_pred - 0.5))
    return tf.reduce_mean(tf.cast(correct_direction, tf.float32))

# ----------------------------------------------------------------------------------

class MPSLayer(tf.keras.layers.Layer):
    """
    A custom Keras Layer for a Matrix Product State (MPS) tensor network.
    """
    def __init__(self, bond_dim, output_dim=1, l2_lambda=1e-4, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.bond_dim = bond_dim
        self.output_dim = output_dim 
        self.l2_lambda = l2_lambda

    def build(self, input_shape):
        """
        Initializes the MPS tensor weights with robust logic for different chain lengths.
        """
        num_sites = input_shape[1]
        feature_dim = input_shape[2]
        reg = regularizers.l2(self.l2_lambda)

        self.mps_tensors = []
        if num_sites is None:
             raise ValueError("The number of sites (input_shape[1]) cannot be None.")

        if num_sites == 1:
            self.mps_tensors.append(
                self.add_weight(shape=(feature_dim, self.output_dim),
                                initializer='glorot_uniform', regularizer=reg, name='mps_0'))
        else:
            # First tensor
            self.mps_tensors.append(
                self.add_weight(shape=(feature_dim, self.bond_dim),
                                initializer='glorot_uniform', regularizer=reg, name='mps_0'))
            # Middle tensors
            for i in range(1, num_sites - 1):
                self.mps_tensors.append(
                    self.add_weight(shape=(self.bond_dim, feature_dim, self.bond_dim),
                                    initializer='glorot_uniform', regularizer=reg, name=f'mps_{i}'))
            # Last tensor
            self.mps_tensors.append(
                self.add_weight(shape=(self.bond_dim, feature_dim),
                                initializer='glorot_uniform', regularizer=reg, name=f'mps_{num_sites-1}'))
        
        super(MPSLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Performs the forward pass by contracting the MPS with the input using tf.tensordot.
        """
        def contract_one_sample(sample):
            # Contract first feature vector with the first MPS tensor
            # sample[0] shape: (feature_dim,)
            # mps_tensors[0] shape: (feature_dim, bond_dim)
            result = tf.tensordot(sample[0], self.mps_tensors[0], axes=[[0], [0]])

            # Sequentially contract the remaining tensors
            num_sites = sample.shape[0]
            for i in range(1, num_sites):
                feature_vec = sample[i] # Shape: (feature_dim,)
                mps_tensor = self.mps_tensors[i] # Shape: (bond_dim, feature_dim, ...)
                
                # Contract the feature vector with its corresponding MPS tensor
                # This creates an operator for this site
                # For middle tensors: shape (bond, feat, bond) -> (bond, bond)
                # For the last tensor: shape (bond, feat) -> (bond,)
                op_i = tf.tensordot(feature_vec, mps_tensor, axes=[[0], [1]])
                
                # Contract the result from the previous sites with the new operator
                result = tf.tensordot(result, op_i, axes=1)
            
            return result

        batch_results = tf.map_fn(
            contract_one_sample, inputs, fn_output_signature=tf.float32
        )
        # The final result is a scalar for each sample in the batch.
        # We expand its dimension to be compatible with the next Dense layer.
        return tf.expand_dims(batch_results, axis=-1)

    def compute_output_shape(self, input_shape):
        """Explicitly defines the output shape for Keras."""
        batch_size = input_shape[0]
        return (batch_size, 1)

# ----------------------------------------------------------------------------------

def build_model(input_shape, bond_dim=50, dropout_rate=0.3):
    """Builds and compiles the Keras model."""
    inputs = Input(shape=input_shape)
    
    x = MPSLayer(bond_dim=bond_dim)(inputs)
    
    x = BatchNormalization()(x)
    x = Dense(16)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', directional_accuracy]
    )
    return model