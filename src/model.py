import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensornetwork as tn
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout

# --- THIS IS THE FIX ---
# Explicitly set the backend to TensorFlow before creating any TN objects.
tn.set_default_backend("tensorflow")

def directional_accuracy(y_true, y_pred):
    """Calculates directional accuracy for a classification problem."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    correct_direction = tf.equal(tf.sign(y_true - 0.5), tf.sign(y_pred - 0.5))
    return tf.reduce_mean(tf.cast(correct_direction, tf.float32))



# A custom Keras Layer for a simple MPS contraction
class MPSLayer(tf.keras.layers.Layer):
    def __init__(self, bond_dim, output_dim=1, l2_lambda=1e-4, **kwargs):
        super(MPSLayer, self).__init__(**kwargs)
        self.bond_dim = bond_dim
        self.output_dim = output_dim
        self.l2_lambda = l2_lambda

    def build(self, input_shape):
        num_sites = input_shape[1]
        feature_dim = input_shape[2]
        reg = regularizers.l2(self.l2_lambda)

        self.mps_tensors = []
        self.mps_tensors.append(
            self.add_weight(shape=(feature_dim, self.bond_dim),
                            initializer='glorot_uniform',
                            regularizer=reg,
                            name='mps_0'))
        for i in range(1, num_sites - 1):
            self.mps_tensors.append(
                self.add_weight(shape=(self.bond_dim, feature_dim, self.bond_dim),
                                initializer='glorot_uniform',
                                regularizer=reg,
                                name=f'mps_{i}'))
        self.mps_tensors.append(
            self.add_weight(shape=(self.bond_dim, feature_dim),
                            initializer='glorot_uniform',
                            regularizer=reg,
                            name=f'mps_{num_sites-1}'))

    def call(self, inputs):
        def contract_one_sample(sample):
            result = tf.tensordot(sample[0], self.mps_tensors[0], axes=[[0], [0]])
            for i in range(1, len(self.mps_tensors)):
                mps_tensor = self.mps_tensors[i]
                feature_vec = sample[i]
                if len(mps_tensor.shape) == 3:
                    axes = [[1], [0]]
                else:
                    axes = [[1], [0]]
                op_i = tf.tensordot(mps_tensor, feature_vec, axes=axes)
                result = tf.tensordot(result, op_i, axes=[[-1], [0]])
            return result

        batch_results = tf.map_fn(
            contract_one_sample, inputs, fn_output_signature=tf.float32
        )
        return tf.expand_dims(batch_results, axis=-1)


def build_model(input_shape, bond_dim=8, dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    mps_output = MPSLayer(bond_dim=bond_dim)(inputs)
    mps_output = Dropout(dropout_rate)(mps_output)
    
    # Binary classification output with sigmoid
    outputs = Dense(1, activation='sigmoid')(mps_output)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', directional_accuracy]  # you can keep directional_accuracy as extra
    )
    return model
