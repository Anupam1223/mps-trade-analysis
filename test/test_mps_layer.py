import os
import sys

import numpy as np
import pytest
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Assuming your MPSLayer is defined in src/model.py
from src.mps_model import MPSLayer


@pytest.mark.parametrize(
    "batch_size, num_sites, num_features, bond_dim, output_dim",
    [
        (1, 5, 8, 4, 2),
        (2, 10, 6, 6, 3),
        (4, 7, 5, 8, 4),
    ],
)
def test_mps_layer_output_shape(
    batch_size, num_sites, num_features, bond_dim, output_dim
):
    """
    Test that the MPSLayer produces the correct output shape without errors.
    """
    # Create random input data
    input_data = np.random.rand(batch_size, num_sites, num_features).astype(np.float32)

    # Build a minimal model to wrap MPSLayer
    inputs = Input(shape=(num_sites, num_features))
    mps_output = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)(inputs)
    model = Model(inputs=inputs, outputs=mps_output)

    # Perform a forward pass
    logits = model.predict(input_data)

    # Check output shape: (batch_size, output_dim)
    assert logits.shape == (
        batch_size,
        output_dim,
    ), f"Expected output shape ({batch_size}, {output_dim}), got {logits.shape}"


@pytest.mark.parametrize(
    "num_sites, num_features, bond_dim, output_dim",
    [
        (5, 8, 4, 2),
        (10, 6, 6, 3),
        (7, 5, 8, 4),
    ],
)
def test_mps_layer_weights_shapes(num_sites, num_features, bond_dim, output_dim):
    """
    Test that the MPSLayer creates weights (tensors) of the expected shapes.
    """
    # Build the layer
    layer = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)
    # Build weights by providing an input shape
    layer.build((None, num_sites, num_features))

    # Extract the weight shapes
    shapes = [w.shape for w in layer.mps_tensors]

    # Manually compute expected shapes
    expected_shapes = []
    label_site = num_sites // 2
    for i in range(num_sites):
        if i == 0:
            expected_shapes.append((num_features, bond_dim))
        elif i == label_site:
            expected_shapes.append((bond_dim, num_features, bond_dim, output_dim))
        elif i == num_sites - 1:
            expected_shapes.append((bond_dim, num_features))
        else:
            expected_shapes.append((bond_dim, num_features, bond_dim))

    assert (
        shapes == expected_shapes
    ), f"Weight shapes do not match. Expected {expected_shapes}, got {shapes}"


def test_mps_layer_forward_consistency():
    """
    Test that repeated forward passes with the same input yield the same output (determinism).
    """
    batch_size, num_sites, num_features, bond_dim, output_dim = 3, 6, 7, 5, 2
    input_data = np.random.rand(batch_size, num_sites, num_features).astype(np.float32)

    # Wrap in a model
    inputs = Input(shape=(num_sites, num_features))
    mps_output = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)(inputs)
    model = Model(inputs=inputs, outputs=mps_output)

    # Perform two forward passes
    logits1 = model.predict(input_data)
    logits2 = model.predict(input_data)

    # Assert element-wise equality within tolerance
    np.testing.assert_allclose(logits1, logits2, rtol=1e-6, atol=1e-6)

def test_mps_layer_output_variance():
    batch_size, num_sites, num_features, bond_dim, output_dim = 10, 8, 5, 4, 1
    input_data = np.random.randn(batch_size, num_sites, num_features).astype(np.float32)

    inputs = Input(shape=(num_sites, num_features))
    mps_output = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)(inputs)
    model = Model(inputs=inputs, outputs=mps_output)

    logits = model.predict(input_data)

    # Check that outputs vary (i.e., not all the same)
    std = np.std(logits)
    assert std > 1e-4, f"Output variance too low: {std}, model may be degenerate"

def test_mps_layer_gradient_flow():
    batch_size, num_sites, num_features, bond_dim, output_dim = 5, 6, 7, 4, 1
    input_data = tf.random.normal((batch_size, num_sites, num_features))
    target_data = tf.random.normal((batch_size, output_dim))

    inputs = Input(shape=(num_sites, num_features))
    mps_output = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)(inputs)
    model = Model(inputs=inputs, outputs=mps_output)

    with tf.GradientTape() as tape:
        preds = model(input_data)
        loss = tf.reduce_mean(tf.square(preds - target_data))
    
    grads = tape.gradient(loss, model.trainable_weights)
    
    # Check that at least one gradient is not None or zero
    has_grad = any(g is not None and tf.reduce_sum(tf.abs(g)) > 0 for g in grads)
    assert has_grad, "No gradient flow detected through MPSLayer"

def test_mps_layer_learns_identity():


    # Identity mapping test: model should learn to return input sum
    batch_size, num_sites, num_features = 32, 6, 4
    bond_dim, output_dim = 8, 1

    X = np.random.rand(batch_size, num_sites, num_features).astype(np.float32)
    y = np.sum(X, axis=(1, 2), keepdims=True)  # Output is total sum

    inputs = Input(shape=(num_sites, num_features))
    mps_output = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)(inputs)
    model = Model(inputs=inputs, outputs=mps_output)

    model.compile(optimizer=Adam(1e-2), loss=MeanSquaredError())
    history = model.fit(X, y, epochs=100, verbose=0)

    final_loss = history.history['loss'][-1]
    assert final_loss < 1e-3, f"Model failed to learn simple sum function, loss={final_loss}"


if __name__ == "__main__":
    pytest.main()
