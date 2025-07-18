import tensorflow as tf
import numpy as np
import pytest
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Assuming your MPSLayer is defined in src/model.py
from src.model import MPSLayer

@pytest.mark.parametrize("batch_size, num_sites, num_features, bond_dim, output_dim", [
    (1, 5, 8, 4, 2),
    (2, 10, 6, 6, 3),
    (4, 7, 5, 8, 4),
])
def test_mps_layer_output_shape(batch_size, num_sites, num_features, bond_dim, output_dim):
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
    assert logits.shape == (batch_size, output_dim), \
        f"Expected output shape ({batch_size}, {output_dim}), got {logits.shape}"


@pytest.mark.parametrize("num_sites, num_features, bond_dim, output_dim", [
    (5, 8, 4, 2),
    (10, 6, 6, 3),
    (7, 5, 8, 4),
])
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

    assert shapes == expected_shapes, \
        f"Weight shapes do not match. Expected {expected_shapes}, got {shapes}"


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

if __name__ == "__main__":
    pytest.main()
