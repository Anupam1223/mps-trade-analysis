import tensorflow as tf
import tensornetwork as tn
import numpy as np
from tqdm import trange
import time

# Set the backend for TensorNetwork
tn.set_default_backend("tensorflow")

@tf.function(jit_compile=True)
def contract_network_for_batch(feature_vectors, mps_tensors, label_site):
    """
    Contracts the MPS network with a batch of feature vectors.
    This function is JIT-compiled for performance.

    Args:
        feature_vectors: A batch of data of shape (batch_size, num_sites, feature_map_dim).
        mps_tensors: The list of MPS core tensors.
        label_site: The index of the site where the output label is attached.

    Returns:
        The logits for the batch.
    """
    num_sites = len(mps_tensors)

    def contract_one_sample(elements):
        """Defines the full network contraction for one sample using ncon."""
        sample_data = elements
        
        tensors_to_contract = []
        network_indices = []
        
        # Build the ncon network string dynamically
        for i in range(num_sites):
            feature_idx = i + 1
            tensors_to_contract.append(sample_data[i])
            network_indices.append([feature_idx])
            
            tensors_to_contract.append(mps_tensors[i])
            
            # Define indices for each MPS tensor based on its position
            if i == 0:
                # Left-most tensor: [feature, right_bond]
                mps_indices = [feature_idx, 101]
            elif i == label_site:
                # Label tensor: [left_bond, feature, right_bond, output_label]
                mps_indices = [100 + i, feature_idx, 101 + i, -1] # -1 is the open output index
            elif i == num_sites - 1:
                # Right-most tensor: [left_bond, feature]
                mps_indices = [100 + i, feature_idx]
            else:
                # Bulk tensor: [left_bond, feature, right_bond]
                mps_indices = [100 + i, feature_idx, 101 + i]
            network_indices.append(mps_indices)
        
        return tn.ncon(tensors_to_contract, network_indices)

    # Use tf.map_fn to apply the contraction over the entire batch
    logits = tf.map_fn(
        contract_one_sample,
        elems=feature_vectors,
        fn_output_signature=tf.float32
    )
    return logits

def train_with_dmrg(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    Trains the MPS model using a two-site DMRG-like optimization algorithm.

    Args:
        model: The Keras model containing the MPSLayer.
        X_train, y_train: Training data and labels.
        X_test, y_test: Validation data and labels.
        epochs: The number of full sweeps (epochs).
        batch_size: The batch size for stochastic updates.

    Returns:
        A dictionary containing the training history.
    """
    mps_layer = model.get_layer('mps_layer')
    num_sites = len(mps_layer.mps_tensors)
    bond_dim = mps_layer.bond_dim
    optimizer = model.optimizer
    loss_fn = model.loss
    
    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    
    history = {
        'loss': [], 'sparse_categorical_accuracy': [],
        'val_loss': [], 'val_sparse_categorical_accuracy': []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()
        
        # --- FORWARD SWEEP (left to right) ---
        print("-> Forward Sweep")
        for i in trange(num_sites - 1, desc="Optimizing bond"):
            for X_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    # The tape automatically watches the trainable MPS tensors
                    feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
                    logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
                    loss = loss_fn(y_batch, logits)
                
                # Compute and apply gradients ONLY for the two sites being optimized
                tensors_to_optimize = [mps_layer.mps_tensors[i], mps_layer.mps_tensors[i+1]]
                grads = tape.gradient(loss, tensors_to_optimize)
                optimizer.apply_gradients(zip(grads, tensors_to_optimize))

                # --- SVD Truncation Step (Renormalization) ---
                T1, T2 = mps_layer.mps_tensors[i], mps_layer.mps_tensors[i+1]
                
                # Combine the two tensors
                bond_tensor = tn.ncon([T1, T2], [(-1, -2, 1), (1, -3, -4)])
                
                # Reshape for SVD
                combined_shape = bond_tensor.shape
                matrix_to_decompose = tf.reshape(bond_tensor, (combined_shape[0] * combined_shape[1], combined_shape[2] * combined_shape[3]))
                
                # Perform SVD and truncate
                s, u, v = tf.linalg.svd(matrix_to_decompose)
                u_trunc = u[:, :bond_dim]
                s_trunc = tf.linalg.diag(s[:bond_dim])
                v_trunc = v[:, :bond_dim] # v is already transposed in tf.linalg.svd output
                
                # Split back into two tensors, absorbing s into the right tensor for a forward sweep
                new_T1 = tf.reshape(u_trunc, (combined_shape[0], combined_shape[1], -1))
                new_T2_unshaped = s_trunc @ tf.transpose(v_trunc)
                new_T2 = tf.reshape(new_T2_unshaped, (-1, combined_shape[2], combined_shape[3]))

                # Update the model's weights
                mps_layer.mps_tensors[i].assign(new_T1)
                mps_layer.mps_tensors[i+1].assign(new_T2)


        # --- BACKWARD SWEEP (right to left) ---
        print("<- Backward Sweep")
        for i in trange(num_sites - 2, -1, -1, desc="Optimizing bond"):
            for X_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
                    logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
                    loss = loss_fn(y_batch, logits)
                
                tensors_to_optimize = [mps_layer.mps_tensors[i], mps_layer.mps_tensors[i+1]]
                grads = tape.gradient(loss, tensors_to_optimize)
                optimizer.apply_gradients(zip(grads, tensors_to_optimize))

                # --- SVD Truncation Step ---
                T1, T2 = mps_layer.mps_tensors[i], mps_layer.mps_tensors[i+1]
                bond_tensor = tn.ncon([T1, T2], [(-1, -2, 1), (1, -3, -4)])
                combined_shape = bond_tensor.shape
                matrix_to_decompose = tf.reshape(bond_tensor, (combined_shape[0] * combined_shape[1], combined_shape[2] * combined_shape[3]))
                
                s, u, v = tf.linalg.svd(matrix_to_decompose)
                u_trunc = u[:, :bond_dim]
                s_trunc = tf.linalg.diag(s[:bond_dim])
                v_trunc = v[:, :bond_dim]
                
                # Absorb s into the left tensor for a backward sweep
                new_T1_unshaped = u_trunc @ s_trunc
                new_T1 = tf.reshape(new_T1_unshaped, (combined_shape[0], combined_shape[1], -1))
                new_T2 = tf.reshape(tf.transpose(v_trunc), (-1, combined_shape[2], combined_shape[3]))

                mps_layer.mps_tensors[i].assign(new_T1)
                mps_layer.mps_tensors[i+1].assign(new_T2)

        # --- EVALUATION at the end of each epoch ---
        train_loss, train_acc = model.evaluate(train_dataset, verbose=0)
        val_loss, val_acc = model.evaluate(test_dataset, verbose=0)
        
        history['loss'].append(train_loss)
        history['sparse_categorical_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_sparse_categorical_accuracy'].append(val_acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Time: {epoch_time:.2f}s - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return history
