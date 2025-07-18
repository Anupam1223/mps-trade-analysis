import tensorflow as tf
import tensornetwork as tn
import numpy as np
from tqdm import trange
import time

# Set the backend for TensorNetwork
tn.set_default_backend("tensorflow")


@tf.function
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


def _dmrg_update_step(mps_tensors, i, bond_dim, sweep_direction, label_site):
    """
    Performs the two-site update using SVD, handling all edge cases including
    dynamic bond dimension and the special label tensor shape.
    """
    T1 = mps_tensors[i]
    T2 = mps_tensors[i+1]

    # Temporarily permute the label tensor to move the bond to the last axis
    is_label_tensor = (len(T1.shape) == 4) and (i == label_site)
    if is_label_tensor:
        # Permute from (left_bond, feat, right_bond, out) to (left_bond, feat, out, right_bond)
        T1 = tf.transpose(T1, perm=[0, 1, 3, 2])

    # Contract the two tensors to form the bond tensor
    bond_axis_T1 = len(T1.shape) - 1
    bond_axis_T2 = 0
    bond_tensor = tf.tensordot(T1, T2, axes=[[bond_axis_T1], [bond_axis_T2]])

    # Reshape into a matrix for SVD
    left_dims = T1.shape[:-1]
    right_dims = T2.shape[1:]
    left_size = np.prod(left_dims).item()
    right_size = np.prod(right_dims).item()
    matrix_to_decompose = tf.reshape(bond_tensor, (left_size, right_size))
    
    # Perform SVD
    s, u, v = tf.linalg.svd(matrix_to_decompose)

    # Dynamic truncation and padding to prevent reshape errors
    num_singular_values = s.shape[0]
    new_dim = min(num_singular_values, bond_dim)

    # Truncate using the new, safe dimension.
    u_trunc = u[:, :new_dim]
    s_trunc = tf.linalg.diag(s[:new_dim])
    vh_trunc = tf.transpose(v[:, :new_dim]) 

    # Distribute singular values based on sweep direction
    if sweep_direction == 'forward':
        new_T1_matrix_trunc = u_trunc
        new_T2_matrix_trunc = s_trunc @ vh_trunc
    else: # backward
        new_T1_matrix_trunc = u_trunc @ s_trunc
        new_T2_matrix_trunc = vh_trunc

    # Pad the truncated matrices with zeros to restore them to the full bond_dim.
    padding_amount = bond_dim - new_dim
    new_T1_matrix = tf.pad(new_T1_matrix_trunc, [[0, 0], [0, padding_amount]])
    new_T2_matrix = tf.pad(new_T2_matrix_trunc, [[0, padding_amount], [0, 0]])
    
    # Reshape the padded matrices back into tensors
    new_T1_shape = (*left_dims, bond_dim)
    new_T2_shape = (bond_dim, *right_dims)
    new_T1 = tf.reshape(new_T1_matrix, new_T1_shape)
    new_T2 = tf.reshape(new_T2_matrix, new_T2_shape)

    # If T1 was the label tensor, permute it back to its original dimension order
    if is_label_tensor:
        new_T1 = tf.transpose(new_T1, perm=[0, 1, 3, 2])

    # Update the tensors in the list by assigning the new values in-place
    mps_tensors[i].assign(new_T1)
    mps_tensors[i+1].assign(new_T2)


def train_with_dmrg(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    Trains the MPS model using a two-site DMRG-like optimization algorithm.
    """
    mps_layer = model.get_layer('mps_layer')
    num_sites = len(mps_layer.mps_tensors)
    bond_dim = mps_layer.bond_dim
    loss_fn = model.loss
    
    # Get learning rate as a standard Python float.
    learning_rate = float(model.optimizer.learning_rate.numpy())

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    # The history object will be a plain dictionary, not a Keras History object.
    history = {'loss': [], 'sparse_categorical_accuracy': [], 'val_loss': [], 'val_sparse_categorical_accuracy': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()
        
        # --- FORWARD SWEEP ---
        print("-> Forward Sweep")
        for i in trange(num_sites - 1, desc="Optimizing bond"):
            bond_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            tensors_to_optimize = [mps_layer.mps_tensors[i], mps_layer.mps_tensors[i+1]]
            
            train_bond_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
            for X_batch, y_batch in train_bond_dataset:
                with tf.GradientTape() as tape:
                    feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
                    logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
                    loss = loss_fn(y_batch, logits)
                
                grads = tape.gradient(loss, tensors_to_optimize)
                clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
                bond_optimizer.apply_gradients(zip(clipped_grads, tensors_to_optimize))
            
            _dmrg_update_step(mps_layer.mps_tensors, i, bond_dim, 'forward', mps_layer.label_site)

        # --- BACKWARD SWEEP ---
        print("<- Backward Sweep")
        for i in trange(num_sites - 2, -1, -1, desc="Optimizing bond"):
            bond_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            tensors_to_optimize = [mps_layer.mps_tensors[i], mps_layer.mps_tensors[i+1]]
            
            train_bond_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
            for X_batch, y_batch in train_bond_dataset:
                with tf.GradientTape() as tape:
                    feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
                    logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
                    loss = loss_fn(y_batch, logits)
                
                grads = tape.gradient(loss, tensors_to_optimize)
                clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
                bond_optimizer.apply_gradients(zip(clipped_grads, tensors_to_optimize))

            _dmrg_update_step(mps_layer.mps_tensors, i, bond_dim, 'backward', mps_layer.label_site)

        # --- MANUAL EVALUATION at the end of each epoch ---
        # Evaluate on Training Data
        train_losses = []
        train_accuracies = []
        train_dataset_eval = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        for X_batch, y_batch in train_dataset_eval:
            feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
            logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
            loss = loss_fn(y_batch, logits)
            train_losses.append(loss.numpy())
            
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            y_batch_int = tf.cast(y_batch, dtype=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_batch_int), dtype=tf.float32))
            train_accuracies.append(acc.numpy())

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accuracies)
        
        # Evaluate on Validation Data
        val_losses = []
        val_accuracies = []
        for X_batch, y_batch in test_dataset:
            feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
            logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
            loss = loss_fn(y_batch, logits)
            val_losses.append(loss.numpy())

            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            y_batch_int = tf.cast(y_batch, dtype=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_batch_int), dtype=tf.float32))
            val_accuracies.append(acc.numpy())

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accuracies)
        
        history['loss'].append(train_loss)
        history['sparse_categorical_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_sparse_categorical_accuracy'].append(val_acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Time: {epoch_time:.2f}s - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return history
