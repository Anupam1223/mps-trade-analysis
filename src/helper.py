import tensorflow as tf

def svd_decompose_input(encoded_inputs, bond_dim):
    """
    Apply SVD to reduce the sequence input into MPS format.
    encoded_inputs: shape (batch_size, time_steps * feature_dim)
    """
    s = tf.shape(encoded_inputs)
    batch_size = s[0]
    flat_dim = s[1]

    # Reshape to 2D matrix [batch_size, flat_dim]
    u, s, vh = tf.linalg.svd(encoded_inputs, full_matrices=False)
    
    # Truncate to bond_dim
    u_trunc = u[:, :, :bond_dim]
    s_trunc = s[:, :bond_dim]
    vh_trunc = vh[:, :bond_dim, :]

    # Return: (batch_size, bond_dim), singular vectors
    compressed = tf.matmul(u_trunc, tf.linalg.diag(s_trunc))
    return compressed, vh_trunc
