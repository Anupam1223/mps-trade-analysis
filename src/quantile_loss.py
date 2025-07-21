# src/quantile_loss.py
import tensorflow as tf

def quantile_loss(quantile):
    """
    Creates a quantile loss function.
    """
    def loss(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * err, (quantile - 1) * err), axis=-1)
    return loss