import tensorflow as tf


def explainer_train_step(feature_m, adj_m, batch_size, model, explainer):
    """Training step function for explainer networks

    :param feature_m: Feature matrices of the observation
    :param adj_m: Adjacency matrices of the observation
    :param batch_size: Amount of training samples to be used during one training step
    :param model: The model to be explained
    :param explainer: The model to train
    :return: The loss of the training step (Fidelity^-)
    """

    # Predict actions (for which an explanation shall be computed)
    q_values = model((feature_m, adj_m))
    indices = tf.cast(tf.argmax(q_values, axis=1), tf.int32)
    indices = tf.stack([tf.range(batch_size), indices], axis=1)

    with tf.GradientTape() as tape:

        # Predict mask
        mask = explainer((feature_m, adj_m))

        # Mask the original feature matrices
        masked_feature_m = tf.math.multiply(feature_m, mask)

        # Predict masked features
        noisy_q_values = model((masked_feature_m, adj_m))

        q_values = tf.gather_nd(q_values, indices)
        noisy_q_values = tf.gather_nd(noisy_q_values, indices)

        # Compute fidelity minus between original- and masked matrices prediction
        fidelity_minus = explainer.loss["masked_q_values_loss"](q_values, noisy_q_values)

    grads = tape.gradient(fidelity_minus, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return fidelity_minus
