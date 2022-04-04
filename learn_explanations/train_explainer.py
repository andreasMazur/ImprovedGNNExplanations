from double_q_learning.preprocessing import AMT_NODES, ADJ_MATRIX_SPARSE, FEATURE_DIM

import tensorflow as tf
import numpy as np
import random


def epsilon_greedy_strategy(q_network, epsilon, state, action_space=6):
    """Epsilon-greedy strategy

    Returns a random action with the probability 'epsilon'.

    :param state: The state for which an actions shall be predicted
    :param epsilon: The current epsilon value
    :param q_network: The q-network used to predict actions
    :param action_space: Amount of possible actions to choose from
    :return: An action
    """
    if np.random.random() > epsilon:
        state = tf.reshape(state, (1,) + state.shape)
        q_values, _ = q_network((state, ADJ_MATRIX_SPARSE))
        q_values = q_values[0]
        return np.argmax(q_values)
    else:
        return np.random.randint(0, action_space)


def explanation_gradient(states,
                         fidelity_reg,
                         model):
    """Explanation assisted experience replay for double q-learning.

    :param fidelity_reg:
    :param states:
    :param model:
    :return:
    """

    with tf.GradientTape() as tape:
        #################################
        # EXPLANATION LOSS (Fidelity)
        #################################
        q_values, explanation = model((states, ADJ_MATRIX_SPARSE))
        explanation_q_values, _ = model((explanation, ADJ_MATRIX_SPARSE))
        fidelity = model.loss["mse"](q_values, explanation_q_values)
        fidelity_loss = fidelity_reg * fidelity

    grads = tape.gradient(fidelity_loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return {"fidelity": fidelity}


def explanation_train_step(replay_memory,
                           batch_size,
                           fidelity_reg,
                           model,
                           trace=None):
    """One training step 'double experience replay'."""

    # Sample experiences
    mini_batch = np.array(random.sample(replay_memory, batch_size), np.float32)

    # Convert data to tensors
    states = tf.constant(mini_batch)
    fidelity_reg = tf.constant(fidelity_reg)

    # loss = explanation_gradient(states, fidelity_reg, model)

    # Train network
    if trace is None:
        trace = tf.function(explanation_gradient).get_concrete_function(
            tf.TensorSpec([None, AMT_NODES, FEATURE_DIM], dtype=tf.float32),  # states
            tf.TensorSpec((), dtype=tf.float32),  # fidelity regularization factor
            model
        )
        loss = trace(states, fidelity_reg)
    else:
        loss = trace(states, fidelity_reg)

    return loss, trace
