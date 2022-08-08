from preprocessing import AMT_NODES, ADJ_MATRIX_SPARSE, FEATURE_DIM

import numpy as np
import random
import tensorflow as tf


def epsilon_greedy_strategy(q_network, epsilon, state, action_space=6):
    """Epsilon-greedy strategy

    Parameters
    ----------
    state: np.ndarray
        The state for which an actions shall be predicted
    epsilon: float
        The current epsilon value
    q_network: tf.keras.Model
        The deep Q-network used to predict actions
    action_space: int
        Amount of possible actions to choose from

    Returns
    -------
    int
        The index of the largest Q-value, i.e. the action
    """
    if np.random.random() > epsilon:
        state = tf.reshape(state, (1,) + state.shape)
        q_values, _ = q_network((state, ADJ_MATRIX_SPARSE))
        q_values = q_values[0]
        return np.argmax(q_values)
    else:
        return np.random.randint(0, action_space)


def proxy_gradient(states,
                   fidelity_reg,
                   model):
    """Calculates the gradient to train for good proxies

    Parameters
    ----------
    states: np.ndarray
        A batch of observations, i.e. feature matrices, from the environment
    fidelity_reg: float
        A regularization constant for the fidelity loss
    model: tf.keras.Model
        The deep Q-network to train

    Returns
    -------
    dict:
        The current fidelity of a predicted proxy w.r.t. the original observation
    """

    with tf.GradientTape() as tape:
        #################################
        # PROXY LOSS (Fidelity)
        #################################
        q_values, proxy = model((states, ADJ_MATRIX_SPARSE))
        proxy_q_values, _ = model((proxy, ADJ_MATRIX_SPARSE))
        fidelity = model.loss["mse"](q_values, proxy_q_values)
        fidelity_loss = fidelity_reg * fidelity

    grads = tape.gradient(fidelity_loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return {"fidelity": fidelity}


def proxy_train_step(replay_memory,
                     batch_size,
                     fidelity_reg,
                     model,
                     trace=None):
    """One training step of double experience replay

    Parameters
    ----------
    replay_memory: dequeue
        A double ended queue representing the replay memory of the deep Q-learning agent
    batch_size: int
        The batch size
    fidelity_reg: float
        A regularization constant for the fidelity loss
    model: tf.keras.Model
        The deep Q-network to train
    trace: tf.types.ConcreteFunction
        The trace of the gradient function including the model to train

    Returns
    -------
    (float, tf.types.ConcreteFunction)
        The fidelity loss and the Tensorflow-trace of the loss function including the model (such that we don't need to
        retrace everytime and can re-use gradient function over and over again.)
    """

    # Sample experiences
    mini_batch = np.array(random.sample(replay_memory, batch_size), np.float32)

    # Convert data to tensors
    states = tf.constant(mini_batch)
    fidelity_reg = tf.constant(fidelity_reg)

    # Train network
    if trace is None:
        trace = tf.function(proxy_gradient).get_concrete_function(
            tf.TensorSpec([None, AMT_NODES, FEATURE_DIM], dtype=tf.float32),  # states
            tf.TensorSpec((), dtype=tf.float32),  # fidelity regularization factor
            model
        )
        loss = trace(states, fidelity_reg)
    else:
        loss = trace(states, fidelity_reg)

    return loss, trace
