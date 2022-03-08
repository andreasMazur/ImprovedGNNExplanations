import tensorflow as tf
import numpy as np
import random

from double_q_learning.preprocessing import FEATURE_DIM, AMT_NODES


@tf.function
def compute_target_values(q_values,
                          qs_next_state,
                          dones,
                          batch_size,
                          actions,
                          rewards,
                          discount_factor):
    """Computes the target q-values for experience replay.

    :param q_values:
    :param qs_next_state:
    :param dones:
    :param batch_size:
    :param actions:
    :param rewards:
    :param discount_factor:
    :return:
    """

    # Determine the indices of the actions which DID NOT end an episode (within q_values-tensor)
    no_dones = tf.math.logical_not(dones)
    indices_no_dones = tf.boolean_mask(tf.range(batch_size), no_dones)
    actions_no_dones = tf.boolean_mask(actions, no_dones)
    rewards_no_dones = tf.boolean_mask(rewards, no_dones)
    update_indices = tf.stack([indices_no_dones, actions_no_dones], axis=1)
    update_indices = tf.cast(update_indices, tf.int32)

    # Determine target q-values
    qs_next_state_no_dones = tf.boolean_mask(qs_next_state, no_dones)
    update_values = rewards_no_dones + discount_factor * tf.math.reduce_max(
        qs_next_state_no_dones, axis=1
    )

    # Overwrite old q-values with target q-values [no dones]
    q_values = tf.tensor_scatter_nd_update(q_values, update_indices, update_values)

    # Determine the indices of the actions which DID end an episode
    indices_dones = tf.boolean_mask(tf.range(batch_size), dones)
    actions_dones = tf.boolean_mask(actions, dones)
    update_indices = tf.stack([indices_dones, actions_dones], axis=1)
    update_indices = tf.cast(update_indices, tf.int32)

    # Target q-values for done-actions are simply the corresponding rewards
    rewards_dones = tf.boolean_mask(rewards, dones)

    # Overwrite old q-values with target q-values [dones] and return
    return tf.tensor_scatter_nd_update(q_values, update_indices, rewards_dones)


def train_step(replay_memory,
               batch_size,
               discount_factor,
               model,
               target_model,
               trace=None,
               amt_nodes=AMT_NODES):
    """One training step 'double experience replay'.

    :param amt_nodes:
    :param replay_memory:
    :param batch_size:
    :param discount_factor:
    :param model:
    :param target_model:
    :param trace:
    :return:
    """

    # Sample experiences
    mini_batch = np.array(random.sample(replay_memory, batch_size), dtype=object)

    # Convert data to tensors
    states = tf.constant(np.stack(mini_batch[:, 0]).astype(np.float32))
    adj_matrices = tf.constant(np.stack(mini_batch[:, 1]).astype(np.int32))
    next_states = tf.constant(np.stack(mini_batch[:, 4]).astype(np.float32))
    next_adj_matrices = tf.constant(np.stack(mini_batch[:, 5]).astype(np.int32))
    actions = tf.constant(mini_batch[:, 2].astype(np.int32))
    rewards = tf.constant(mini_batch[:, 3].astype(np.float32))
    dones = tf.constant(np.array((mini_batch[:, 6])).astype(np.bool))

    # Train network
    if trace is None:

        trace = tf.function(double_experience_replay).get_concrete_function(
            tf.TensorSpec([None], dtype=tf.bool),  # dones
            tf.TensorSpec((), dtype=tf.int32),  # batch_size
            tf.TensorSpec([None], dtype=tf.int32),  # actions
            tf.TensorSpec([None], dtype=tf.float32),  # rewards
            tf.TensorSpec((), dtype=tf.float32),  # discount_factor
            tf.TensorSpec([None, amt_nodes, FEATURE_DIM], dtype=tf.float32),  # states
            tf.TensorSpec([None, amt_nodes, amt_nodes], dtype=tf.int32),  # adj_matrices
            tf.TensorSpec([None, amt_nodes, FEATURE_DIM], dtype=tf.float32),  # next_states
            tf.TensorSpec([None, amt_nodes, amt_nodes], dtype=tf.int32),  # next_adj_matrices
            model,
            target_model
        )

        loss = trace(
            dones,
            tf.constant(batch_size),
            actions,
            rewards,
            tf.constant(discount_factor),
            states,
            adj_matrices,
            next_states,
            next_adj_matrices
        )
    else:
        loss = trace(
            dones,
            tf.constant(batch_size),
            actions,
            rewards,
            tf.constant(discount_factor),
            states,
            adj_matrices,
            next_states,
            next_adj_matrices
        )

    return loss.numpy(), trace


def double_experience_replay(dones,
                             batch_size,
                             actions,
                             rewards,
                             discount_factor,
                             states,
                             adj_matrices,
                             next_states,
                             next_adj_matrices,
                             model,
                             target_model):
    """Explanation assisted experience replay for double q-learning.

    :param dones:
    :param batch_size:
    :param actions:
    :param rewards:
    :param discount_factor:
    :param states:
    :param adj_matrices:
    :param next_states:
    :param next_adj_matrices:
    :param model:
    :param target_model:
    :return:
    """

    with tf.GradientTape() as tape:
        q_values = model((states, adj_matrices))
        next_actions = tf.math.argmax(model((next_states, next_adj_matrices)), axis=1)
        next_actions = tf.cast(next_actions, tf.int32)
        next_actions = tf.stack([tf.range(batch_size), next_actions], axis=1)

        next_q_values = tf.gather_nd(target_model((next_states, next_adj_matrices)), next_actions)
        next_q_values = tf.reshape(next_q_values, (-1, 1))

        target_q_values = compute_target_values(
            q_values,
            next_q_values,
            dones,
            batch_size,
            actions,
            rewards,
            discount_factor
        )

        loss = model.loss["q_values"](target_q_values, q_values)

    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss
