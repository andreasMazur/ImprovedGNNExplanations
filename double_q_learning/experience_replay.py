from preprocessing import FEATURE_DIM, AMT_NODES, ADJ_MATRIX_SPARSE

import numpy as np
import tensorflow as tf
import random


@tf.function
def compute_target_values(q_values,
                          qs_next_state,
                          dones,
                          batch_size,
                          actions,
                          rewards,
                          discount_factor):
    """Computes the target q-values for experience replay."""

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
    """One training step 'double experience replay'"""

    # Sample experiences
    mini_batch = np.array(random.sample(replay_memory, batch_size), dtype=object)

    # Convert data to tensors
    states = tf.constant(np.stack(mini_batch[:, 0]).astype(np.float32))
    next_states = tf.constant(np.stack(mini_batch[:, 3]).astype(np.float32))
    actions = tf.constant(mini_batch[:, 1].astype(np.int32))
    rewards = tf.constant(mini_batch[:, 2].astype(np.float32))
    dones = tf.constant(np.array((mini_batch[:, 4])).astype(np.bool))

    # loss = double_experience_replay(
    #     dones,
    #     tf.constant(batch_size),
    #     actions,
    #     rewards,
    #     tf.constant(discount_factor),
    #     states,
    #     next_states,
    #     model,
    #     target_model
    # )

    # Train network
    if trace is None:

        trace = tf.function(double_experience_replay).get_concrete_function(
            tf.TensorSpec([None], dtype=tf.bool),  # dones
            tf.TensorSpec((), dtype=tf.int32),  # batch_size
            tf.TensorSpec([None], dtype=tf.int32),  # actions
            tf.TensorSpec([None], dtype=tf.float32),  # rewards
            tf.TensorSpec((), dtype=tf.float32),  # discount_factor
            tf.TensorSpec([None, amt_nodes, FEATURE_DIM], dtype=tf.float32),  # states
            tf.TensorSpec([None, amt_nodes, FEATURE_DIM], dtype=tf.float32),  # next_states
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
            next_states
        )
    else:
        loss = trace(
            dones,
            tf.constant(batch_size),
            actions,
            rewards,
            tf.constant(discount_factor),
            states,
            next_states
        )

    if isinstance(loss, tf.Tensor):
        loss = loss.numpy()

    return loss, trace


def double_experience_replay(dones,
                             batch_size,
                             actions,
                             rewards,
                             discount_factor,
                             states,
                             next_states,
                             model,
                             target_model):
    """Explanation assisted experience replay for double q-learning."""

    with tf.GradientTape() as tape:
        q_values = model((states, ADJ_MATRIX_SPARSE))
        next_q_values = model((next_states, ADJ_MATRIX_SPARSE))

        next_actions = tf.math.argmax(next_q_values, axis=1)
        next_actions = tf.cast(next_actions, tf.int32)
        next_actions = tf.stack([tf.range(batch_size), next_actions], axis=1)

        target_q_values = target_model((next_states, ADJ_MATRIX_SPARSE))
        target_q_values = tf.gather_nd(target_q_values, next_actions)
        target_q_values = tf.reshape(target_q_values, (-1, 1))

        target_q_values = compute_target_values(
            q_values,
            target_q_values,
            dones,
            batch_size,
            actions,
            rewards,
            discount_factor
        )

        loss = model.loss["mse"](target_q_values, q_values)

    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss
