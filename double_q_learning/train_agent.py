from collections import deque

from advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.experience_replay import train_step
from double_q_learning.utils import plot_stats
from neural_networks import deep_q_network
from preprocessing import ADJ_MATRIX_SPARSE

import json
import logging
import numpy as np
import sys
import tensorflow as tf


def determine_current_epsilon(training_step,
                              fifty_percent_chance,
                              start_epsilon=1.0,
                              min_epsilon=0.01):
    """Determines the current epsilon value.

    :param fifty_percent_chance: The training step in which epsilon shall be .5
    :param training_step: The current training step
    :param start_epsilon: The initial value for epsilon
    :param min_epsilon: The minimal value for epsilon
    :return: A probability for choosing a random action (epsilon)
    """
    return max(
        min_epsilon,
        start_epsilon / (1 + (training_step / fifty_percent_chance))
    )


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
        q_values = q_network((state, ADJ_MATRIX_SPARSE))
        if tf.rank(q_values) == 2:
            q_values = q_values[0]
        return np.argmax(q_values)
    else:
        return np.random.randint(0, action_space)


def train(model_name="rl_agent_8",
          learning_rate=.001,
          discount_factor=.95,
          batch_size=64,
          fifty_percent_chance=3_000,
          update_target=500,
          graph_layers=None,
          dense_layers=None,
          INITIAL_REPLAY_MEM_LENGTH=1_000,
          TARGET_PERFORMANCE=20):
    """The training procedure for the RL-agent (meta function for double Q-learning)

    :param model_name: The name under which the agent shall be stored
    :param learning_rate: The learning rate for the deep-Q-network
    :param discount_factor: The discount factor for the expected discounted return
    :param batch_size: The batch size for training examples in one training step
    :param fifty_percent_chance: The training step in which epsilon shall be .5
    :param update_target: The frequency in which the target Q-network will be updated
                          (in training steps)
    :param graph_layers: The shapes of the general graph convolutions applied to the input
    :param dense_layers: The shapes of the dense layers used to classify the graph embedding
    :param INITIAL_REPLAY_MEM_LENGTH: The amount of samples within the replay memory before starting
                                      the actual training procedure
    :param TARGET_PERFORMANCE: Termination condition - Average amount of required steps for an
                               episode for the past 100 (see win_size variable) steps
    """

    if dense_layers is None:
        dense_layers = []

    if graph_layers is None:
        graph_layers = [256]

    #########
    # Setup
    #########
    h_set = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "batch_size": batch_size,
        "fifty_percent_chance": fifty_percent_chance,
        "update_target": update_target,
        "graph_layers": graph_layers,
        "dense_layers": dense_layers,
        "INITIAL_REPLAY_MEM_LENGTH": INITIAL_REPLAY_MEM_LENGTH,
        "TARGET_PERFORMANCE": TARGET_PERFORMANCE
    }

    env = AdvancedTaxiEnv()
    env.seed(123456789)
    replay_memory = deque(maxlen=250_000)
    model = deep_q_network(
        lr=h_set["learning_rate"],
        graph_layers=h_set["graph_layers"]
    )
    model.summary()
    target_model = deep_q_network(
        lr=h_set["learning_rate"],
        graph_layers=h_set["graph_layers"]
    )
    target_model.set_weights(model.get_weights())

    epsilon = 1.0
    performance = 200
    episode_num = -1
    training_step = 0
    win_size = 1_000
    episode_steps_mem = []
    episode_steps_window = deque(maxlen=win_size)
    episode_rewards_mem = []
    episode_rewards_window = deque(maxlen=win_size)
    avg_batch_losses_mem = []
    avg_batch_losses = deque(maxlen=win_size)
    avg_episode_steps_mem = []
    epsilon_mem = []
    target_updates = []
    trace = None

    ############
    # Warm Up
    ############
    assert INITIAL_REPLAY_MEM_LENGTH >= batch_size, \
        f"Replay memory must at least contain {batch_size} (batch_size) experiences."
    while len(replay_memory) < INITIAL_REPLAY_MEM_LENGTH:
        sys.stdout.write(
            f"\rInitialize replay memory.. ({len(replay_memory)}/{INITIAL_REPLAY_MEM_LENGTH}) "
        )
        state = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_strategy(model, epsilon, state)
            next_state, reward, done, info = env.step(action)
            replay_memory.append(
                (
                    state,
                    tf.constant(action),
                    tf.constant(reward),
                    next_state,
                    tf.constant(done)
                )
            )
            state = next_state
    print("done.")

    ############
    # Training
    ############
    while performance > TARGET_PERFORMANCE:
        episode_num += 1
        episode_steps = 0
        episode_reward = 0
        losses = []

        state = env.reset()
        done = False
        while not done:
            sys.stdout.write(
                f"\rEpisode: {episode_num} - "
                f"Step: {episode_steps} - "
                f"Steps trained: {training_step} - "
                f"Avg. step required: {performance:.3f} - "
                f"Avg. episode reward: {np.mean(episode_rewards_window):.3f} - "
                f"Avg. batch loss: {np.mean(avg_batch_losses):.3f} - "
                f"Epsilon: {epsilon:.3f} "
            )

            # Episode step
            action = epsilon_greedy_strategy(model, epsilon, state)
            next_state, reward, done, info = env.step(action)
            replay_memory.append(
                (
                    state,
                    tf.constant(action),
                    tf.constant(reward),
                    next_state,
                    tf.constant(done)
                )
            )
            state = next_state

            # Training
            loss, trace = train_step(
                replay_memory,
                batch_size,
                discount_factor,
                model,
                target_model,
                trace
            )
            training_step += 1
            epsilon = determine_current_epsilon(training_step, fifty_percent_chance)

            # Update target network
            if training_step % update_target == 0 and training_step > 0:
                target_model.set_weights(model.get_weights())
                target_updates.append(episode_num)
                print("Updated target model!")

            # Collect statistics
            episode_reward += reward
            episode_steps += 1
            losses.append(loss)

        episode_steps_mem.append(episode_steps)
        episode_steps_window.append(episode_steps)

        episode_rewards_mem.append(episode_reward)
        episode_rewards_window.append(episode_reward)

        episode_loss = np.mean(losses)
        avg_batch_losses.append(episode_loss)
        avg_batch_losses_mem.append(episode_loss)

        performance = np.mean(episode_steps_window)
        avg_episode_steps_mem.append(performance)

        epsilon_mem.append(epsilon)

        if episode_num % 250 == 0 and episode_num > 0:
            # Save model weights
            model.save_weights(f"./checkpoints/{model_name}")

            # Plot statistics
            if update_target < 600:
                temp_target_updates = None
            else:
                temp_target_updates = target_updates.copy()
            plot_stats(
                file_name=f"{model_name}_plot",
                path=f"../misc/double_q_learning/checkpoints",
                exploration_rate=epsilon_mem,
                target_updates_=temp_target_updates,
                episode_steps=episode_steps_mem,
                performance=avg_episode_steps_mem,
                episode_rewards=episode_rewards_mem,
                avg_losses=avg_batch_losses_mem,
            )

            # Store hyper parameter
            with open(f"./checkpoints/{model_name}.json", "w") as f:
                json.dump(h_set, f, indent=4, sort_keys=True)

            tf.keras.backend.clear_session()

    # Save model weights
    model.save_weights(f"./checkpoints/{model_name}")

    # Plot statistics
    if update_target < 600:
        target_updates = None
    plot_stats(
        file_name=f"{model_name}_plot",
        path=f"../misc/double_q_learning/checkpoints",
        exploration_rate=epsilon_mem,
        target_updates_=target_updates,
        episode_steps=episode_steps_mem,
        performance=avg_episode_steps_mem,
        episode_rewards=episode_rewards_mem,
        avg_losses=avg_batch_losses_mem,
    )

    # Store hyper parameter
    with open(f"./checkpoints/{model_name}.json", "w") as f:
        json.dump(h_set, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    tf.random.set_seed(123)
    np.random.seed(123)
    tf.get_logger().setLevel(logging.ERROR)
    np.set_printoptions(threshold=5, precision=2)

    train()
