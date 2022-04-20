from double_q_learning.advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.neural_networks import deep_q_network, load_agent
from double_q_learning.utils import plot_stats
from learn_explanations.train_explainer import epsilon_greedy_strategy, explanation_train_step

from collections import deque
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import json
import logging
import sys


def create_h_sets(learning_rates,
                  batch_sizes,
                  graph_layers,
                  expl_graph_layers,
                  fidelity_reg):
    """Creates hyperparameter sets for explainer networks for grid search

    :param fidelity_reg:
    :param batch_sizes:
    :param learning_rates: The different learning rates which shall be used
    :param graph_layers: The different graph layer shapes to be used
    :param expl_graph_layers:
    :return: A list of hyperparameter sets for the grid search
    """

    hyper_parameters = []
    test_set_number = 0
    for lr in learning_rates:
        for bs in batch_sizes:
            for gl in graph_layers:
                for e_gl in expl_graph_layers:
                    for fpc_fid in fidelity_reg:
                        hyper_parameters.append(
                            {
                                "name": f"test_set_{test_set_number}",
                                "learning_rate": lr,
                                "batch_size": bs,
                                "graph_layers": gl,
                                "expl_graph_layers": e_gl,
                                "fidelity_reg": fpc_fid
                            }
                        )
                        test_set_number += 1
    return hyper_parameters


def grid_search(INITIAL_REPLAY_MEM_LENGTH=1_000,
                AMT_TRAINING_EPISODES=4_000):
    """Function that trains the explanation branch of an extended GNN-architecture.

    The 'extended GNN-architecture' refers to the fact, that an explanation branch is added
    to the feature-embedding layer of a Q-network.

    Grid-search:
        Instead of training one network until a goal criterion is met, train multiple networks
        for a fixed amount of training episodes.

    :param INITIAL_REPLAY_MEM_LENGTH: The amount of samples within the replay memory before starting
                                      the actual training procedure
    :param AMT_TRAINING_EPISODES:
    """

    h_sets = create_h_sets(
        learning_rates=[.001],
        batch_sizes=[64],
        graph_layers=[[256]],  # depends on what model you retrain
        expl_graph_layers=[[128], [128, 128]],
        fidelity_reg=[.001, 1., .01]
    )

    print(f"Created {len(h_sets)} many hyperparameter sets!")

    for h_set in tqdm(h_sets):

        #########
        # Setup
        #########
        env = AdvancedTaxiEnv()
        tf.random.set_seed(123)
        np.random.seed(123)
        seed = env.seed(123)
        replay_memory = deque(maxlen=250_000)
        model = load_agent("../double_q_learning/checkpoints/rl_agent_8", h_set)

        fidelity_reg = h_set["fidelity_reg"]
        episode_num = -1
        training_step = 0
        win_size = 150
        trace = None

        avg_fidelity_mem = []
        avg_fidelity_window = deque(maxlen=win_size)

        ############
        # Warm Up
        ############
        assert INITIAL_REPLAY_MEM_LENGTH >= h_set["batch_size"], \
            f"Replay memory must at least contain {h_set['batch_size']} (batch_size) experiences."
        while len(replay_memory) < INITIAL_REPLAY_MEM_LENGTH:
            sys.stdout.write(
                f"\rInitialize replay memory.. ({len(replay_memory)}/{INITIAL_REPLAY_MEM_LENGTH}) "
            )
            state = env.reset()
            done = False
            while not done:
                action = epsilon_greedy_strategy(model, .0, state)
                next_state, reward, done, info = env.step(action)
                replay_memory.append(state)
                state = next_state
        print("done.")

        ############
        # Training
        ############
        while episode_num < AMT_TRAINING_EPISODES:
            episode_num += 1
            episode_steps = 0
            episode_reward = 0
            episode_fidelities = []

            state = env.reset()
            done = False
            while not done:
                sys.stdout.write(
                    f"\r{h_set['name']}: "
                    f"Episode: {episode_num} - "
                    f"Step: {episode_steps} - "
                    f"Steps trained: {training_step} - "
                    f"Avg. fidelity: {np.mean(avg_fidelity_window):.3f} "
                )

                # Episode step
                action = epsilon_greedy_strategy(model, .0, state)
                next_state, reward, done, info = env.step(action)
                replay_memory.append(state)
                state = next_state

                # Training
                loss, trace = explanation_train_step(
                    replay_memory,
                    h_set["batch_size"],
                    fidelity_reg,
                    model,
                    trace=trace
                )
                training_step += 1

                # Collect statistics
                episode_reward += reward
                episode_steps += 1
                episode_fidelities.append(loss["fidelity"])

            episode_avg_fidelity = np.mean(episode_fidelities)
            avg_fidelity_window.append(episode_avg_fidelity)
            avg_fidelity_mem.append(episode_avg_fidelity)

            if episode_num % 250 == 0 and episode_num > 0:
                # Save model weights
                model.save_weights(f"./checkpoints/{h_set['name']}")

                # Plot statistics
                plot_stats(
                    file_name=f"{h_set['name']}_plot",
                    path=f"./checkpoints",
                    exploration_rate=[],
                    avg_fidelity=avg_fidelity_mem
                )

                # Store hyper parameter
                h_set["seed"] = seed[0]
                with open(f"./checkpoints/{h_set['name']}.json", "w") as f:
                    json.dump(h_set, f, indent=4, sort_keys=True)

                # tf.keras.backend.clear_session()

        # Save model weights
        model.save_weights(f"./checkpoints/{h_set['name']}")

        # Plot statistics
        plot_stats(
            file_name=f"{h_set['name']}_plot",
            path=f"./checkpoints",
            exploration_rate=[],
            avg_fidelity=avg_fidelity_mem
        )

        # Store hyper parameter
        h_set["seed"] = seed[0]
        with open(f"./checkpoints/{h_set['name']}.json", "w") as f:
            json.dump(h_set, f, indent=4, sort_keys=True)

        avg_fidelity_mem = np.array(avg_fidelity_mem)
        np.save(f"./checkpoints/{h_set['name']}.npy", avg_fidelity_mem)

if __name__ == "__main__":
    tf.get_logger().setLevel(logging.ERROR)
    np.set_printoptions(threshold=5, precision=2)

    grid_search()
