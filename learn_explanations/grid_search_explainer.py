import random

from double_q_learning.advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.neural_networks import deep_q_network, explainer_network
from double_q_learning.preprocessing import ADJ_MATRIX, AMT_NODES, FEATURE_DIM
from double_q_learning.utils import plot_stats
from learn_explanations.train_explainer import explainer_train_step

from collections import deque

import tensorflow as tf
import numpy as np
import json


def create_h_sets(learning_rates, graph_layers):
    """Creates hyperparameter sets for explainer networks for grid search

    :param learning_rates: The different learning rates which shall be used
    :param graph_layers: The different graph layer shapes to be used
    :return: A list of hyperparameter sets for the grid search
    """

    hyper_parameters = []
    test_set_number = 0
    for lr in learning_rates:
        for gl in graph_layers:
            hyper_parameters.append(
                {
                    "name": f"test_set_{test_set_number}",
                    "learning_rate": lr,
                    "graph_layers": gl
                }
            )
            test_set_number += 1
    return hyper_parameters


def grid_search(batch_size=64):
    """Performs a grid search over a set of different hyperparameter for explainer networks"""

    env = AdvancedTaxiEnv()
    env.seed(123)

    agent = deep_q_network(
        lr=.001,
        graph_layers=[64],
        dense_layers=[]
    )
    agent.load_weights(
        "../double_q_learning/checkpoints/rl_agent"
    )

    h_sets = create_h_sets([.001], [[64, 128]])
    REPETITIONS = 500

    for h_set in h_sets:
        explainer = explainer_network(lr=h_set["learning_rate"], graph_layers=h_set["graph_layers"])

        trace = None
        losses = []
        losses_window = deque(maxlen=100)
        training_steps = -1
        feature_m_set = []

        while training_steps < REPETITIONS:
            done = False
            state = env.reset()

            # Get training data by running episodes
            while not done:
                state = tf.reshape(state, (1,) + state.shape)

                feature_m_set.append(state)

                q_values = agent((state, ADJ_MATRIX))[0]
                action = np.argmax(q_values)
                state, reward, done, info = env.step(action)

            # Collect feature matrices batch (and adj. matrices)
            feature_m_batch = random.sample(feature_m_set, batch_size)
            feature_m_batch = tf.cast(tf.concat(feature_m_batch, axis=0), tf.float32)
            adj_m_batch = tf.cast(tf.stack([ADJ_MATRIX for _ in range(batch_size)]), tf.int32)

            # Train
            if trace is None:
                trace = tf.function(explainer_train_step).get_concrete_function(
                    tf.TensorSpec([None, AMT_NODES, FEATURE_DIM], dtype=tf.float32),
                    tf.TensorSpec([None, AMT_NODES, AMT_NODES], dtype=tf.int32),
                    tf.TensorSpec((), dtype=tf.int32),
                    agent,
                    explainer
                )
                loss = trace(
                    feature_m_batch,
                    adj_m_batch,
                    tf.constant(batch_size)
                )
            else:
                loss = trace(
                    feature_m_batch,
                    adj_m_batch,
                    tf.constant(batch_size)
                )

            losses.append(loss)
            losses_window.append(loss)
            fidelity = np.mean(losses_window)
            training_steps += 1
            print(
                f"Test set: {h_set['name']} - "
                f"Training steps: {training_steps} / {REPETITIONS} - "
                f"Current average fidelity: {fidelity:.3f} - "
                f"Latest fidelity: {losses[-1]:.3f} "
            )

        plot_stats(
            file_name=f"{h_set['name']}_plot",
            path=f"./checkpoints",
            exploration_rate=[],
            avg_fidelity=losses
        )

        # Store hyper parameter
        with open(f"./checkpoints/{h_set['name']}.json", "w") as f:
            json.dump(h_set, f, indent=4, sort_keys=True)

        explainer.save_weights(f"./checkpoints/{h_set['name']}")


if __name__ == "__main__":
    tf.random.set_seed(123)
    np.random.seed(123)
    grid_search()
