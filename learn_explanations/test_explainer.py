from double_q_learning.advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.neural_networks import load_agent
from double_q_learning.preprocessing import draw_discrete_graph, ADJ_MATRIX_SPARSE, draw_heat_graph
from zorro_algorithm.zorro_utils import mean_squared_error

from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import time


if __name__ == "__main__":
    """Simple function to see how the agent behaves"""
    np.set_printoptions(threshold=5, precision=2)

    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [128],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    model = load_agent("../double_q_learning/checkpoints/rl_agent_6", h_set)
    model.load_weights("./checkpoints/test_set_1")

    # Load environment
    env = AdvancedTaxiEnv()

    MAX_EPISODES = 1
    for episode_number in range(MAX_EPISODES):
        done = False
        state = env.reset()
        step_number = -1
        while not done:
            step_number += 1

            env.render()
            time.sleep(1)

            # Agent predicting q-values and explanation
            state = tf.reshape(state, (1,) + state.shape)
            state = tf.cast(state, tf.float32)
            q_values, explanation = model((state, ADJ_MATRIX_SPARSE))
            action = np.argmax(q_values[0])

            noisy_q_values = model((explanation, ADJ_MATRIX_SPARSE))
            noisy_q_values = noisy_q_values[0]
            noisy_action = np.argmax(noisy_q_values)

            fid = mean_squared_error(q_values, noisy_q_values)

            # Draw the explanation graph
            fig, _ = plt.subplots(1, 2, figsize=(10, 5))
            fig.tight_layout()

            plt.subplot(121)
            draw_discrete_graph(state, action_=action, title="Observation", show=False)

            plt.subplot(122)
            draw_heat_graph(
                explanation[0], fid=fid, action_=noisy_action, title="Explanation", show=False
            )

            plt.savefig(
                f"./explanations/Explanation_{episode_number}_{step_number}.svg", format="svg"
            )
            plt.show()
            # Next env step
            state, reward, done, info = env.step(action)
