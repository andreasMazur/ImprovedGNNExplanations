import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging

from double_q_learning.advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.neural_networks import load_agent
from double_q_learning.preprocessing import ADJ_MATRIX_SPARSE, draw_discrete_graph
from zorro_algorithm.zorro import zorro_wrapper
from zorro_algorithm.zorro_utils import mean_squared_error

if __name__ == "__main__":
    """Simple function to test the Zorro algorithm."""

    tf.get_logger().setLevel(logging.ERROR)

    # Load the agent
    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [128],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    model = load_agent("../double_q_learning/checkpoints/rl_agent_6", h_set)
    model.load_weights("../learn_explanations/checkpoints/test_set_1")

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

            # Agent predicting q-values and explanation
            state = tf.reshape(state, (1,) + state.shape)
            state = tf.cast(state, tf.float32)
            q_values, _ = model((state, ADJ_MATRIX_SPARSE))
            action = np.argmax(q_values[0])

            # Compute explanation with zorro
            zorro_expl, action_zo, q_values_zo = zorro_wrapper(
                model, state, ADJ_MATRIX_SPARSE, action
            )
            fid = mean_squared_error(q_values, q_values_zo)

            # Draw the explanation graph
            fig, _ = plt.subplots(1, 2, figsize=(10, 5))
            fig.tight_layout()

            plt.subplot(121)
            draw_discrete_graph(state, action_=action, title="Observation", show=False)

            plt.subplot(122)
            draw_discrete_graph(
                zorro_expl, fid=fid, action_=action_zo, title="Explanation", show=False
            )

            plt.savefig(
                f"./explanations/Zorro_Explanation_{episode_number}_{step_number}.svg", format="svg"
            )
            plt.show()

            # Next env step
            state, reward, done, info = env.step(action)
