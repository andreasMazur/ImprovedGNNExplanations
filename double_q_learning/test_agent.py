from double_q_learning.advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.neural_networks import deep_q_network
from double_q_learning.preprocessing import ADJ_MATRIX_SPARSE

import tensorflow as tf
import numpy as np
import time


if __name__ == "__main__":
    """Simple function to see how the agent behaves"""

    # Load the agent
    model = deep_q_network(.001, [128])
    model.load_weights("./checkpoints/rl_agent_6")
    model.summary()

    # Load environment
    env = AdvancedTaxiEnv()

    MAX_EPISODES = 5
    for _ in range(MAX_EPISODES):
        done = False
        state = env.reset()
        while not done:
            env.render()
            time.sleep(1)

            state = tf.reshape(state, (1,) + state.shape)
            state = tf.cast(state, tf.float32)
            q_values = model((state, ADJ_MATRIX_SPARSE))[0]
            action = np.argmax(q_values)
            state, reward, done, info = env.step(action)
