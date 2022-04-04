from double_q_learning.advanced_taxi_env import AdvancedTaxiEnv
from double_q_learning.neural_networks import load_agent
from double_q_learning.preprocessing import ADJ_MATRIX_SPARSE, draw_discrete_graph, draw_heat_graph
from zorro_algorithm.zorro import zorro_wrapper
from zorro_algorithm.zorro_utils import mean_squared_error

from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np


def visualize_fidelities(ref_fid, zo_fid, zn_fid):

    plt.figure(figsize=(12, 8))
    plt.title("Fidelities")
    plt.ylabel("step fidelity")
    plt.xlabel("episode step")

    plt.plot(ref_fid, label="Explanation Network Output")
    plt.plot(zo_fid, label="Zorro on Input")
    plt.plot(zn_fid, label="Zorro on Explanation")

    plt.grid()
    plt.legend()
    plt.savefig("./ShortExperimentImages/ShortExperiment_fidelities.svg", format="svg")
    plt.show()


def visualize_graphs(state, a_orig, ref, fid_ref, a_ref, zo, fid_zo, a_zo, zn, fid_zn, a_zn, idx):
    
    fig, _ = plt.subplots(2, 2)
    fig.suptitle(f"Episode step {idx.split('_')[1]}")
    fig.tight_layout()

    plt.subplot(221)
    draw_discrete_graph(state, action_=a_orig, title="Observation", show=False)

    plt.subplot(222)
    draw_heat_graph(ref, fid=fid_ref, action_=a_ref, title="Network Explanation", show=False)

    plt.subplot(223)
    draw_discrete_graph(zo, fid=fid_zo, action_=a_zo, title="Zorro Original", show=False)

    plt.subplot(224)
    draw_heat_graph(zn, fid=fid_zn, action_=a_zn, title="Zorro Noisy", show=False)

    plt.savefig(f"./ShortExperimentImages/ShortExperiment_Explanation_{idx}.svg", format="svg")
    plt.show()


def use_network(model, state):
    """Network wrapper."""

    if len(state.shape) == 2:
        state = tf.expand_dims(state, axis=0)
    state = tf.cast(state, tf.float32)
    q_values, explanation = model((state, ADJ_MATRIX_SPARSE))
    q_values = q_values[0]
    return np.argmax(q_values), explanation, q_values


def main():
    """The main experiment."""

    # Setup environment and seeds
    env = AdvancedTaxiEnv()
    tf.random.set_seed(0)
    np.random.seed(0)
    env.seed(0)

    # Load the agent
    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [128],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    model = load_agent("./double_q_learning/checkpoints/rl_agent_6", h_set)
    model.load_weights("./learn_explanations/checkpoints/test_set_1")

    MAX_EPISODES = 5
    for episode_number in range(MAX_EPISODES):
        # Track episode statistics
        fid_ref_mem = []
        fid_zo_mem = []
        fid_zn_mem = []
        ref_expl_mem = []
        zo_expl_mem = []
        zn_expl_mem = []

        step_number = -1
        done = False
        state = env.reset()
        while not done:
            step_number += 1
            env.render()
            action, explanation, q_values = use_network(model, state)
            action_ref, _, q_values_ref = use_network(model, explanation)
            fid_ref = mean_squared_error(q_values, q_values_ref)

            # Apply Zorro algorithm on original input
            zorro_original, action_zo, q_values_zo = zorro_wrapper(
                model, state, ADJ_MATRIX_SPARSE, action
            )
            fid_zo = mean_squared_error(q_values, q_values_zo)

            # Apply Zorro algorithm on explanation
            zorro_noisy, action_zn, q_values_zn = zorro_wrapper(
                model, explanation, ADJ_MATRIX_SPARSE, action
            )
            fid_zn = mean_squared_error(q_values, q_values_zn)

            visualize_graphs(
                state, action,
                explanation, fid_ref, action_ref,
                zorro_original, fid_zo, action_zo,
                zorro_noisy, fid_zn, action_zn,
                f"{episode_number}_{step_number}"
            )
            print(f"fid_zo: {fid_zo} - fid_zn: {fid_zn} - fid_ref: {fid_ref}")

            fid_ref_mem.append(fid_ref)
            fid_zo_mem.append(fid_zo)
            fid_zn_mem.append(fid_zn)

            if len(explanation.shape) == 3:
                explanation = explanation[0]
            ref_expl_mem.append(explanation)

            if len(zorro_original.shape) == 3:
                zorro_original = zorro_original[0]
            zo_expl_mem.append(zorro_original)

            if len(zorro_noisy.shape) == 3:
                zorro_noisy = zorro_noisy[0]
            zn_expl_mem.append(zorro_noisy)

            state, reward, done, info = env.step(action)

        visualize_fidelities(fid_ref_mem, fid_zo_mem, fid_zn_mem)

        ref_expl_mem = np.stack(ref_expl_mem)
        zo_expl_mem = np.stack(zo_expl_mem)
        zn_expl_mem = np.stack(zn_expl_mem)
        np.save(f"ExperimentImages/ref_expl_mem_{episode_number}.npy", ref_expl_mem)
        np.save(f"ExperimentImages/zo_expl_mem_{episode_number}.npy", zo_expl_mem)
        np.save(f"ExperimentImages/zn_expl_mem_{episode_number}.npy", zn_expl_mem)


if __name__ == "__main__":
    np.set_printoptions(threshold=5, precision=2)
    main()
