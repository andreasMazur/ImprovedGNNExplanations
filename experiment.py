from matplotlib import pyplot as plt

from advanced_taxi_env import AdvancedTaxiEnv
from neural_networks import load_agent
from preprocessing import ADJ_MATRIX_SPARSE, draw_discrete_graph, draw_heat_graph
from zorro_algorithm.zorro import zorro_wrapper
from zorro_algorithm.zorro_utils import mean_squared_error

import numpy as np
import tensorflow as tf


def visualize_actions(actions, ref_actions, zo_actions, zp_actions, idx):

    actions = np.array(actions)
    ref_actions = np.array(ref_actions)
    zo_actions = np.array(zo_actions)
    zp_actions = np.array(zp_actions)
    actions = {
        "Proxy": (actions == ref_actions).astype(np.int32).sum(),
        "Zorro Observation": (actions == zo_actions).astype(np.int32).sum(),
        "Zorro Proxy": (actions == zp_actions).astype(np.int32).sum()
    }

    plt.bar(actions.keys(), actions.values(), color="maroon", width=.3)
    plt.xlabel("Experiment")
    plt.ylabel("Correctly predicted actions")
    plt.title("Action Comparison")
    plt.savefig(f"./ExperimentImages/Experiment_actions_{idx}.svg", format="svg")
    plt.show()


def visualize_fidelities(ref_fid, zo_fid, zp_fid, idx):

    plt.figure(figsize=(12, 8))
    plt.title("Fidelity Values")
    plt.ylabel("Step Fidelity")
    plt.xlabel("Episode Step")

    plt.plot(ref_fid, label="Proxy")
    plt.plot(zo_fid, label="Zorro Observation")
    plt.plot(zp_fid, label="Zorro Proxy")

    plt.grid()
    plt.legend()
    plt.savefig(f"./ExperimentImages/Experiment_{idx}_fidelities.svg", format="svg")
    plt.show()


def visualize_graphs(state, a_orig, ref, fid_ref, a_ref, zo, fid_zo, a_zo, zp, fid_zp, a_zp, idx):
    
    fig, _ = plt.subplots(1, 4, figsize=(14, 3.5))
    # fig.suptitle(f"Episode step {idx.split('_')[1]}")
    fig.tight_layout()

    plt.subplot(141)
    draw_discrete_graph(state, action_=a_orig, title="Observation", show=False)

    plt.subplot(142)
    draw_heat_graph(ref, fid=fid_ref, action_=a_ref, title="Proxy", show=False)

    plt.subplot(143)
    draw_discrete_graph(zo, fid=fid_zo, action_=a_zo, title="Zorro Observation", show=False)

    plt.subplot(144)
    draw_heat_graph(zp, fid=fid_zp, action_=a_zp, title="Zorro Proxy", show=False)

    plt.savefig(f"./ExperimentImages/Experiment_Proxy_{idx}.svg", format="svg")
    plt.show()


def use_network(model, state):
    """Network wrapper."""

    if len(state.shape) == 2:
        state = tf.expand_dims(state, axis=0)
    state = tf.cast(state, tf.float32)
    q_values, proxy = model((state, ADJ_MATRIX_SPARSE))
    q_values = q_values[0]
    return np.argmax(q_values), proxy, q_values


def main(agent_checkpoint="./double_q_learning/checkpoints/rl_agent",
         proxy_checkpoint="./learn_proxies/checkpoints/test_set"):
    """The main experiment."""

    # Setup environment and seeds
    seed = 144301
    env = AdvancedTaxiEnv()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Load the agent: MAKE SURE TO SAME HYPERPARAMETERS AS USED TO TRAIN THE PROXY BRANCH WHICH IS STORED IN
    #                 `proxy_checkpoint`!
    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [256],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    model = load_agent(agent_checkpoint, h_set)
    model.load_weights(proxy_checkpoint)

    MAX_EPISODES = 150
    for episode_number in range(MAX_EPISODES):
        # Track episode statistics
        fid_ref_mem = []
        fid_zo_mem = []  # zo -> Zorro observation
        fid_zp_mem = []  # zp -> Zorro proxy

        ref_expl_mem = []
        zo_expl_mem = []
        zp_expl_mem = []

        action_mem = []
        action_ref_mem = []
        action_zo_mem = []
        action_zp_mem = []

        step_number = -1
        done = False
        state = env.reset()
        while not done and step_number < 35:
            step_number += 1
            env.render()

            # Compute agent prediction and proxy for original input and agent prediction for proxy
            action, proxy, q_values = use_network(model, state)
            action_ref, _, q_values_ref = use_network(model, proxy)

            # Compute fidelity
            fid_ref = mean_squared_error(q_values, q_values_ref)

            # Apply Zorro algorithm on original input
            zorro_original, action_zo, fid_zo = zorro_wrapper(model, state, state)

            # Apply Zorro algorithm on proxy
            zorro_noisy, action_zp, fid_zp = zorro_wrapper(model, proxy, state)

            visualize_graphs(
                state, action,
                proxy, fid_ref, action_ref,
                zorro_original, fid_zo, action_zo,
                zorro_noisy, fid_zp, action_zp,
                f"{episode_number}_{step_number}"
            )
            print(f"fid_zo: {fid_zo} - fid_zp: {fid_zp} - fid_ref: {fid_ref}")

            fid_ref_mem.append(fid_ref)
            fid_zo_mem.append(fid_zo)
            fid_zp_mem.append(fid_zp)

            action_mem.append(action)
            action_ref_mem.append(action_ref)
            action_zo_mem.append(action_zo)
            action_zp_mem.append(action_zp)

            if len(proxy.shape) == 3:
                proxy = proxy[0]
            ref_expl_mem.append(proxy)

            if len(zorro_original.shape) == 3:
                zorro_original = zorro_original[0]
            zo_expl_mem.append(zorro_original)

            if len(zorro_noisy.shape) == 3:
                zorro_noisy = zorro_noisy[0]
            zp_expl_mem.append(zorro_noisy)

            state, reward, done, info = env.step(action)

        visualize_fidelities(fid_ref_mem, fid_zo_mem, fid_zp_mem, idx=episode_number)
        visualize_actions(action_mem, action_ref_mem, action_zo_mem, action_zp_mem, idx=episode_number)

        # Store episode proxies
        ref_expl_mem = np.stack(ref_expl_mem)
        zo_expl_mem = np.stack(zo_expl_mem)
        zp_expl_mem = np.stack(zp_expl_mem)
        np.save(f"ExperimentImages/ref_expl_mem_{episode_number}.npy", ref_expl_mem)
        np.save(f"ExperimentImages/zo_expl_mem_{episode_number}.npy", zo_expl_mem)
        np.save(f"ExperimentImages/zp_expl_mem_{episode_number}.npy", zp_expl_mem)

        # Store episode fidelities
        fid_ref_mem = np.stack(fid_ref_mem)
        fid_zo_mem = np.stack(fid_zo_mem)
        fid_zp_mem = np.stack(fid_zp_mem)
        np.save(f"ExperimentImages/ref_fid_mem_{episode_number}.npy", fid_ref_mem)
        np.save(f"ExperimentImages/zo_fid_mem_{episode_number}.npy", fid_zo_mem)
        np.save(f"ExperimentImages/zp_fid_mem_{episode_number}.npy", fid_zp_mem)

        # Store episode actions
        action_mem = np.stack(action_mem)
        action_ref_mem = np.stack(action_ref_mem)
        action_zo_mem = np.stack(action_zo_mem)
        action_zp_mem = np.stack(action_zp_mem)
        np.save(f"ExperimentImages/action_mem_{episode_number}.npy", action_mem)
        np.save(f"ExperimentImages/ref_action_mem_{episode_number}.npy", action_ref_mem)
        np.save(f"ExperimentImages/zo_action_mem_{episode_number}.npy", action_zo_mem)
        np.save(f"ExperimentImages/zp_action_mem_{episode_number}.npy", action_zp_mem)


if __name__ == "__main__":
    np.set_printoptions(threshold=5, precision=2)
    main()
