"""This script defines intermediary tool to communicate with the Zorro algorithm and tests them

Therefore, the script requires the following libraries:
    - numpy
    - tensorflow

It defines the following functions:
    - get_best_explanation: Choose the best mask from all given masks
    - create_mask: Convert the Zorro explanation to a matrix
    - compute_fidelity: Computes the fidelity (MSE between Q-values)
    - new_feature_fidelity: Wrapper to compute a new fidelity, given an additional feature in Zorro's explanation
    - new_node_fidelity: Wrapper to compute a new fidelity, given an additional node in Zorro's explanation
    - mean_squared_error: Mean squared error to compute fidelity
"""

from neural_networks import load_agent
from preprocessing import AMT_NODES, FEATURE_DIM, ADJ_MATRIX_SPARSE

import numpy as np
import tensorflow as tf


def get_best_explanation(masks, model_, to_mask, original_input):
    """Choose the best mask from all given masks

    Parameters
    ----------
    masks: list
        The explanations given by Zorro
    model_: tf.keras.Model
        The model to analyse
    to_mask: np.ndarray
        The feature matrix which shall be masked (original observation/proxy) => resulting explanation
    original_input:
        The original observation which will be taken into account when computing the fidelity of the mask

    Returns
    -------
    (np.ndarray, int, float)
        The explanation, the action predicted for that explanation and the fidelity of that explanation
    """
    if len(to_mask.shape) == 3:
        to_mask = to_mask[0]
    if len(original_input.shape) == 3:
        original_input = original_input[0]
    chosen_explanation, chosen_action, chosen_q_values, best_fidelity = None, None, None, np.inf
    q_values, _ = model_((tf.expand_dims(original_input, axis=0), ADJ_MATRIX_SPARSE))
    action = np.argmax(q_values[0])
    for (V_s, F_s, _) in masks:
        if len(V_s) == 0 or len(F_s) == 0:
            continue
        # Compute explanation
        mask = create_mask(V_s, F_s)
        explanation = tf.cast(to_mask * mask, tf.float32)

        # Compute model's response for given explanation
        mask_q_values, _ = model_((tf.expand_dims(explanation, axis=0), ADJ_MATRIX_SPARSE))
        mask_action = np.argmax(mask_q_values[0])
        mask_fidelity = mean_squared_error(q_values, mask_q_values)

        if chosen_action == action:
            # If currently chosen action is the target action, only update when mask action
            # also equals target action.
            if mask_action == action and mask_fidelity > best_fidelity:
                chosen_explanation = explanation
                chosen_action = mask_action
                best_fidelity = mask_fidelity
        elif mask_fidelity < best_fidelity:
            chosen_explanation = explanation
            chosen_action = mask_action
            best_fidelity = mask_fidelity

    return chosen_explanation, chosen_action, best_fidelity


def create_mask(V_s, F_s):
    """Convert the Zorro explanation to a matrix

    Parameters
    ----------
    V_s: np.ndarray
        Zorro's selected nodes (indices w.r.t. a feature matrix)
    F_s: np.ndarray
        Zorro's selected features (indices w.r.t. a feature matrix)

    Returns
    -------
    np.ndarray
        The mask as a matrix (2D array)
    """

    mask = np.zeros((AMT_NODES, FEATURE_DIM), dtype=np.int32)
    for node in V_s:
        for feature in F_s:
            mask[node, feature] = 1
    return mask


def compute_fidelity(gnn, X, A, V_s, F_s, samples=300):
    """Computes the fidelity (MSE between Q-values)

    Parameters
    ----------
    gnn: tf.keras.Model
        The graph neural network to analyse
    X: np.ndarray
        A feature matrix
    A: np.ndarray
        An adjacency matrix
    V_s: np.ndarray
        Zorro's selected nodes (indices w.r.t. X)
    F_s: np.ndarray
        Zorro's selected features (indices w.r.t. X)
    samples: int
        The amount of random matrices for the unselected nodes and features to sample

    Returns
    -------
    float
        The RDT-fidelity (V_s, F_s) described in [1]

    References
    ----------
    [1]:
        Funke, Thorben, Megha Khosla, and Avishek Anand.
        Zorro: Valid, sparse, and stable explanations in graph neural networks.
        arXiv preprint arXiv:2105.08621 (2021).
    """

    # Original prediction for X
    q_values, _ = gnn((X, A))
    q_values = q_values[0]
    action = np.argmax(q_values)

    # Noisy input memory
    noisy_input_mem = []

    # Create mask
    mask = create_mask(V_s, F_s)
    masked_features = mask * X

    for _ in range(samples):

        # Create random values
        negated_mask = np.logical_not(mask).astype(np.int32)
        random_values = np.random.uniform(size=(AMT_NODES, FEATURE_DIM))

        # Modify input
        modified_input = masked_features + (negated_mask * random_values)
        noisy_input_mem.append(modified_input)

    noisy_input_mem = np.concatenate(noisy_input_mem, axis=0)

    # Noisy prediction
    noisy_q_values, _ = gnn((noisy_input_mem, A))
    noisy_actions = np.argmax(noisy_q_values, axis=-1)

    # Comparison
    comparison = (action == noisy_actions).astype(np.int32)
    fidelity = np.sum(comparison) / samples

    return fidelity


def new_feature_fidelity(feature, V_s, F_s, gnn, X, A):
    """Wrapper to compute a new fidelity, given an additional feature in Zorro's explanation

    Parameters
    ----------
    feature: int
        The new feature that is added to F_s
    V_s: np.ndarray
        Zorro's previously selected nodes (indices w.r.t. X)
    F_s: np.ndarray
        Zorro's previously selected features (indices w.r.t. X)
    gnn: tf.keras.Model
        The graph neural network to analyse
    X: np.ndarray
        A feature matrix
    A: np.ndarray
        An adjacency matrix

    Returns
    -------
    float
        The fidelity of the explanation (V_s, F_s) when `feature` is added to F_s
    """
    F_s = np.concatenate([np.array([feature]), F_s])
    return compute_fidelity(gnn, X, A, V_s, F_s)


def new_node_fidelity(node, V_s, F_s, gnn, X, A):
    """Wrapper to compute a new fidelity, given an additional node in Zorro's explanation

    Parameters
    ----------
    node: int
        The new node that is added to V_s
    V_s: np.ndarray
        Zorro's previously selected nodes (indices w.r.t. X)
    F_s: np.ndarray
        Zorro's previously selected features (indices w.r.t. X)
    gnn: tf.keras.Model
        The graph neural network to analyse
    X: np.ndarray
        A feature matrix
    A: np.ndarray
        An adjacency matrix


    Returns
    -------
    float
        The fidelity of the explanation (V_s, F_s) when `node` is added to V_s
    """
    V_s = np.concatenate([np.array([node]), V_s])
    return compute_fidelity(gnn, X, A, V_s, F_s)


def mean_squared_error(q_original, q_noisy):
    """Mean squared error to compute fidelity

    Computes q-value-vector-wise difference, squares the difference
    and computes the mean deviation between two q-value vectors.

    Lastly it means over all mean-differences if multiple vector-pairs
    are given.

    Parameters
    ----------
    q_original: np.ndarray
        Q-values predicted for the original observation
    q_noisy: np.ndarray
        Q-values predicted for an explanation of the original observation

    Returns
    -------
    float
        The mean squared error between the original Q-values and the "noisy" Q-values
    """

    return np.square(q_original - q_noisy).mean(axis=-1).mean()


if __name__ == "__main__":
    # Load the agent
    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [256],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    model = load_agent("../double_q_learning/checkpoints/rl_agent", h_set)
    model.load_weights("../learn_proxies/checkpoints/test_set")
    X_ = np.random.normal(size=(1, AMT_NODES, FEATURE_DIM))
    V_s_, F_s_ = {1, 2, 3}, {2}
    V_p_, F_p_ = set(np.arange(AMT_NODES)), set(np.arange(FEATURE_DIM))

    ############################
    # Test fidelity computation
    ############################
    fid_1 = compute_fidelity(model, X_, ADJ_MATRIX_SPARSE, V_s_, F_s_)
    fid_2 = compute_fidelity(model, X_, ADJ_MATRIX_SPARSE, V_p_, F_p_)

    print(f"fid_selected: {fid_1}; fid_all: {fid_2}")
