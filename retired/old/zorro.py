from double_q_learning.preprocessing import FEATURE_DIM, ADJ_MATRIX_SPARSE, AMT_NODES
from double_q_learning.neural_networks import load_agent

from zorro_algorithm.discrete_mask import DiscreteMask

import tensorflow as tf
import numpy as np
import logging
import sys


def get_best_mask(masks):
    """Choose the best mask from all given masks

    :param masks: Masks to choose from
    :return: The mask with the highest fidelity
    """

    best_fidelity = 0.0
    chosen_mask = None
    for (V_s, F_s, mask_fidelity) in masks:
        if mask_fidelity > best_fidelity:
            chosen_mask = (V_s, F_s)

    assert chosen_mask is not None, "No mask found!"

    return chosen_mask


def form_mask(V_s, F_s):
    mask = np.zeros((AMT_NODES, FEATURE_DIM))
    for node in V_s:
        for feature in F_s:
            mask[node, feature] = 1
    return mask


def get_explanations(X: tf.Tensor,
                     A: tf.Tensor,
                     threshold_fidelity: float,
                     K: int,
                     V_p: set,
                     F_p: set,
                     gnn: tf.keras.Model):
    """Recursively compute explanations with a desired level of fidelity.

    One recursion returns at most one explanation. More explanations may result
    due to more recursion steps.

    :param X: The feature matrix, for which explanations (in form of discrete masks)
              shall be computed.
    :param A: The corresponding adjacency matrix for X.
    :param threshold_fidelity: A threshold fidelity, which shall not be exceeded by the fidelity
                            computed of an explanation.
    :param K: Determines the K best nodes or features, respectively, which are used to extend
              a given explanation.
    :param V_p: The set of nodes which can be included in an explanation.
    :param F_p: The set of features which can be included in an explanation.
    :param gnn: The neural network, at the hand of which explanations shall be computed.

    :return: A set of explanations (discrete masks) for the prediction of gnn(X, A)
    """

    # Set of explanations
    S = []
    graph_mask = DiscreteMask(V_p, F_p)

    # Determine first element in the explanation from the selectable nodes or features
    remaining_elems_left = graph_mask.init_mask(X, A, gnn)

    # Add new elements to the mask as long as the fidelity remains
    # "good enough", which is determined by the target fidelity.
    mask_fidelity = graph_mask.get_mask_fidelity(X, A, gnn)

    # As long as we produce a fidelity that is smaller than our threshold fidelity,
    # we add more values to the mask such that more original values retain their
    # values.
    while mask_fidelity <= threshold_fidelity and remaining_elems_left:  # (IV)
        sys.stdout.write(
            f"\rZORRO::: "
            f"Current mask V_s: {graph_mask.V_s} F_s: {graph_mask.F_s} - "
            f"Remaining V_r: {graph_mask.V_r} F_r: {graph_mask.F_r} - "
            f"Mask fidelity: {mask_fidelity:.3f}"
        )
        R_Fp, R_Vp = graph_mask.get_current_ranking(X, A, gnn)
        remaining_elems_left = graph_mask.add_element_to_mask(R_Vp, R_Fp)
        mask_fidelity = graph_mask.get_mask_fidelity(X, A, gnn)

    '''
    +-------------------------------------+--------------------------------------------------------+
    | mask_fidelity <= threshold_fidelity | remaining_elems_left                                   |
    | ------------------------------------+--------------------------------------------------------|
    |                 0                   |       0           : Return mask, no recursion     (I)  |
    |                 0                   |       1           : Return mask, recursion        (II) |
    |                 1                   |       0           : Return no mask, no recursion  (III)|
    |                 1                   |       1           : While loop                    (IV) |
    +----------------------------------------------------------------------------------------------+
    



    '''

    if mask_fidelity >= threshold_fidelity and not remaining_elems_left:  # (I)
        if len(graph_mask.V_s) + len(graph_mask.F_s) > 0:
            return [(graph_mask.V_s, graph_mask.F_s, mask_fidelity)]
        else:
            return []
    elif mask_fidelity <= threshold_fidelity and not remaining_elems_left:  # (III)
        return []
    else:  # (II)
        if len(graph_mask.V_s) + len(graph_mask.F_s) == 0:
            return []
        explanation = [(graph_mask.V_s, graph_mask.F_s, mask_fidelity)]
        S.extend(explanation)
        S.extend(get_explanations(
            X,
            A,
            threshold_fidelity,
            K,
            graph_mask.V_p,
            graph_mask.F_r,
            gnn
        ))
        S.extend(get_explanations(
            X,
            A,
            threshold_fidelity,
            K,
            graph_mask.V_r,
            graph_mask.F_p,
            gnn
        ))

    return S


def zorro(gnn, input_features, input_adj, target_fidelity, K):
    """Compute a set of disjoint explanations with a desired level of fidelity.

    An explanation is defined as the tuple:
    * V: Relevant nodes
    * F: Relevant features

    :param K:
    :param target_fidelity:
    :param query_node:
    :param input_adj:
    :param input_features:
    :param amt_convs: l_hop neighborhood
    :param gnn:
    :return: A set of explanations for the prediction of a given graph.
    """

    assert isinstance(input_features, tf.Tensor), "Use tf.Tensor for features matrix"
    assert isinstance(input_adj, tf.SparseTensor), "Use tf.SparseTensor for adjacency matrix"

    if tf.rank(input_features) < 3:
        input_features = tf.expand_dims(input_features, axis=0)

    ######################################################
    # DETERMINE QUERY NODE NEIGHBORHOOD AND NODE FEATURES (left out for now)
    ######################################################
    # neighborhood = determine_neighborhood(
    #     tf.sparse.to_dense(input_adj), tf.constant(query_node), tf.constant(amt_convs)
    # )
    # neighborhood = set(tf.reshape(neighborhood, (-1,)).numpy())
    # node_features = set(tf.range(tf.shape(input_features)[2]).numpy())

    shape = tf.shape(input_features)
    neighborhood = set(tf.range(shape[1]).numpy())
    node_features = set(tf.range(shape[2]).numpy())

    #######################
    # COMPUTE EXPLANATIONS
    #######################
    explanations = get_explanations(
        input_features,
        input_adj,
        target_fidelity,
        K,
        neighborhood,
        node_features,
        gnn
    )

    return explanations


# Sanity check
if __name__ == "__main__":
    # TODO: At some point in program execution, tensorflow retraces a lot. Fix it.
    tf.get_logger().setLevel(logging.ERROR)
    np.set_printoptions(threshold=5, precision=2)

    ####################################
    # Test l-hop neighbor determination
    ####################################
    # sample_adj = tf.constant([
    #     [0, 1, 1, 0, 0, 0],
    #     [1, 0, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 0]
    # ])
    # neighbors_ = determine_neighborhood(
    #     sample_adj, query_node=tf.constant(0), l_hop=tf.constant(3)
    # )
    # print(f"L-hop neighborhood {3}:", neighbors_)
    # neighbors_ = determine_neighborhood(
    #     sample_adj, query_node=tf.constant(0), l_hop=tf.constant(2)
    # )
    # print(f"L-hop neighborhood {2}:", neighbors_)
    # neighbors_ = determine_neighborhood(
    #     sample_adj, query_node=tf.constant(0), l_hop=tf.constant(1)
    # )
    # print(f"L-hop neighborhood {1}:", neighbors_)
    # neighbors_ = determine_neighborhood(
    #     sample_adj, query_node=tf.constant(0), l_hop=tf.constant(0)
    # )
    # print(f"L-hop neighborhood {0}:", neighbors_)

    #######################
    # Test fidelity metric
    #######################
    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [64],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    test_net = load_agent("../double_q_learning/checkpoints/rl_agent_4", h_set)
    test_net.load_weights(
        "../learn_explanations/checkpoints/transfer_learning_no_rounding_2/test_set_0"
    )
    # Zorro is a post-hoc instance based explanation algorithm. That is,
    # we compute explanations for one sample at a time after we've trained
    # the model.
    sample_input = tf.random.uniform((1, 25, FEATURE_DIM))
    # V_s_ = tf.constant([3, 4, 5])
    # F_s_ = tf.constant([1])
    # f_value_ = paper_fidelity(
    #     sample_input,
    #     ADJ_MATRIX_SPARSE,
    #     V_s_,
    #     F_s_,
    #     test_net
    # )
    # print(
    #     f"Computed fidelity value: {f_value_} for discrete mask given by "
    #     f"\n{V_s_.numpy(), F_s_.numpy()} and matrix \nX = \n{sample_input}"
    # )

    ########################
    # Test ranking function
    ########################
    # features_ranking_, nodes_ranking_ = create_ranking(
    #     nodes=tf.constant([1, 2, 3, 4], dtype=tf.int32),
    #     features=tf.constant([1, 2, 3, 4, 5], dtype=tf.int32),
    #     selected_nodes=tf.constant([5], dtype=tf.int32),
    #     selected_features=tf.constant([6], dtype=tf.int32),
    #     X=sample_input,
    #     A=ADJ_MATRIX_SPARSE,
    #     gnn=test_net
    # )
    # print(f"(unsorted) Features ranking:\n {features_ranking_}")
    # print(f"(unsorted) Nodes ranking:\n {nodes_ranking_}")

    # Test entire Zorro-algorithm
    result_ = zorro(test_net, sample_input, ADJ_MATRIX_SPARSE, .5, 10)
    print(f"\nDiscrete masks (output Zorro-algorithm): {result_}")
