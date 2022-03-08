from spektral.layers import MessagePassing
from spektral.layers.pooling.pool import Pool

from expl_assisted_dql.preprocessing import ADJ_MATRIX, FEATURE_DIM
from retired.q_network import gen_q_network_2

from zorro_algorithm.discrete_mask import DiscreteMask
from zorro_algorithm.zorro_utils import determine_neighborhood, fidelity, create_ranking

import tensorflow as tf
import numpy as np
import logging
import sys


def get_explanations(X: tf.Tensor,
                     A: tf.Tensor,
                     target_fidelity: float,
                     K: int,
                     V_p: set,
                     F_p: set,
                     gnn: tf.keras.Model,
                     max_recursions: int = 5,
                     recursion_step: int = 0):
    """Recursively compute explanations with a desired level of fidelity.

    One recursion returns at most one explanation. More explanations may result
    due to more recursion steps.

    :param X: The feature matrix, for which explanations (in form of discrete masks)
              shall be computed.
    :param A: The corresponding adjacency matrix for X.
    :param target_fidelity: A threshold fidelity, which shall not be exceeded by the fidelity
                            computed of an explanation.
    :param K: Determines the K best nodes or features, respectively, which are used to extend
              a given explanation.
    :param V_p: The set of nodes which can be included in an explanation.
    :param F_p: The set of features which can be included in an explanation.
    :param gnn: The neural network, at the hand of which explanations shall be computed.
    :param max_recursions: Maximal amount of recursions (and therefore max. amount of explanations)
    :param recursion_step: The amount of recursion steps so far.

    :return: A set of explanations (discrete masks) for the prediction of gnn(X, A)
    """

    # Return if maximal amount of recursions is reached
    if max_recursions <= recursion_step:
        return []

    # Set of explanations
    S = []
    graph_mask = DiscreteMask(V_p, F_p)

    # Determine ranking between nodes and features
    R_Vp, R_Fp = graph_mask.get_current_ranking(K, X, A, gnn)

    # Determine first element in the explanation from the selectable nodes or features
    remaining_elems_left = graph_mask.add_element_to_mask(R_Vp, R_Fp, X, A, gnn)

    # Add new elements to the mask as long as the fidelity remains
    # "good enough", which is determined by the target fidelity.
    mask_fidelity = graph_mask.get_mask_fidelity(X, A, gnn)

    # As long as we produce a fidelity that is bigger than our target fidelity,
    # we add more values to the mask such that more original values retain their
    # values (s.t. fidelity improves, thus reduces).
    # NOTE: We use a different fidelity measure than in the original Zorro-algorithm.
    while mask_fidelity >= target_fidelity and remaining_elems_left:  # (IV)
        sys.stdout.write(
            f"\rZORRO::: Recursion step: {recursion_step} - "
            f"Current mask V_s: {graph_mask.V_s} F_s: {graph_mask.F_s} - "
            f"Remaining V_r: {graph_mask.V_r} F_r: {graph_mask.F_r} - "
            f"Mask fidelity: {mask_fidelity:.3f}"
        )
        R_Vp, R_Fp = graph_mask.get_current_ranking(K, X, A, gnn)
        remaining_elems_left = graph_mask.add_element_to_mask(R_Vp, R_Fp, X, A, gnn)
        mask_fidelity = graph_mask.get_mask_fidelity(X, A, gnn)

    '''
    +----------------------------------------------------------------------------------------------+
    | mask_fidelity >= target_fidelity | remaining_elems_left                                      |
    | ---------------------------------+---------------------                                      |
    |                 0                |          0           : Return old mask, no recursion (I)  |
    |                 0                |          1           : Return old mask, recursion    (II) |
    |                 1                |          0           : Return new mask, no recursion (III)|
    |                 1                |          1           : While loop                    (IV) |
    +----------------------------------------------------------------------------------------------+
    
    Additionally, keep in mind that the last computed mask may have already 
    exceeded the target fidelity. If so, do not add it to the set of possible
    explanations. Instead, add the previous mask.  
    '''

    if mask_fidelity > target_fidelity and not remaining_elems_left:  # (I)
        if len(graph_mask.old_V_s) + len(graph_mask.old_F_s) > 0:
            return [(graph_mask.old_V_s, graph_mask.old_F_s)]
        else:
            return []
    elif mask_fidelity <= target_fidelity and not remaining_elems_left:  # (III)
        if len(graph_mask.V_s) + len(graph_mask.F_s) > 0:
            return [(graph_mask.V_s, graph_mask.F_s)]
        else:
            return []
    else:  # (II)
        explanation = [(graph_mask.old_V_s, graph_mask.old_F_s)]
        S.extend(explanation)
        S.extend(get_explanations(
            X,
            A,
            target_fidelity,
            K,
            graph_mask.V_p,
            graph_mask.old_F_r,
            gnn,
            max_recursions,
            recursion_step + 1
        ))
        S.extend(get_explanations(
            X,
            A,
            target_fidelity,
            K,
            graph_mask.old_V_r,
            graph_mask.F_p,
            gnn,
            max_recursions,
            recursion_step + 2
        ))

    return S


def zorro(gnn, input_features, input_adj, query_node, target_fidelity, K, max_recursions=5):
    """Compute a set of disjoint explanations with a desired level of fidelity.

    An explanation is defined as the the tuple:
    * V: Relevant nodes
    * F: Relevant features

    :return: A set of explanations for the prediction of a given graph.
    """

    assert isinstance(input_features, tf.Tensor), "Use tf.Tensor for features matrix"
    assert isinstance(input_adj, tf.Tensor), "Use tf.Tensor for adjacency matrix"

    if tf.rank(input_features) < 3:
        input_features = tf.expand_dims(input_features, axis=0)

    #############################################
    # DETERMINE L-HOP NEIGHBORHOOD OF QUERY_NODE
    #############################################
    l_hop = 0
    no_more_gconvs = False
    for layer in gnn.layers:

        # Skip all layers before the first convolution

        # Amount of message passing layers determines the L-hop neighborhood of the query node
        is_gconv = issubclass(type(layer), MessagePassing)
        if is_gconv:
            l_hop += 1

        # Check for validity of network architecture
        if not no_more_gconvs:
            no_more_gconvs = issubclass(type(layer), Pool)
        if no_more_gconvs and is_gconv:
            raise ValueError(
                "Cannot determine L-hop neighborhood of query node after pooling. Do not apply "
                "another message passing layer after pooling the node embeddings."
            )

    ######################################################
    # DETERMINE QUERY NODE NEIGHBORHOOD AND NODE FEATURES
    ######################################################
    neighborhood = determine_neighborhood(input_adj, tf.constant(query_node), tf.constant(l_hop))
    neighborhood = set(tf.reshape(neighborhood, (-1,)).numpy())
    node_features = set(tf.range(tf.shape(input_features)[2]).numpy())

    #######################
    # COMPUTE EXPLANATIONS
    #######################
    if tf.rank(input_adj) < 3:
        input_adj = tf.expand_dims(input_adj, axis=0)
    explanations = get_explanations(
        input_features,
        input_adj,
        target_fidelity,
        K,
        neighborhood,
        node_features,
        gnn,
        max_recursions
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
    sample_adj = tf.constant([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0]
    ])
    neighbors_ = determine_neighborhood(
        sample_adj, query_node=tf.constant(0), l_hop=tf.constant(3)
    )
    print(f"L-hop neighborhood {3}:", neighbors_)
    neighbors_ = determine_neighborhood(
        sample_adj, query_node=tf.constant(0), l_hop=tf.constant(2)
    )
    print(f"L-hop neighborhood {2}:", neighbors_)
    neighbors_ = determine_neighborhood(
        sample_adj, query_node=tf.constant(0), l_hop=tf.constant(1)
    )
    print(f"L-hop neighborhood {1}:", neighbors_)
    neighbors_ = determine_neighborhood(
        sample_adj, query_node=tf.constant(0), l_hop=tf.constant(0)
    )
    print(f"L-hop neighborhood {0}:", neighbors_)

    #######################
    # Test fidelity metric
    #######################
    test_net = gen_q_network_2(.001, [], [])
    test_net.summary()
    # Zorro is a post-hoc instance based explanation algorithm. That is,
    # we compute explanations for one sample at a time after we've trained
    # the model.
    sample_input = tf.random.uniform((1, 25, FEATURE_DIM))
    V_s_ = tf.constant([3, 4, 5])
    F_s_ = tf.constant([1])
    f_value_ = fidelity(
        sample_input,
        tf.expand_dims(ADJ_MATRIX, axis=0),
        V_s_,
        F_s_,
        test_net
    )
    print(
        f"Computed fidelity value: {f_value_} for discrete mask given by "
        f"\n{V_s_.numpy(), F_s_.numpy()} and matrix \nX = \n{sample_input}"
    )

    ########################
    # Test ranking function
    ########################
    features_ranking_, nodes_ranking_ = create_ranking(
        nodes=tf.constant([1, 2, 3, 4], dtype=tf.int32),
        features=tf.constant([1, 2, 3, 4, 5], dtype=tf.int32),
        selected_nodes=tf.constant([5], dtype=tf.int32),
        selected_features=tf.constant([6], dtype=tf.int32),
        X=sample_input,
        A=tf.expand_dims(ADJ_MATRIX, axis=0),
        gnn=test_net
    )
    print(f"(unsorted) Features ranking:\n {features_ranking_}")
    print(f"(unsorted) Nodes ranking:\n {nodes_ranking_}")

    # Test entire Zorro-algorithm
    result_ = zorro(test_net, sample_input, ADJ_MATRIX, 13, .01, 10)
    print(f"\nDiscrete mask (output Zorro-algorithm): {result_}")
