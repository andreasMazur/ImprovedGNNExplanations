from zorro_algorithm.zorro_utils import create_ranking, compute_feature_fidelities, \
    compute_node_fidelities, paper_fidelity

import tensorflow as tf
import numpy as np


class DiscreteMask:

    def __init__(self, V_p: set, F_p: set):
        """Data structure to manage all sets needed for Zorro.

        :param V_p: The set of nodes which can be included in an explanation.
        :param F_p: The set of features which can be included in an explanation.
        """

        self.V_p = V_p
        self.F_p = F_p

        # Remaining nodes/features to choose from
        self.V_r = V_p.copy()
        self.F_r = F_p.copy()

        # Selected nodes/features for mask creation
        self.V_s = set()
        self.F_s = set()

    def init_mask(self, X: tf.Tensor, A: tf.SparseTensor, gnn: tf.keras.Model):
        """Initialization of the mask (adds the first element to the mask)

        The next element in the mask will either be:
            - a node if a feature has been selected as the first element
            - a feature if a node has been selected as the first element

        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        """

        # If there are no more possible nodes or features, then return and
        # stop recursion.
        if len(self.V_p) == 0 or len(self.F_p) == 0:
            return False

        nodes = tf.cast(tf.constant(list(self.V_p)), tf.int32)
        features = tf.cast(tf.constant(list(self.F_p)), tf.int32)

        cff_partial = lambda x: compute_feature_fidelities(
            X, A, x, tf.constant([], dtype=tf.int32), nodes, gnn
        )
        features_ranking = tf.map_fn(cff_partial, tf.cast(features, tf.float32))

        cnf_partial = lambda x: compute_node_fidelities(
            X, A, x, tf.constant([], dtype=tf.int32), features, gnn
        )
        nodes_ranking = tf.map_fn(cnf_partial, tf.cast(nodes, tf.float32))

        return self.add_element_to_mask(nodes_ranking, features_ranking)

    def add_element_to_mask(self, nodes_ranking, features_ranking):
        """

        :param nodes_ranking:
        :param features_ranking:
        :param X:
        :param A:
        :param gnn:
        :return:
        """

        best_node, best_node_fid = nodes_ranking[np.argmax(nodes_ranking, axis=0)[1]]
        best_node = tf.cast(best_node, tf.int32).numpy()
        best_feature, best_feature_fid = features_ranking[np.argmax(features_ranking, axis=0)[1]]
        best_feature = tf.cast(best_feature, tf.int32).numpy()
        if best_node_fid <= best_feature_fid:
            self.F_s.add(best_feature)
            self.F_r.discard(best_feature)
        else:
            self.V_s.add(best_node)
            self.V_r.discard(best_node)

        return len(self.V_r) + len(self.F_r) > 0

    def get_current_ranking(self, X: tf.Tensor, A: tf.SparseTensor, gnn: tf.keras.Model):
        """Computes the ranking for remaining nodes/features with respect
           to selected nodes/features.

        :param K: Maximum amount of nodes within the ranking.
        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: node_ranking (not sorted), feature_ranking (not sorted)
        """

        remaining_nodes = tf.cast(tf.constant(list(self.V_r)), tf.int32)
        remaining_features = tf.cast(tf.constant(list(self.F_r)), tf.int32)
        selected_nodes = tf.cast(tf.constant(list(self.V_s)), tf.int32)
        selected_features = tf.cast(tf.constant(list(self.F_s)), tf.int32)

        features_ranking, nodes_ranking = create_ranking(
            remaining_nodes,
            remaining_features,
            selected_nodes,
            selected_features,
            X, A, gnn
        )

        return features_ranking, nodes_ranking

    def get_mask_fidelity(self, X: tf.Tensor, A: tf.Tensor, gnn: tf.keras.Model):
        """Computes the fidelity of the current mask

        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: Fidelity value of the current mask (V_s/F_s)
        """
        selected_nodes = tf.cast(tf.constant(list(self.V_s)), tf.int32)
        selected_features = tf.cast(tf.constant(list(self.F_s)), tf.int32)
        mask_fidelity = paper_fidelity(
            X, A, selected_nodes, selected_features, gnn
        )
        return mask_fidelity
