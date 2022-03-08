from zorro_algorithm.zorro_utils import create_ranking, fidelity

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

        # Remember old remaining nodes/features in case new mask exceeds
        # target fidelity
        self.old_V_r = None
        self.old_F_r = None

        # Remember old selected nodes/features in case new mask exceeds
        # target fidelity
        self.old_V_s = None
        self.old_F_s = None

    def get_current_ranking(self, K: int, X: tf.Tensor, A: tf.Tensor, gnn: tf.keras.Model):
        """Computes the ranking for remaining nodes/features with respect
           to selected nodes/features.

        :param K: Maximum amount of nodes within the ranking.
        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: node_ranking (not sorted), feature_ranking (not sorted)
        """

        R_Fp, R_Vp = create_ranking(
            tf.cast(tf.constant(list(self.V_r)), tf.int32),
            tf.cast(tf.constant(list(self.F_r)), tf.int32),
            tf.cast(tf.constant(list(self.V_s)), tf.int32),
            tf.cast(tf.constant(list(self.F_s)), tf.int32),
            X,
            A,
            gnn
        )

        R_Vp, R_Fp = R_Vp.numpy()[:K], R_Fp.numpy()[:K]

        # TODO: Check why 'vectorized_map' returns rank 1 tensor
        # TODO: for 1-element sets.
        if len(self.V_r) == 1 and len(R_Vp.shape) == 1:
            R_Vp = R_Vp.reshape((1, 1))

        if len(self.F_r) == 1 and len(R_Fp.shape) == 1:
            R_Fp = R_Fp.reshape((1, 1))

        return R_Vp, R_Fp

    def add_element_to_mask(self,
                            node_ranking: np.ndarray,
                            feature_ranking: np.ndarray,
                            X: tf.Tensor,
                            A: tf.Tensor,
                            gnn: tf.keras.Model):
        """Adds an element to the mask.

        :param node_ranking: Ranking of top K nodes. (not sorted)
        :param feature_ranking: Ranking of top K features. (not sorted)
        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: Whether or not remaining nodes/features are still available.
        """

        node_ranking_avail = node_ranking.shape[0] > 0
        if node_ranking_avail:
            best_node_to_add = node_ranking[np.argmin(node_ranking, axis=0)[1], 0]
            V_s_temp = self.V_s.copy()
            V_s_temp.add(int(best_node_to_add))
            changed_n_fid = fidelity(
                X,
                A,
                tf.cast(tf.constant(list(V_s_temp)), tf.int32),
                tf.cast(tf.constant(list(self.F_s)), tf.int32),
                gnn
            )

        feature_ranking_avail = feature_ranking.shape[0] > 0
        if feature_ranking_avail:
            best_feature_to_add = feature_ranking[np.argmin(feature_ranking, axis=0)[1], 0]
            F_s_temp = self.F_s.copy()
            F_s_temp.add(int(best_feature_to_add))

            changed_f_fid = fidelity(
                X,
                A,
                tf.cast(tf.constant(list(self.V_s)), tf.int32),
                tf.cast(tf.constant(list(F_s_temp)), tf.int32),
                gnn
            )

        # 0.) Remember old mask
        # 1.) Bigger fidelity is worse (in this implementation)
        # 2.) Add chosen element to mask
        # 3.) Remove chosen element from remaining nodes/features
        self.old_F_s = self.F_s.copy()
        self.old_V_s = self.V_s.copy()
        self.old_F_r = self.F_r.copy()
        self.old_V_r = self.V_r.copy()
        if node_ranking_avail and feature_ranking_avail:
            if changed_n_fid >= changed_f_fid:
                self.F_s = F_s_temp
                self.F_r.discard(best_feature_to_add)
            else:
                self.V_s = V_s_temp
                self.V_r.discard(best_node_to_add)
        elif node_ranking_avail:
            self.V_s = V_s_temp
            self.V_r.discard(best_node_to_add)
        elif feature_ranking_avail:
            self.F_s = F_s_temp
            self.F_r.discard(best_feature_to_add)
        else:
            raise RuntimeError("Neither node nor feature ranking given!")

        return len(self.V_r) + len(self.F_r) > 0

    def get_mask_fidelity(self, X: tf.Tensor, A: tf.Tensor, gnn: tf.keras.Model):
        """Computes the fidelity of the current mask

        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: Fidelity value (fidelity-) of the current mask (V_s/F_s)
        """
        mask_fidelity = fidelity(
            X,
            A,
            tf.cast(tf.constant(list(self.V_s)), tf.int32),
            tf.cast(tf.constant(list(self.F_s)), tf.int32),
            gnn
        )
        return mask_fidelity
