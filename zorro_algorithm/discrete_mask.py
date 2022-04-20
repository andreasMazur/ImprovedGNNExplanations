import numpy as np

from double_q_learning.neural_networks import load_agent
from double_q_learning.preprocessing import AMT_NODES, FEATURE_DIM, ADJ_MATRIX_SPARSE
from zorro_algorithm.zorro_utils import new_feature_fidelity, new_node_fidelity, compute_fidelity


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

        # Remember best mask
        self.best_V_p = None
        self.best_F_p = None
        self.best_fidelity = -np.inf

    def init_mask(self, X, A, gnn):
        """Initialization of the mask (adds the first element to the mask)

        The next element in the mask will either be:
            - a node if a feature has been selected as the first element
            - a feature if a node has been selected as the first element

        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        """

        nodes = np.array(list(self.V_p), dtype=np.int32)
        features = np.array(list(self.F_p), dtype=np.int32)

        # Compute initial feature ranking (considering all nodes as selected)
        features_ranking = []
        for feature in features:
            fid = new_feature_fidelity(
                feature, nodes, np.array([], dtype=np.int32), gnn, X, A
            )
            features_ranking.append((feature, fid))
        features_ranking = np.array(features_ranking)

        # Compute initial node ranking (considering all features as selected)
        nodes_ranking = []
        for node in nodes:
            fid = new_node_fidelity(
                node, np.array([], dtype=np.int32), features, gnn, X, A
            )
            nodes_ranking.append((node, fid))
        nodes_ranking = np.array(nodes_ranking)

        return self.add_element_to_mask(nodes_ranking, features_ranking)

    def add_element_to_mask(self, nodes_ranking, features_ranking):
        """Adds an element to the selected nodes and features.

        :param nodes_ranking: A ranking of nodes based on their fidelities .
        :param features_ranking: A ranking of features based on their fidelities.
        :return: True or false depending on whether there are more than one remaining elements
                 left.
        """

        if nodes_ranking.shape[0] == 0:
            all_fidelities_equal = np.all(features_ranking[:, 1] == features_ranking[0, 1])
            if all_fidelities_equal:
                random_idx = np.random.randint(features_ranking.shape[0])
                best_feature, best_feature_fid = features_ranking[random_idx]
            else:
                best_feature, best_feature_fid = features_ranking[
                    np.argmax(features_ranking, axis=0)[1]
                ]
            best_feature = best_feature.astype(np.int32)
            self.F_s.add(best_feature)
            self.F_r.discard(best_feature)
            return len(self.F_r) > 1 and len(self.V_s) > 0
        elif features_ranking.shape[0] == 0:
            all_fidelities_equal = np.all(nodes_ranking[:, 1] == nodes_ranking[0, 1])
            if all_fidelities_equal:
                random_idx = np.random.randint(nodes_ranking.shape[0])
                best_node, best_node_fid = nodes_ranking[random_idx]
            else:
                best_node, best_node_fid = nodes_ranking[np.argmax(nodes_ranking, axis=0)[1]]
            best_node = best_node.astype(np.int32)
            self.V_s.add(best_node)
            self.V_r.discard(best_node)
            return len(self.V_r) > 1 and len(self.F_s) > 0

        best_node, best_node_fid = nodes_ranking[np.argmax(nodes_ranking, axis=0)[1]]
        best_node = best_node.astype(np.int32)
        best_feature, best_feature_fid = features_ranking[np.argmax(features_ranking, axis=0)[1]]
        best_feature = best_feature.astype(np.int32)
        if best_node_fid <= best_feature_fid:
            self.F_s.add(best_feature)
            self.F_r.discard(best_feature)
        else:
            self.V_s.add(best_node)
            self.V_r.discard(best_node)

        return len(self.V_r) + len(self.F_r) > 1

    def compute_current_ranking(self, X, A, gnn):
        """Computes the current ranking for all remaining elements.

        The ranking is computed in terms of fidelity values when added to the current
        selected nodes and features.

        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: The ranking between all remaining nodes and features.
        """

        remaining_nodes = np.array(list(self.V_r), dtype=np.int32)
        remaining_features = np.array(list(self.F_r), dtype=np.int32)
        selected_nodes = np.array(list(self.V_s), dtype=np.int32)
        selected_features = np.array(list(self.F_s), dtype=np.int32)

        # Compute feature ranking
        if selected_nodes.shape[0] > 0:
            features_ranking = []
            for feature in remaining_features:
                fid = new_feature_fidelity(
                    feature, selected_nodes, selected_features, gnn, X, A
                )
                features_ranking.append((feature, fid))
            features_ranking = np.array(features_ranking)
        else:
            fidelities = np.full_like(remaining_features, -np.inf).reshape((-1, 1))
            remaining_features = remaining_features.reshape((-1, 1))
            features_ranking = np.concatenate([remaining_features, fidelities], axis=1)

        # Compute node ranking
        if selected_features.shape[0] > 0:
            nodes_ranking = []
            for node in remaining_nodes:
                fid = new_node_fidelity(
                    node, selected_nodes, selected_features, gnn, X, A
                )
                nodes_ranking.append((node, fid))
            nodes_ranking = np.array(nodes_ranking)
        else:
            fidelities = np.full_like(remaining_nodes, -np.inf).reshape((-1, 1))
            remaining_nodes = remaining_nodes.reshape((-1, 1))
            nodes_ranking = np.concatenate([remaining_nodes, fidelities], axis=1)

        return nodes_ranking, features_ranking

    def compute_mask_fidelity(self, X, A, gnn):
        """Computes the fidelity for the currently selected mask.

        :param X: The original feature matrix for which an explanation shall be computed.
        :param A: The corresponding adjacency matrix.
        :param gnn: The graph neural network to explain.
        :return: The fidelity of the currently selected mask.
        """

        selected_nodes = np.array(list(self.V_s), dtype=np.int32)
        selected_features = np.array(list(self.F_s), dtype=np.int32)
        mask_fidelity = compute_fidelity(gnn, X, A, selected_nodes, selected_features)

        if mask_fidelity > self.best_fidelity:
            self.best_V_p = self.V_s.copy()
            self.best_F_p = self.F_s.copy()
            self.best_fidelity = mask_fidelity

        return mask_fidelity


if __name__ == "__main__":
    # Load the agent
    h_set = {
        "learning_rate": [.001],
        "batch_size": [64],
        "graph_layers": [64],  # depends on what model you retrain
        "expl_graph_layers": [128],
        "fidelity_reg": [.001]
    }
    model = load_agent("../double_q_learning/checkpoints/rl_agent_4", h_set)
    model.load_weights(
        "../learn_explanations/checkpoints/transfer_learning_no_rounding_2/test_set_0"
    )
    X_ = np.random.normal(size=(1, AMT_NODES, FEATURE_DIM))
    V_s_, F_s_ = {1, 2, 3}, {2}
    V_p_, F_p_ = set(np.arange(AMT_NODES)), set(np.arange(FEATURE_DIM))

    ############################
    # Test mask initialization
    ############################
    mask = DiscreteMask(V_p_, F_p_)
    ok_1 = mask.init_mask(X_, ADJ_MATRIX_SPARSE, model)

    ############################
    # Test computing ranking
    ############################
    nr, fr = mask.compute_current_ranking(X_, ADJ_MATRIX_SPARSE, model)

    ############################
    # Test adding to mask
    ############################
    ok_2 = mask.add_element_to_mask(nr, fr)
