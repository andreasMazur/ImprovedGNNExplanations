from matplotlib import pyplot as plt

import gym
import numpy as np
import networkx as nx
import tensorflow as tf


# Model the grid structure of the Taxi-V3 environment with an adjacency matrix
GRID_SIZE = 5
AMT_NODES = GRID_SIZE ** 2
ADJ_MATRIX = np.zeros((AMT_NODES, AMT_NODES))
exceptions = [{1, 2}, {6, 7}, {15, 16}, {20, 21}, {17, 18}, {22, 23}]
for node_to in range(ADJ_MATRIX.shape[0]):
    for node_from in range(ADJ_MATRIX.shape[1]):
        if node_to == node_from or {node_to, node_from} in exceptions:
            continue

        # Decode node-index to node position
        grid_col_to = node_to % 5
        grid_row_to = (node_to - grid_col_to) / 5
        grid_col_from = node_from % 5
        grid_row_from = (node_from - grid_col_from) / 5

        if grid_row_to == grid_row_from and abs(grid_col_to - grid_col_from) == 1:
            ADJ_MATRIX[node_to, node_from] = 1
        elif grid_col_to == grid_col_from and abs(grid_row_to - grid_row_from) == 1:
            ADJ_MATRIX[node_to, node_from] = 1
ADJ_MATRIX = tf.constant(ADJ_MATRIX)
FEATURE_DIM = 9


def preprocess(env_, observation, return_tf_tensor=True):
    """Computes the node-features matrix.

    The node-features matrix contains one hot vectors (see implementation).

    Note: The size of the grid and the obstacles within never change.
          That is, we only need to compute the node-features matrix for each
          observation.

    :param env_: The Taxi-V3 environment.
    :param observation: The observation that shall be converted into a graph.
    :param return_tf_tensor: If False, numpy array is returned instead.
    :return: The node-features matrix for the given observation.
    """

    taxi_row, taxi_col, pass_loc_raw, dest_idx = env_.decode(observation)
    taxi_loc = taxi_row * 5 + taxi_col

    if pass_loc_raw == 0:  # Red (0, 0)
        pass_loc = 0
    elif pass_loc_raw == 1:  # Green (0, 4)
        pass_loc = 4
    elif pass_loc_raw == 2:  # Yellow (4, 0)
        pass_loc = 20
    elif pass_loc_raw == 3:  # Blue (4, 3)
        pass_loc = 23
    else:  # In taxi
        pass_loc = taxi_loc

    if dest_idx == 0:  # Red
        dest_idx = 0
    elif dest_idx == 1:  # Green
        dest_idx = 4
    elif dest_idx == 2:  # Yellow
        dest_idx = 20
    else:  # Blue
        dest_idx = 23

    node_features_matrix = np.zeros((AMT_NODES, FEATURE_DIM))

    # All entities on one field
    if taxi_loc == pass_loc == dest_idx:
        if pass_loc_raw == 4:
            node_features_matrix[taxi_loc] = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        else:
            node_features_matrix[taxi_loc] = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    # Not all entities on one field
    else:
        node_features_matrix[taxi_loc] = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        # Passenger on the same field as taxi
        if pass_loc == taxi_loc:
            if pass_loc_raw == 4:
                node_features_matrix[pass_loc] = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            else:
                node_features_matrix[pass_loc] = [0, 0, 0, 0, 1, 0, 0, 0, 0]

        # Passenger on the same field as destination [Will never be the case]
        elif pass_loc == dest_idx:
            node_features_matrix[pass_loc] = [0, 0, 1, 0, 0, 0, 0, 0, 0]

        # Passenger alone
        else:
            node_features_matrix[pass_loc] = [0, 0, 0, 0, 0, 0, 0, 1, 0]

        # Destination on the same field as taxi
        if dest_idx == taxi_loc:
            node_features_matrix[dest_idx] = [0, 0, 0, 1, 0, 0, 0, 0, 0]

        # Destination alone
        else:
            node_features_matrix[dest_idx] = [0, 0, 0, 0, 0, 0, 1, 0, 0]

    if return_tf_tensor:
        return tf.constant(node_features_matrix)
    else:
        return node_features_matrix


def draw_heat_graph(mask_features):
    """Draws a heat graph representation for a given mask from an explainer.

    NOTE: The coloring only allows to compare the relative importance of
          nodes within a given graph. It does not allow to compare the representation
          of multiple masks as the normalization is done w.r.t. the maximal
          value of the given mask.

    :param mask_features: Output of an explainer network (a continuous mask).
    :return:
    """

    # Create graph
    nx_graph = nx.convert_matrix.from_numpy_matrix(ADJ_MATRIX.numpy())
    for graph_row in range(GRID_SIZE):
        for graph_col in range(GRID_SIZE):
            graph_node_idx = graph_row * 5 + graph_col
            G.nodes[graph_node_idx]["pos"] = (graph_col, -graph_row)
            G.nodes[graph_node_idx]["feature_vec"] = node_features[graph_node_idx]

    # Compute heat-color
    if isinstance(mask_features, tf.Tensor):
        mask_features = mask_features.numpy()
    mask_features = mask_features.sum(axis=1)
    mask_features /= mask_features.max()

    # Draw graph
    nx.draw_networkx(
        nx_graph,
        pos={node_key: node_attr["pos"] for node_key, node_attr in nx_graph.nodes.data()},
        node_color=mask_features,
        cmap=plt.cm.Reds
    )
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(linewidth=80)

    env = gym.make("Taxi-v3")
    obs = env.reset()
    env.render()
    node_features = preprocess(env, obs, False)
    stop = False

    G = nx.convert_matrix.from_numpy_matrix(ADJ_MATRIX.numpy())
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            node_idx = row * 5 + col
            G.nodes[node_idx]["pos"] = (col, -row)
            G.nodes[node_idx]["feature_vec"] = node_features[node_idx]

    color_map = []
    for node in G.nodes.data():
        if np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 1, 0, 0, 0]):  # Pass on taxi picked
            color_map.append("green")
        elif np.array_equal(node[1]["feature_vec"], [0, 1, 0, 0, 0, 0, 0, 0, 0]):  # Pass on dest not dropped
            color_map.append("black")
        elif np.array_equal(node[1]["feature_vec"], [1, 0, 0, 0, 0, 0, 0, 0, 0]):  # Pass on dest dropped
            color_map.append("white")
        elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 0, 0, 0, 1]):  # taxi_loc
            color_map.append("yellow")
        elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 1, 0, 0, 0, 0, 0]):  # Taxi on dest (w/o pass)
            color_map.append("red")
        elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 0, 1, 0, 0]):  # dest_loc
            color_map.append("purple")
        elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 0, 0, 1, 0]):  # pass_loc
            color_map.append("blue")
        elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 1, 0, 0, 0, 0]):  # Pass on taxi not picked
            color_map.append("orange")
        else:  # empty node
            color_map.append("grey")

    nx.draw_networkx(
        G,
        pos={node_key: node_attr["pos"] for node_key, node_attr in G.nodes.data()},
        node_color=color_map
    )
    plt.show()

    while not stop:
        action = int(input(
            "Select action: {0: move south, 1: move north, 2: move east, 3: move west, 4: pickup, 5: drop off}"
        ))
        obs, reward, done, info = env.step(action)
        env.render()
        node_features = preprocess(env, obs, False)

        G = nx.convert_matrix.from_numpy_matrix(ADJ_MATRIX.numpy())
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                node_idx = row * 5 + col
                G.nodes[node_idx]["pos"] = (col, -row)
                G.nodes[node_idx]["feature_vec"] = node_features[node_idx]

        color_map = []
        for node in G.nodes.data():
            if np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 1, 0, 0, 0]):  # Pass on taxi picked
                color_map.append("green")
            elif np.array_equal(node[1]["feature_vec"], [0, 1, 0, 0, 0, 0, 0, 0, 0]):  # Pass on dest not dropped
                color_map.append("black")
            elif np.array_equal(node[1]["feature_vec"], [1, 0, 0, 0, 0, 0, 0, 0, 0]):  # Pass on dest dropped
                color_map.append("white")
            elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 0, 0, 0, 1]):  # taxi_loc
                color_map.append("yellow")
            elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 1, 0, 0, 0, 0, 0]):  # Taxi on dest (w/o pass)
                color_map.append("red")
            elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 0, 1, 0, 0]):  # dest_loc
                color_map.append("purple")
            elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 0, 0, 0, 1, 0]):  # pass_loc
                color_map.append("blue")
            elif np.array_equal(node[1]["feature_vec"], [0, 0, 0, 0, 1, 0, 0, 0, 0]):  # Pass on taxi not picked
                color_map.append("orange")
            else:  # empty node
                color_map.append("grey")

        nx.draw_networkx(
            G,
            pos={node_key: node_attr["pos"] for node_key, node_attr in G.nodes.data()},
            node_color=color_map
        )
        plt.show()
