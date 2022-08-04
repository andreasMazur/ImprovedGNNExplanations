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
ADJ_MATRIX = tf.cast(ADJ_MATRIX, tf.float32)
ADJ_MATRIX_SPARSE = tf.sparse.from_dense(ADJ_MATRIX)
FEATURE_DIM = 4  # previously 10

#########################
# NODE FEATURE ENCODINGS
#########################
taxi_alone = [1, 0, 0, 0]
passenger_alone = [0, 1, 0, 0]
dest_alone = [0, 0, 1, 0]
passenger_taxi = [1, 1, 0, 0]
passenger_taxi_picked = [0, 0, 0, 1]
passenger_dest = [0, 1, 1, 0]
taxi_dest = [1, 0, 1, 0]
taxi_passenger_dest = [1, 1, 1, 0]
taxi_passenger_dest_picked = [0, 0, 1, 1]


def preprocess(env_, observation, return_tf_tensor=True):
    """Computes the node-features matrix.

    The node features are encoded in binary in a 4-dimensional feature vector [w,x,y,z]

    If - w is 1: Taxi is located on this node
       - x is 1: Passenger is located on this node
       - y is 1: Destination is located on this node
       - z is 1: (Needs w and x to be 1) Passenger is picked up by the taxi

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

    # Translate environment encoding of positions into node index
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
            node_features_matrix[taxi_loc] = taxi_passenger_dest_picked
        else:
            node_features_matrix[taxi_loc] = taxi_passenger_dest

    # Not all entities on one field
    else:
        node_features_matrix[taxi_loc] = taxi_alone

        # Passenger on the same field as taxi
        if pass_loc == taxi_loc:
            if pass_loc_raw == 4:
                node_features_matrix[pass_loc] = passenger_taxi_picked
            else:
                node_features_matrix[pass_loc] = passenger_taxi

        # Passenger on the same field as destination [Will never be the case]
        elif pass_loc == dest_idx:
            node_features_matrix[pass_loc] = passenger_dest

        # Passenger alone
        else:
            node_features_matrix[pass_loc] = passenger_alone

        # Destination on the same field as taxi
        if dest_idx == taxi_loc:
            node_features_matrix[dest_idx] = taxi_dest

        # Destination alone
        else:
            node_features_matrix[dest_idx] = dest_alone

    if return_tf_tensor:
        return tf.constant(node_features_matrix)
    else:
        return node_features_matrix


def colorize(graph):
    """Computes a color map for a graph."""
    color_map_ = []
    for node_ in graph.nodes.data():
        if np.array_equal(node_[1]["feature_vec"], passenger_taxi_picked):
            color_map_.append("green")
        elif np.array_equal(node_[1]["feature_vec"], taxi_passenger_dest_picked):
            color_map_.append("black")
        elif np.array_equal(node_[1]["feature_vec"], taxi_passenger_dest):
            color_map_.append("white")
        elif np.array_equal(node_[1]["feature_vec"], taxi_alone):
            color_map_.append("yellow")
        elif np.array_equal(node_[1]["feature_vec"], taxi_dest):
            color_map_.append("cyan")
        elif np.array_equal(node_[1]["feature_vec"], dest_alone):
            color_map_.append("purple")
        elif np.array_equal(node_[1]["feature_vec"], passenger_alone):
            color_map_.append("blue")
        elif np.array_equal(node_[1]["feature_vec"], passenger_taxi):
            color_map_.append("orange")
        elif np.array_equal(node_[1]["feature_vec"], [0, 0, 0, 0]):  # empty location
            color_map_.append("grey")
        else:
            color_map_.append("red")

    return color_map_


def draw_heat_graph(explanation, fid=None, action_=None, title=None, show=True):
    """Draws a heat graph representation for a given explanation.

    The color ('heat') of a node is determined by its L2-norm.

    :param explanation: Output of an explainer network (a continuous mask).
    :param fid: The fidelity of that explanation
    :param action_: Chosen action for this explanation
    :param title: The title of the plot
    :param show: Show the image
    """

    # Create graph
    nx_graph = nx.convert_matrix.from_numpy_matrix(ADJ_MATRIX.numpy())
    for graph_row in range(GRID_SIZE):
        for graph_col in range(GRID_SIZE):
            graph_node_idx = graph_row * 5 + graph_col
            nx_graph.nodes[graph_node_idx]["pos"] = (graph_col, -graph_row)

    # Compute heat-color
    if len(explanation.shape) == 3:
        explanation = explanation[0]
    if isinstance(explanation, tf.Tensor):
        explanation = explanation.numpy()
    # explanation = np.linalg.norm(explanation, axis=-1)

    # Normalize feature vectors
    divisor = np.linalg.norm(explanation, axis=-1)
    divisor[divisor == 0.] = 1.  # don't divide through zero
    explanation = (explanation.T / divisor).T
    # Clip alpha values in range [0.5, 1]
    explanation[:, 3] = 1/2 * explanation[:, 3] + 0.5  # alpha value

    # node_color = explanation[:, :3]  # nodes have always max alpha value
    # edge_color = np.ones_like(explanation) - explanation  # inverse color
    # edge_color[:, 3] = explanation[:, 3]  # alpha value for edges

    # Draw graph
    # cmap = plt.cm.Reds
    nx.draw_networkx(
        nx_graph,
        pos={node_key: node_attr["pos"] for node_key, node_attr in nx_graph.nodes.data()},
        node_color=explanation,
        # edgecolors=edge_color,
        font_color="white"
        # cmap=cmap
    )
    # sm = plt.cm.ScalarMappable(
    #     cmap=cmap, norm=plt.Normalize(vmin=explanation.min(), vmax=explanation.max())
    # )
    # plt.colorbar(sm)

    if fid is not None:
        plt.ylabel(f"Fidelity: {fid:.3f}")

    if action_ is not None:
        plt.xlabel(f"Action: {action_num_to_str(action_)}")

    if title is not None:
        plt.title(title)

    if show:
        plt.show()


def action_num_to_str(action_):
    """Converts a number to the string description of the corresponding action."""
    if action_ == 0:
        return "move south"
    elif action_ == 1:
        return "move north"
    elif action_ == 2:
        return "move east"
    elif action_ == 3:
        return "move west"
    elif action_ == 4:
        return "pickup passenger"
    else:
        return "drop off passenger"


def draw_discrete_graph(explanation, fid=None, action_=None, title=None, show=True):
    """Draws the graph representation for a given observation.

    :param explanation: Output of an explanation branch.
    :param fid: The fidelity of that explanation
    :param action_: Chosen action for this explanation
    :param title: The title of the plot.
    :param show: Show the image
    """

    if tf.rank(explanation) > 2:
        explanation = explanation[0]

    # Create graph
    nx_graph = nx.convert_matrix.from_numpy_matrix(ADJ_MATRIX.numpy())
    for graph_row in range(GRID_SIZE):
        for graph_col in range(GRID_SIZE):
            graph_node_idx = graph_row * 5 + graph_col
            nx_graph.nodes[graph_node_idx]["pos"] = (graph_col, -graph_row)
            nx_graph.nodes[graph_node_idx]["feature_vec"] = explanation[graph_node_idx]

    color_map_ = colorize(nx_graph)

    nx.draw_networkx(
        nx_graph,
        pos={node_key: node_attr["pos"] for node_key, node_attr in nx_graph.nodes.data()},
        node_color=color_map_
    )

    if fid is not None:
        plt.ylabel(f"Fidelity: {fid:.3f}")

    if action_ is not None:
        plt.xlabel(f"Action: {action_num_to_str(action_)}")

    if title is not None:
        plt.title(title)

    if show:
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

    color_map = colorize(G)

    nx.draw_networkx(
        G,
        pos={node_key: node_attr["pos"] for node_key, node_attr in G.nodes.data()},
        node_color=color_map
    )
    plt.savefig("SampleObservation.svg", format="svg")
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

        color_map = colorize(G)

        nx.draw_networkx(
            G,
            pos={node_key: node_attr["pos"] for node_key, node_attr in G.nodes.data()},
            node_color=color_map
        )
        plt.show()
