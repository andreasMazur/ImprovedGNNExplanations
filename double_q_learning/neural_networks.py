from tensorflow.keras.layers import Layer, Dense, Input, Reshape
from tensorflow.keras.optimizers import Adam
from spektral.layers import GeneralConv, DiffPool

from double_q_learning.preprocessing import AMT_NODES, FEATURE_DIM

import tensorflow as tf


class PreprocessAdj(Layer):
    """Layer that takes care about shape and data type of the adjacency matrix"""

    def call(self, inputs, *args, **kwargs):
        if tf.keras.backend.ndim(inputs) > 2:
            inputs = inputs[0]
        inputs = tf.cast(inputs, tf.float32)
        return tf.sparse.from_dense(inputs)


class Rounding(Layer):
    """Layer that rounds numbers"""

    def call(self, inputs, *args, **kwargs):
        return tf.math.round(inputs)


def deep_q_network(lr, graph_layers, dense_layers, amt_actions=6):
    """The deep-Q-network used to train the RL-agent.

    :param lr: Learning rate
    :param graph_layers: The shapes of the general graph convolutions applied to the input
    :param dense_layers: The shapes of the dense layers used to classify the graph embedding
    :param amt_actions: The amount of possible actions
    :return: A configured deep-Q-network
    """
    node_features_in = Input(shape=(AMT_NODES, FEATURE_DIM))
    adj_matrix_in = Input(shape=(AMT_NODES, AMT_NODES), dtype=tf.int32)
    adj_matrix = PreprocessAdj()(adj_matrix_in)

    node_f = GeneralConv(
        channels=256, dropout=0.0, batch_norm=False, activation='prelu', aggregate='sum'
    )([node_features_in, adj_matrix])
    for gl in graph_layers:
        node_f = GeneralConv(
            channels=gl, dropout=0.0, batch_norm=True, activation='prelu', aggregate='sum'
        )([node_f, adj_matrix])

    node_f, adj_matrix = DiffPool(1, channels=amt_actions, activation="relu")([node_f, adj_matrix])
    node_f = Reshape((amt_actions,))(node_f)

    for d in dense_layers:
        node_f = Dense(d, activation="relu")(node_f)
    q_values = Dense(amt_actions, "linear")(node_f)

    model = tf.keras.Model(inputs=[node_features_in, adj_matrix_in], outputs=q_values)
    model.compile(
        loss={"q_values": tf.keras.losses.MeanSquaredError()},
        optimizer=Adam(learning_rate=lr)
    )

    return model


def explainer_network(lr, graph_layers, amt_nodes=AMT_NODES, feature_dim=FEATURE_DIM):
    """Neural network that predicts explanations for given observations

    :param lr: Learning rate
    :param graph_layers: The shapes of the general graph convolutions applied to the input
    :param amt_nodes: The amount of nodes within a graph
    :param feature_dim: The amount of features per node
    :return: A configured explainer network
    """

    node_features_in = Input(shape=(amt_nodes, feature_dim), name="features_in")
    adj_matrix_in = Input(shape=(amt_nodes, amt_nodes), name="adj_matrix_in", dtype=tf.int32)
    adj_matrix = PreprocessAdj()(adj_matrix_in)

    node_f = GeneralConv(
        channels=256, dropout=0.0, batch_norm=False, activation="prelu", aggregate="sum"
    )([node_features_in, adj_matrix])
    for gl in graph_layers:
        node_f = GeneralConv(
            channels=gl, dropout=0.0, batch_norm=True, activation="prelu", aggregate="sum"
        )([node_f, adj_matrix])

    # Compute mask
    mask = GeneralConv(
        channels=feature_dim,
        dropout=0.0,
        batch_norm=True,
        activation="softmax",
        aggregate="sum",
        name="mask"
    )([node_f, adj_matrix])

    model = tf.keras.Model(
        inputs=[node_features_in, adj_matrix_in],
        outputs=[mask]
    )

    model.compile(
        loss={"masked_q_values_loss": tf.keras.losses.MeanSquaredError()},
        optimizer=Adam(learning_rate=lr),
    )

    return model
