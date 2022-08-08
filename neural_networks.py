from keras.layers import Activation
from spektral.layers import GeneralConv, DiffPool
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.optimizers import Adam

from preprocessing import AMT_NODES, FEATURE_DIM, ADJ_MATRIX_SPARSE

import tensorflow as tf


def proxy_branch(lr, graph_layers, amt_nodes=AMT_NODES, feature_dim=FEATURE_DIM):
    """Neural network that predicts proxies for given observations

    :param lr: Learning rate
    :param graph_layers: The shapes of the general graph convolutions applied to the input
    :param amt_nodes: The amount of nodes within a graph
    :param feature_dim: The amount of features per node
    :return: A configured proxy branch
    """

    node_features_in = Input(shape=(amt_nodes, feature_dim), name="feature_embedding")
    adj_matrix = Input(tensor=ADJ_MATRIX_SPARSE, name="adj_matrix")

    node_f = None
    for gl in graph_layers:
        input_ = node_f if node_f is not None else node_features_in
        node_f = GeneralConv(
            channels=gl, dropout=0.0, batch_norm=False, activation="prelu", aggregate="sum",
            name="gcn_25x128"
        )([input_, adj_matrix])

    proxy = GeneralConv(
        channels=FEATURE_DIM,
        dropout=0.0,
        batch_norm=False,
        activation="prelu",
        aggregate="sum",
        name=f"proxy_input_25x{FEATURE_DIM}"
    )([node_f, adj_matrix])

    proxy = Activation(activation="sigmoid", name="Sigmoid")(proxy)

    model = tf.keras.Model(
        inputs=[node_features_in, adj_matrix],
        outputs=[proxy]
    )

    model.compile(
        loss={"mse": tf.keras.losses.MeanSquaredError()},
        optimizer=Adam(learning_rate=lr),
    )

    return model


def deep_q_network(lr, graph_layers, amt_actions=6):
    """Deep q-network that simultaneously predicts its own inputs

    :param lr: Learning rate
    :param graph_layers: The shapes of the general graph convolutions applied to the input
    :param amt_actions: The amount of possible actions
    :return: A configured proxy branch and deep q-network
    """
    node_features_in = Input(shape=(AMT_NODES, FEATURE_DIM), name="Feature Matrix")
    adj_matrix_in = Input(tensor=ADJ_MATRIX_SPARSE, name="Adj. Matrix")

    #####################
    # COMPUTE EMBEDDING
    #####################
    node_f = GeneralConv(
        channels=256, dropout=0.0, batch_norm=False, activation="prelu", aggregate="sum"
    )([node_features_in, adj_matrix_in])
    for idx, gl in enumerate(graph_layers):
        if idx == len(graph_layers) - 1:
            node_f = GeneralConv(
                channels=gl,
                dropout=0.0,
                batch_norm=False,
                activation="prelu",
                aggregate="sum",
                name=f"feature_embedding"
            )([node_f, adj_matrix_in])
        else:
            node_f = GeneralConv(
                channels=gl,
                dropout=0.0,
                batch_norm=False,
                activation="prelu",
                aggregate="sum"
            )([node_f, adj_matrix_in])

    #####################
    # POOLING
    #####################
    node_f, adj_matrix = DiffPool(
        1, channels=amt_actions, activation="relu", name="pooling"
    )([node_f, adj_matrix_in])
    node_f = Reshape((amt_actions,))(node_f)

    #####################
    # Q-VALUE PREDICTION
    #####################
    q_values = Dense(amt_actions, activation="linear", name="prediction")(node_f)

    model = tf.keras.Model(
        inputs=[node_features_in, adj_matrix_in], outputs=[q_values]
    )
    model.compile(
        loss={"mse": tf.keras.losses.MeanSquaredError()},
        optimizer=Adam(learning_rate=lr)
    )

    return model


def load_agent(load_path, h_set):
    """Loads a trained agent and adds a proxy branch.

    Furthermore, it freezes the q-value prediction branch s.t. only the proxy branch will
    be trained.

    :param load_path: The path from where the base model will be loaded
    :param h_set: The hyperparameter set that fits to the base model and tells how many
                  additional graph convolutions shall be added
    :return: A neural network with added proxy branch
    """

    # Load trained agent
    base_model = deep_q_network(lr=None, graph_layers=h_set["graph_layers"])
    base_model.load_weights(load_path)

    # Pop everything from pooling on, s.t. the feature embedding is the new output
    feature_embedding = base_model.get_layer("feature_embedding").output
    q_values = base_model.get_layer("prediction").output
    base_model = tf.keras.Model(inputs=base_model.input, outputs=[q_values, feature_embedding])
    base_model.trainable = False

    # Initialize proxy branch
    proxy_b = proxy_branch(
        lr=h_set["learning_rate"],
        graph_layers=h_set["expl_graph_layers"],
        feature_dim=base_model.outputs[1].shape[2]
    )

    # Connect proxy branch on top of feature embedding
    node_features_in = tf.keras.Input(shape=(AMT_NODES, FEATURE_DIM))
    adj_in = Input(tensor=ADJ_MATRIX_SPARSE, name="Adj. Matrix")
    q_values, f_embedding = base_model((node_features_in, adj_in))
    proxy = proxy_b((f_embedding, adj_in))

    full_model = tf.keras.Model(inputs=[node_features_in, adj_in], outputs=[q_values, proxy])

    full_model.compile(
        loss={"mse": tf.keras.losses.MeanSquaredError()},
        optimizer=Adam(learning_rate=h_set["learning_rate"])
    )
    full_model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(".")
    tb_callback.set_model(full_model)

    return full_model


if __name__ == "__main__":

    #################
    # DEEP Q-NETWORK
    #################
    test_input = tf.random.normal((64, AMT_NODES, FEATURE_DIM))
    deep_q_net = deep_q_network(.001, [64], 6)
    deep_q_net.summary()
    tf.keras.utils.plot_model(deep_q_net, "./miscellaneous/deep_q_network.svg", dpi=None, rankdir="LR")
    q_values_ = deep_q_net((test_input, ADJ_MATRIX_SPARSE))
    print(f"Q-values shape: {tf.shape(q_values_)}")
    deep_q_net.save_weights("./miscellaneous/test_q_net")

    h_set_ = {
        "name": f"TEST",
        "learning_rate": .001,
        "batch_size": 64,
        "graph_layers": [64],
        "expl_graph_layers": [64],
        "dense_layers": [],
        "fidelity_reg": .001
    }

    ###############
    # PROXY BRANCH
    ###############
    feature_dim_ = 10  # model_1.get_layer("feature_embedding").output.shape[2]
    test_input_2 = tf.random.normal((64, AMT_NODES, feature_dim_))
    proxy_b_ = proxy_branch(
        lr=h_set_["learning_rate"],
        graph_layers=h_set_["expl_graph_layers"],
        feature_dim=feature_dim_
    )
    proxy_b_.summary()
    tf.keras.utils.plot_model(proxy_b_, "./miscellaneous/proxy_branch.svg", dpi=None, rankdir="LR")
    proxy_ = proxy_b_((test_input_2, ADJ_MATRIX_SPARSE))
    print(f"Proxy shape: {tf.shape(proxy_)}")

    ###################
    # COMBINED NETWORK
    ###################
    model_3 = load_agent("./miscellaneous/test_q_net", h_set_)
    model_3.summary()
    tf.keras.utils.plot_model(model_3, "./miscellaneous/full_net.svg", dpi=None, rankdir="LR")
    q_values_, proxy_ = model_3((test_input, ADJ_MATRIX_SPARSE))
    print(f"Q-values shape: {tf.shape(q_values_)}; Proxy shape: {tf.shape(proxy_)}")
