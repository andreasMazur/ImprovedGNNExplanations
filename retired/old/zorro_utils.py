import tensorflow as tf


# @tf.function
def vectorized_map_wrapper(function, values: tf.Tensor):
    """Function that ensures that values have a proper shape.

    :param function: The function to map over each value of 'values'.
    :param values: A tensor containing the input values for 'function'.
    :return: A new tensor in which each entry was mapped to a new value by 'function'.
    """
    # 'vectorized_map' assumes the outer tensor to contain tensors of values (not constants!)
    # Thus, each element we want to map has to be of at least rank 1.
    values_rank = tf.rank(values)
    if tf.math.less(values_rank, tf.constant(2)):
        values = tf.expand_dims(values, axis=1)

    # If there are no values to map, the map can be buggy.
    # Thus, just return an empty result tensor.
    values_shape = tf.shape(values)
    if tf.math.equal(values_shape[0], tf.constant(0)):
        dummy_input = tf.random.normal(values_shape[1:])
        dummy_input = tf.expand_dims(dummy_input, axis=0)
        dummy_input = tf.cast(dummy_input, values.dtype)
        dummy_output = tf.vectorized_map(function, dummy_input)
        output_shape = tf.shape(dummy_output)
        # The output shape should capture that we have no result elements
        # but also the shape of a result element.
        output_shape = tf.concat([tf.constant([0]), output_shape[1:]], axis=0)
        dummy_result = tf.constant([])
        dummy_result = tf.reshape(dummy_result, output_shape)
        dummy_result = tf.cast(dummy_result, dummy_output.dtype)
        return dummy_result
    else:
        result = tf.vectorized_map(function, values)

        # TODO: result has rank 1 if values contains only one row (sometimes wanted sometimes not)
        # if tf.rank(result) < 2:
        #     print()
        return result


# @tf.function
def get_row_indices(columns: tf.Tensor, amt_columns: tf.Tensor, row_number: tf.Tensor):
    rows = tf.cast(tf.repeat(row_number, amt_columns), tf.int32)
    return tf.stack([rows, columns], axis=1)


# @tf.function
def get_col_indices(rows: tf.Tensor, amt_rows: tf.Tensor, col_number: tf.Tensor):
    columns = tf.cast(tf.repeat(col_number, amt_rows), tf.int32)
    return tf.stack([rows, columns], axis=1)


# @tf.function
def get_noisy_value(X: tf.Tensor, amt_rows: tf.Tensor, update_idx: tf.Tensor):
    random_row = tf.random.uniform((), 0, amt_rows, dtype=tf.int32)
    return X[random_row, update_idx[1]]


# @tf.function
def get_mask_indices_old(X: tf.Tensor, V_s: tf.Tensor, F_s: tf.Tensor):
    # TODO: Fix s.t. it works like figure 1 in paper
    # Mark rows and columns given in the explanation as not-changable
    X_shape = tf.shape(X)

    columns = tf.range(X_shape[1])
    v_partial = lambda x: get_row_indices(columns, X_shape[1], x)
    v_indices = tf.reshape(vectorized_map_wrapper(v_partial, V_s), (-1, 2))

    rows = tf.range(X_shape[0])
    f_partial = lambda x: get_col_indices(rows, X_shape[0], x)
    f_indices = tf.reshape(vectorized_map_wrapper(f_partial, F_s), (-1, 2))

    return tf.concat([v_indices, f_indices], axis=0)


# @tf.function
def get_mask_indices(V_s: tf.Tensor, F_s: tf.Tensor):
    indices = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False
    )
    pointer = tf.constant(0)
    for v in V_s:
        for f in F_s:
            indices = indices.write(pointer, [v, f])
            pointer = pointer + tf.constant(1)
    indices = indices.stack()
    return indices


# @tf.function
def paper_fidelity(X: tf.Tensor,
                   A: tf.Tensor,
                   V_s: tf.Tensor,
                   F_s: tf.Tensor,
                   gnn: tf.keras.Model,
                   amt_samples: tf.Tensor = 25):

    if tf.rank(X) > 2:
        X = X[0]
    Ys_probabilities = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False
    )
    pointer = tf.constant(0)

    # Mark rows and columns given in the explanation as not-changable
    fixed_indices = get_mask_indices(V_s, F_s)

    # If there are no selected rows or columns, all entries will be random.
    # Thus, we return zero.
    if tf.math.reduce_all(tf.shape(fixed_indices) == tf.TensorShape([0])):
        return tf.constant(0., dtype=tf.float64)

    X_shape = tf.shape(X)
    for _ in tf.range(amt_samples):
        # Determine indices to update
        update_indices_mask = tf.tensor_scatter_nd_update(
            tf.zeros_like(X), fixed_indices, tf.ones(tf.shape(fixed_indices)[0])
        )
        update_indices_mask = tf.math.logical_not(tf.cast(update_indices_mask, tf.bool))
        update_indices = tf.cast(tf.where(update_indices_mask), tf.int32)

        # Determine new (noise) values for update indices
        get_noisy_value_partial = lambda x: get_noisy_value(X, X_shape[0], x)
        update_values = vectorized_map_wrapper(get_noisy_value_partial, update_indices)

        # Compute noisy X
        Y = tf.tensor_scatter_nd_update(X, update_indices, update_values)
        Y = tf.expand_dims(Y, axis=0)

        # Store prediction of noisy X
        # Predict immediately as single mode is supported by all conv-layers in Spektral
        Y_pred, _ = gnn((Y, A))
        Y_pred = Y_pred[0]
        Ys_probabilities = Ys_probabilities.write(pointer, Y_pred)
        pointer = pointer + tf.constant(1)

    # Compute prediction for original feature matrix
    X = tf.expand_dims(X, axis=0)
    X_probabilities, _ = gnn((X, A))
    X_probabilities = X_probabilities[0]
    X_max_index = tf.argmax(X_probabilities)

    # Compute probability Fidelity- evaluation metric
    Ys_probabilities = Ys_probabilities.stack()
    Ys_max_indices = tf.argmax(Ys_probabilities, axis=-1)

    # Compute fidelity according to the paper
    fidelity_value = tf.math.count_nonzero(tf.cast(Ys_max_indices == X_max_index, tf.float32))
    fidelity_value = tf.cast(fidelity_value, tf.int32)
    fidelity_value = fidelity_value / tf.shape(Ys_max_indices)[0]

    return fidelity_value


# @tf.function
def fidelity(X: tf.Tensor,
             A: tf.Tensor,
             V_s: tf.Tensor,
             F_s: tf.Tensor,
             gnn: tf.keras.Model,
             amt_samples: tf.Tensor = 25):
    """Computes the fidelity of an explanation. [FIDELITY^-]

    The fidelity of an explanation depends on the neural network
    and the noise distribution. It is described as the expected
    value of whether the noisy inputs will get the same
    prediction as the non-noisy inputs.

    An explanation (V_s, F_s) determines what entries in X
    retain their values (V_s -> row, F_s -> cols).

    Compare the predictions of the altered and the original
    feature matrix.

    :param X: The original feature matrix for which an explanation shall be computed.
    :param A: The corresponding adjacency matrix.
    :param V_s: Selected nodes (row in X) which retain (some of) their values.
    :param F_s: Selected features of nodes (col in X) which retain their values.
    :param gnn: The graph neural network to explain.
    :param amt_samples: The amount of noisy feature matrices to produce. Higher number
                        comes with longer computation but might return more credible
                        fidelity values.
    :return: The absolute value of the mean deviation of the noisy predictions from
             the original prediction over all samples.
    """

    if tf.rank(X) > 2:
        X = X[0]
    Ys_probabilities = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False
    )
    pointer = tf.constant(0)

    # Mark rows and columns given in the explanation as not-changable
    fixed_indices = get_mask_indices(V_s, F_s)

    X_shape = tf.shape(X)
    for _ in tf.range(amt_samples):
        # Determine indices to update
        update_indices_mask = tf.tensor_scatter_nd_update(
            tf.zeros_like(X), fixed_indices, tf.ones(tf.shape(fixed_indices)[0])
        )
        update_indices_mask = tf.math.logical_not(tf.cast(update_indices_mask, tf.bool))
        update_indices = tf.cast(tf.where(update_indices_mask), tf.int32)

        # Determine new (noise) values for update indices
        get_noisy_value_partial = lambda x: get_noisy_value(X, X_shape[0], x)
        update_values = vectorized_map_wrapper(get_noisy_value_partial, update_indices)

        # Compute noisy X
        Y = tf.tensor_scatter_nd_update(X, update_indices, update_values)
        Y = tf.expand_dims(Y, axis=0)

        # Store prediction of noisy X
        # Predict immediately as single mode is supported by all conv-layers in Spektral
        Y_pred, _ = gnn((Y, A))
        Y_pred = Y_pred[0]
        Ys_probabilities = Ys_probabilities.write(pointer, Y_pred)
        pointer = pointer + tf.constant(1)

    # Compute prediction for original feature matrix
    X = tf.expand_dims(X, axis=0)
    X_probabilities, _ = gnn((X, A))
    X_probabilities = X_probabilities[0]
    X_max_value = tf.math.reduce_max(X_probabilities)
    X_max_index = tf.argmax(X_probabilities)

    # Compute probability Fidelity- evaluation metric
    Ys_probabilities = Ys_probabilities.stack()
    Ys_values = Ys_probabilities[:, X_max_index]
    Xs_values = tf.fill(tf.shape(Ys_values), X_max_value)
    fidelity_value = tf.math.reduce_mean(Xs_values - Ys_values)

    return tf.math.abs(fidelity_value)


# @tf.function
def is_not_element(query: tf.Tensor, tensor: tf.Tensor):
    """Checks whether a query is contained within the first dimension of a tensor.

    :param query: The value which is searched for within the first dim of 'tensor'
    :param tensor: The tensor in which one searches for 'query'
    :return: Boolean value whether or not 'query' is contained within first dim of
             'tensor'.
    """

    truth_values = tf.math.equal(query, tensor)
    return tf.math.logical_not(tf.math.reduce_any(truth_values))


# @tf.function
def determine_neighborhood(adj_matrix: tf.Tensor, query_node: tf.int32, l_hop: tf.int32):
    """Computes a set of neighbors_ for a query node.

    The neighbors_ of a node within a graph are determined by the
    amount of allowed hops and the topology of the given graph
    (i.e. the allowed paths within the graph).

    :param adj_matrix: The adjacency matrix of the graph in which
                       neighbors_ shall be searched
    :param query_node: The node for which neighbors_ shall be searched
    :param l_hop: The amount of allowed hops for neighbor search
    :return: A tensor of neighbors_ for 'query_node'.
    """

    # neighbors_set contains the neighbors
    neighbors_set = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False
    )

    # neighbor_pointer is an index pointing to the current node to expand
    neighbor_pointer = tf.constant(0, dtype=tf.int32)

    # Initialize neighbors-set with query node
    neighbors_set = neighbors_set.write(0, query_node)
    hop_number = tf.constant(0, dtype=tf.int32)

    # hop_pointer is an index pointing to the last added node of the previous hop
    hop_pointer = tf.constant(0, dtype=tf.int32)

    # Misc initialization
    new_position = tf.constant(1, dtype=tf.int32)
    row_indices = tf.range(tf.shape(adj_matrix)[1], dtype=tf.int32)
    added_neighbors = tf.constant(0, dtype=tf.int32)

    repeat = tf.math.less(hop_number, l_hop)
    while repeat:

        # Extract 1-columns from the adjacency matrix in the row of the current node
        current_node = neighbors_set.read(neighbor_pointer)
        current_node_neighbors = adj_matrix[current_node]
        one_columns = tf.boolean_mask(row_indices, current_node_neighbors)

        # Check and add new 1-columns to neighbors-set
        for col in one_columns:
            if is_not_element(col, neighbors_set.stack()):
                neighbors_set = neighbors_set.write(new_position, col)
                new_position = new_position + tf.constant(1, dtype=tf.int32)
                added_neighbors = added_neighbors + tf.constant(1, dtype=tf.int32)

        # Increment neighbor_pointer
        neighbor_pointer = neighbor_pointer + tf.constant(1, dtype=tf.int32)

        # If neighbor_pointer exceeded hop_pointer, shift hop_pointer by the amount of added
        # neighbors
        if tf.math.greater(neighbor_pointer, hop_pointer):
            hop_number = hop_number + tf.constant(1, dtype=tf.int32)
            hop_pointer = hop_pointer + added_neighbors
            added_neighbors = tf.constant(0, dtype=tf.int32)

        # Repeat while we have not reached 'l_hop' and new neighbors are added
        repeat = tf.math.logical_and(
            tf.math.less(hop_number, l_hop),
            tf.math.less(neighbor_pointer, neighbors_set.size())
        )

    return neighbors_set.stack()


# @tf.function
def find_corresponding_value(ranking: tf.Tensor, original_f: tf.Tensor, fidelity_value: tf.Tensor):

    mask = tf.math.equal(fidelity_value, original_f)
    mask = tf.reshape(mask, (-1,))
    return tf.squeeze(tf.boolean_mask(ranking, mask))


# @tf.function
def sort_ranking(ranking: tf.Tensor):
    """[DEPRECATED] Sorts a tensor of explanations after their fidelities.

    :param ranking: The unsorted fidelity ranking.
                    The last column has to contain the fidelities.
    :return: The sorted ranking.
    """

    # In case there was no nodes/features given to rank
    if tf.math.equal(tf.shape(ranking)[0], tf.constant(0)):
        return tf.constant([], shape=(0, 2))
    else:
        fidelities = ranking[:, -1]
        sorted_fidelities = tf.sort(fidelities)
        fcv_partial = lambda x: find_corresponding_value(ranking, fidelities, x)
        result = vectorized_map_wrapper(fcv_partial, sorted_fidelities)

        # TODO: result sometimes has rank 1
        # if tf.rank(result) < 2:
        #     result = vectorized_map_wrapper(fcv_partial, sorted_fidelities)

        return result


# @tf.function
def compute_feature_fidelities(X: tf.Tensor,
                               A: tf.Tensor,
                               feature: tf.Tensor,
                               selected_features: tf.Tensor,
                               selected_nodes: tf.Tensor,
                               gnn: tf.keras.Model):

    feature = tf.cast(feature, tf.int32)
    feature = tf.expand_dims(feature, axis=0)
    selected_features = tf.concat([selected_features, feature], axis=0)

    fidelity_value = paper_fidelity(X, A, selected_nodes, selected_features, gnn)
    fidelity_value = tf.expand_dims(fidelity_value, axis=0)

    return tf.concat([tf.cast(feature, tf.float32), tf.cast(fidelity_value, tf.float32)], axis=0)


# @tf.function
def compute_node_fidelities(X: tf.Tensor,
                            A: tf.Tensor,
                            node: tf.Tensor,
                            selected_nodes: tf.Tensor,
                            selected_features: tf.Tensor,
                            gnn: tf.keras.Model):

    node = tf.cast(node, tf.int32)
    node = tf.expand_dims(node, axis=0)
    selected_nodes = tf.concat([selected_nodes, node], axis=0)

    fidelity_value = paper_fidelity(X, A, selected_nodes, selected_features, gnn)
    fidelity_value = tf.expand_dims(fidelity_value, axis=0)

    return tf.concat([tf.cast(node, tf.float32), tf.cast(fidelity_value, tf.float32)], axis=0)


# @tf.function
def create_ranking(remaining_nodes: tf.Tensor,
                   remaining_features: tf.Tensor,
                   selected_nodes: tf.Tensor,
                   selected_features: tf.Tensor,
                   X: tf.Tensor,
                   A: tf.Tensor,
                   gnn: tf.keras.Model):
    """Creates a ranking of nodes or features, depending on their computed fidelity.

    The return tensor contains the node or feature, respectively, in the first column
    and their corresponding fidelity value in the second column.

    :param remaining_nodes: Variable nodes. For each, compute the fidelity value and rank them.
    :param remaining_features: Variable features. For each, compute the fidelity value and rank them.
    :param selected_nodes: Nodes (rows in X), which are already selected as being constant.
    :param selected_features: Features (columns in X), which are already selected as being constant.
    :param X: The original feature matrix for which an explanation shall be computed.
    :param A: The corresponding adjacency matrix.
    :param gnn: The graph neural network to explain.
    :return: Sorted tensors of possible amendments to an explanation.
    """

    # Create ranking for fixed feature and multiple nodes.
    # Only makes sense if nodes are already selected.
    if tf.shape(selected_nodes)[0] > tf.constant(0):
        cff_partial = lambda x: compute_feature_fidelities(
            X, A, x, selected_features, selected_nodes, gnn
        )
        features_ranking = tf.map_fn(cff_partial, tf.cast(remaining_features, tf.float32))
    else:
        if tf.shape(remaining_features) == tf.TensorShape([]):
            print()
        nones = tf.fill(tf.shape(remaining_features)[0], tf.constant(-1.))
        nones = tf.reshape(nones, (-1, 1))
        remaining_features = tf.reshape(remaining_features, (-1, 1))
        remaining_features = tf.cast(remaining_features, tf.float32)
        features_ranking = tf.concat([remaining_features, nones], axis=1)

    # Create ranking for fixed node and multiple features.
    # Only makes sense if features are already selected.
    if tf.shape(selected_features)[0] > tf.constant(0):
        cnf_partial = lambda x: compute_node_fidelities(
            X, A, x, selected_nodes, selected_features, gnn
        )
        nodes_ranking = tf.map_fn(cnf_partial, tf.cast(remaining_nodes, tf.float32))
    else:
        nones = tf.fill(tf.shape(remaining_features)[0], tf.constant(-1.))
        nones = tf.reshape(nones, (-1, 1))
        remaining_nodes = tf.reshape(remaining_nodes, (-1, 1))
        remaining_nodes = tf.cast(remaining_nodes, tf.float32)
        nodes_ranking = tf.concat([remaining_nodes, nones], axis=0)

    # Sort rankings
    # features_ranking = sort_ranking(features_ranking)
    # nodes_ranking = sort_ranking(nodes_ranking)

    return features_ranking, nodes_ranking
