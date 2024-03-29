from preprocessing import ADJ_MATRIX_SPARSE
from zorro_algorithm.discrete_mask import DiscreteMask
from zorro_algorithm.zorro_utils import get_best_explanation

import numpy as np
import sys
import tensorflow as tf


def get_explanations(V_p,
                     F_p,
                     threshold_fidelity,
                     X,
                     A,
                     gnn,
                     recursion_depth,
                     max_recursion_depth=np.inf,
                     verbose=True):
    """A possible implementation of the Zorro algorithm from [1].

    Note: In specific, this function represents our implementation of Algorithm (1) in paper [1].

    Parameters
    ----------
    V_p: set
        The set of possible nodes Zorro can select from
    F_p: set
        The set of possible features Zorro can select from
    threshold_fidelity: float
        A threshold fidelity that should be exceeded by the explanations produced by Zorro
    X: np.ndarray
        The feature matrix of the observed graph
    A: np.ndarray
        The adjacency matrix of the observed graph
    gnn: tf.keras.Model
        The graph neural network to analyse
    recursion_depth: int
        The current recursion depth
    max_recursion_depth: int
        The maximal amount of recursions
    verbose: boolean
        Whether to report the algorithm's intermediate results in the console

    Returns
    -------
    list
        A list of disjoint explanations for the graph described by (X, A)

    References
    ----------
    [1]
        Funke, Thorben, Megha Khosla, and Avishek Anand.
        Zorro: Valid, sparse, and stable explanations in graph neural networks.
        arXiv preprint arXiv:2105.08621 (2021).

    """
    S = []
    if recursion_depth < max_recursion_depth:
        mask = DiscreteMask(V_p, F_p)

        # Determine first element in the explanation from the selectable nodes or features
        remaining_elems_left = mask.init_mask(X, A, gnn)

        # Add new elements to the mask as long as the fidelity remains
        # "good enough", which is determined by the target fidelity.
        mask_fidelity = mask.compute_mask_fidelity(X, A, gnn)

        # As long as we produce a fidelity that is smaller than our threshold fidelity,
        # we add more values to the mask such that more original values retain their
        # values.
        loading_signs = ["-", "\\", "|", "/"]
        idx = -1
        while mask_fidelity < threshold_fidelity and remaining_elems_left:  # (IV)
            if verbose:
                sys.stdout.write(
                    f"\rZORRO::: "
                    f"Current mask V_s: {mask.V_s} F_s: {mask.F_s} - "
                    f"Remaining V_r: {mask.V_r} F_r: {mask.F_r} - "
                    f"Mask fidelity: {mask_fidelity:.3f}"
                )
            else:
                loading_sign = loading_signs[idx]
                idx = (idx + 1) % 4
                sys.stdout.write(f"\rZorro algorithm is working. {loading_sign}")
            R_Vr, R_Fr = mask.compute_current_ranking(X, A, gnn)
            remaining_elems_left = mask.add_element_to_mask(R_Vr, R_Fr)
            mask_fidelity = mask.compute_mask_fidelity(X, A, gnn)
        '''
        +-------------------------------------+--------------------------------------------------------+
        |  mask_fidelity < threshold_fidelity | remaining_elems_left                                   |
        | ------------------------------------+--------------------------------------------------------|
        |                 0                   |       0           : Return mask, no recursion     (I)  |
        |                 0                   |       1           : Return mask, recursion        (II) |
        |                 1                   |       0           : Return mask, no recursion     (III)|
        |                 1                   |       1           : While loop                    (IV) |
        +----------------------------------------------------------------------------------------------+
        '''
        if mask_fidelity >= threshold_fidelity and not remaining_elems_left:  # (I)
            return [(mask.V_p, mask.F_p, mask_fidelity)]
        elif mask_fidelity <= threshold_fidelity and not remaining_elems_left:  # (III)
            # Instead of no mask we return the best possible mask if threshold fidelity wasn't
            # exceeded.
            return [(mask.best_V_s, mask.best_F_s, mask.best_fidelity)]
            # return []
        else:  # (II)
            explanation = [(mask.V_s, mask.F_s, mask_fidelity)]
            S.extend(explanation)
            S.extend(get_explanations(
                mask.V_p,
                mask.F_r,
                threshold_fidelity,
                X,
                A,
                gnn,
                recursion_depth+1,
                max_recursion_depth
            ))
            S.extend(get_explanations(
                mask.V_r,
                mask.F_p,
                threshold_fidelity,
                X,
                A,
                gnn,
                recursion_depth+1,
                max_recursion_depth
            ))

    return S


def zorro(gnn, X, A, threshold_fidelity=.7, max_recursion_depth=3):
    """The entry point for the Zorro algorithm

    Parameters
    ----------
    gnn: tf.keras.Model
        The graph neural network to analyse
    X: np.ndarray
        The original feature matrix for which an explanation shall be computed
    A: np.ndarray
        The corresponding adjacency matrix to `X`
    threshold_fidelity: float
        A minimum fidelity to be reached
    max_recursion_depth: int
        Integer that caps depth of recursion tree and therefore amount of recursions.

    Returns
    -------
    list
        A list of disjoint explanations for the graph described by (X, A)
    """

    if len(X.shape) == 2:
        X = tf.expand_dims(X, axis=0)

    shape = X.shape
    V_p = set(np.arange(shape[1]))
    F_p = set(np.arange(shape[2]))

    return get_explanations(V_p, F_p, threshold_fidelity, X, A, gnn, 0, max_recursion_depth)


def zorro_wrapper(gnn, X, original_input, threshold_fidelity=.7):
    """Returns the best explanation from the zorro_algorithm algorithm and serves as an API to our Zorro implementation

    Parameters
    ----------
    gnn: tf.keras.Model
        The graph neural network to explain.
    X: np.ndarray
        The input for which an explanation shall be computed.
    original_input: np.ndarray
        To offer comparability, fidelities are computed always w.r.t. the original input.
    threshold_fidelity: float
        A minimum fidelity to be reached.

    Returns
    -------
    np.ndarray
        The best explanation that Zorro has found for a given observation and target action.
    """

    explanations = zorro(gnn, X, ADJ_MATRIX_SPARSE, threshold_fidelity)
    explanation, action, expl_fidelity = get_best_explanation(explanations, gnn, X, original_input)

    return explanation, action, expl_fidelity
