"""
Functions that provide regular QAOA interface to MA-QAOA entry points (2 angles per layer)
"""
import numpy as np
from numpy import ndarray


def duplicate_angles(input_angles: ndarray, duplication_scheme: list[ndarray]) -> ndarray:
    """
    Duplicates input angles according to the duplication scheme to convert the angle format to MA-QAOA.
    :param input_angles: 1D array of angles.
    :param duplication_scheme: List of 1D arrays. The number of inner arrays is equal to len(input_angles). Each inner array is a list of indices where the corresponding angle is
    to be duplicated. Must be a permutation of all indices from 0 to max, somehow split over the inner arrays.
    :return: Updated list of angles where the angles are duplicated according to the provided scheme.
    """
    output_len = max([max(inner) for inner in duplication_scheme]) + 1
    output_angles = np.zeros((output_len, ))
    for i, inner in enumerate(duplication_scheme):
        output_angles[inner] = input_angles[i]
    return output_angles


def get_random_duplication_scheme(num_edges: int, num_nodes: int, num_params: list[int]) -> list[ndarray]:
    """
    Returns a duplication scheme where each layer is randomly split between the specified number of parameters.
    :param num_edges: Number of edges in the graph.
    :param num_nodes: Number of nodes in the graph.
    :param num_params: List of length 2*p, where each element is the number of independent parameters for each group of gammas or betas.
    :return: Generated duplication scheme for `duplicate_angles` function.
    """
    duplication_scheme = []
    return duplication_scheme  # not implemented


def convert_angles_qaoa_to_multi_angle(angles: ndarray, num_edges: int, num_nodes: int) -> ndarray:
    """
    Repeats each QAOA angle necessary number of times to convert QAOA angle format to MA-QAOA.
    :param angles: angles in QAOA format (2 per layer).
    :param num_edges: Number of edges in the graph.
    :param num_nodes: Number of nodes in the graph.
    :return: angles in MA-QAOA format (individual angle for each node and edge of the graph in each layer).
    """
    duplication_scheme = []
    for i in range(len(angles)):
        block_len = num_edges if i % 2 == 0 else num_nodes
        shift = 0 if len(duplication_scheme) == 0 else duplication_scheme[-1][-1] + 1
        duplication_scheme.append(np.arange(block_len) + shift)
    return duplicate_angles(angles, duplication_scheme)


def qaoa_decorator(ma_qaoa_func: callable, num_edges: int, num_nodes: int) -> callable:
    """
    Duplicates standard QAOA angles to match MA-QAOA format and executes given function that can evaluate target expectation in MA-QAOA ansatz.
    :param ma_qaoa_func: Function that evaluates target expectation in MA-QAOA ansatz.
    :param num_edges: Number of edges in the graph.
    :param num_nodes: Number of nodes in the graph.
    :return: Result of ma_qaoa_func.
    """
    def qaoa_wrapped(*args, **kwargs):
        angles_maqaoa = convert_angles_qaoa_to_multi_angle(args[0], num_edges, num_nodes)
        return ma_qaoa_func(angles_maqaoa, *args[1:], **kwargs)
    return qaoa_wrapped


def qaoa_scheme_decorator(ma_qaoa_func: callable, duplication_scheme: list[ndarray]) -> callable:
    """ Test decorator that uses custom duplication schemes. """
    def qaoa_wrapped(*args, **kwargs):
        angles_maqaoa = duplicate_angles(args[0], duplication_scheme)
        return ma_qaoa_func(angles_maqaoa, *args[1:], **kwargs)
    return qaoa_wrapped


def generate_all_duplication_schemes_p1_22(num_edges: int, num_nodes: int) -> list[list[ndarray]]:
    import itertools as it
    edge_indices = set(range(num_edges))
    edge_subsets = list(it.chain.from_iterable(it.combinations(edge_indices, combo_len) for combo_len in range(1, num_edges // 2 + 1)))
    node_indices = set(range(num_nodes))
    node_subsets = list(it.chain.from_iterable(it.combinations(node_indices, combo_len) for combo_len in range(1, num_nodes // 2 + 1)))
    subset_combos = it.product(edge_subsets, node_subsets)
    duplication_schemes = []
    for edge_combo, node_combo in subset_combos:
        next_scheme = [np.array(list(edge_combo)), np.array(list(edge_indices - set(edge_combo))), np.array(list(node_combo)) + num_edges,
                       np.array(list(node_indices - set(node_combo))) + num_edges]
        duplication_schemes.append(next_scheme)
    return duplication_schemes
