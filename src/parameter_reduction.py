"""
Functions that translate given parameters to MA-QAOA angles.
"""
import numpy as np
from numpy import ndarray
import itertools as it


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


def convert_angles_qaoa_to_ma(angles: ndarray, num_edges: int, num_nodes: int) -> ndarray:
    """
    Repeats each QAOA angle necessary number of times to convert QAOA angle format to MA-QAOA.
    :param angles: angles in QAOA format (2 per layer).
    :param num_edges: Number of edges in the graph.
    :param num_nodes: Number of nodes in the graph.
    :return: angles in MA-QAOA format (individual angle for each node and edge of the graph in each layer).
    """
    maqaoa_angles = []
    for gamma, beta in zip(angles[::2], angles[1::2]):
        maqaoa_angles += [gamma] * num_edges
        maqaoa_angles += [beta] * num_nodes
    return np.array(maqaoa_angles)


def qaoa_decorator(ma_qaoa_func: callable, num_edges: int, num_nodes: int) -> callable:
    """
    Duplicates standard QAOA angles to match MA-QAOA format and calls the provided MA-QAOA function.
    :param ma_qaoa_func: Function that expects MA-QAOA angles as first parameter.
    :param num_edges: Number of edges in the graph.
    :param num_nodes: Number of nodes in the graph.
    :return: Adapted function that accepts angles in QAOA format.
    """
    def qaoa_wrapped(*args, **kwargs):
        angles_maqaoa = convert_angles_qaoa_to_ma(args[0], num_edges, num_nodes)
        return ma_qaoa_func(angles_maqaoa, *args[1:], **kwargs)
    return qaoa_wrapped


def linear_ramp(params: ndarray, p: int) -> ndarray:
    """
    Returns QAOA angles defined by the linear ramp strategy. Linear ramp changes angles linearly from starting to final over p layers.
    :param params: 1D array of 4 numbers: starting and ending gamma, then starting and ending beta.
    :param p: Number of QAOA layers.
    :return: 1D array of corresponding QAOA angles.
    """
    gammas = np.linspace(params[0], params[1], p)
    betas = np.linspace(params[2], params[3], p)
    qaoa_angles = np.array(list(it.chain(*zip(gammas, betas))))
    return qaoa_angles


def linear_decorator(qaoa_func: callable, p: int) -> callable:
    """
    Translates linear ramp parameters to match QAOA format and calls the provided QAOA function.
    :param qaoa_func: Function that expects QAOA angles as first parameter.
    :param p: Number of QAOA layers.
    :return: Adapted function that accepts angles in linear ramp format.
    """
    def linear_wrapped(*args, **kwargs):
        qaoa_angles = linear_ramp(args[0], p)
        return qaoa_func(qaoa_angles, *args[1:], **kwargs)
    return linear_wrapped


def convert_angles_tqa_qaoa(params: ndarray, p: int) -> ndarray:
    """
    Returns QAOA angles defined by the TQA strategy. TQA changes angles linearly from 0 to given final gamma over p layers. Beta is changed from final gamma to 0.
    :param params: 1D array with 1 number: final gamma for the angle progression.
    :param p: Number of QAOA layers.
    :return: 1D array of corresponding QAOA angles.
    """
    gammas = np.linspace(params[0] / p, params[0], p)
    betas = params[0] - gammas
    qaoa_angles = np.array(list(it.chain(*zip(gammas, betas))))
    return qaoa_angles


def tqa_decorator(qaoa_func: callable, p: int) -> callable:
    """
    Translates TQA parameters to match QAOA format and calls the provided QAOA function.
    :param qaoa_func: Function that expects QAOA angles as first parameter.
    :param p: Number of QAOA layers.
    :return: Adapted function that accepts angles in TQA format.
    """
    def tqa_wrapped(*args, **kwargs):
        qaoa_angles = convert_angles_tqa_qaoa(args[0], p)
        return qaoa_func(qaoa_angles, *args[1:], **kwargs)
    return tqa_wrapped


def qaoa_scheme_decorator(ma_qaoa_func: callable, duplication_scheme: list[ndarray]) -> callable:
    """ Test decorator that uses custom duplication schemes. """
    def qaoa_wrapped(*args, **kwargs):
        angles_maqaoa = duplicate_angles(args[0], duplication_scheme)
        return ma_qaoa_func(angles_maqaoa, *args[1:], **kwargs)
    return qaoa_wrapped


def generate_all_duplication_schemes_p1_22(num_edges: int, num_nodes: int) -> list[list[ndarray]]:
    """ Test """
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
