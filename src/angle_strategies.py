"""
Functions that translate given parameters to MA-QAOA angles.
"""
import numpy as np
from numpy import ndarray
import itertools as it

from src.data_processing import DiscreteSineTransform, DiscreteCosineTransform


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


def convert_angles_fourier_to_qaoa(fourier_angles: ndarray) -> ndarray:
    """
    Converts QAOA angles from the fourier to qaoa search space.
    The method is taken from Zhou, Wang, Choi, Pichler, Lukin; Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices.
    https://doi.org/10.1103/PhysRevX.10.021067
    :param fourier_angles: Fourier angles. Size: 2*p. Format: u_1, v_1, ..., u_p, v_p
    :return: QAOA angles.
    """
    us = fourier_angles[::2]
    vs = fourier_angles[1::2]
    gammas = DiscreteSineTransform.transform(us)
    betas = DiscreteCosineTransform.transform(vs)
    qaoa_angles = np.array(list(it.chain(*zip(gammas, betas))))
    return qaoa_angles


def convert_angles_qaoa_to_fourier(qaoa_angles: ndarray) -> ndarray:
    """
    Converts QAOA angles from qaoa to fourier search space.
    :param qaoa_angles: Angles in qaoa search space.
    :return: Angles in fourier search space.
    """
    gammas = qaoa_angles[::2]
    betas = qaoa_angles[1::2]
    us = DiscreteSineTransform.inverse(gammas)
    vs = DiscreteCosineTransform.inverse(betas)
    fourier_angles = np.array(list(it.chain(*zip(us, vs))))
    return fourier_angles


def fourier_decorator(qaoa_func: callable) -> callable:
    """
    Decorates QAOA-style function with Fourier angle interface.
    :param qaoa_func: Function that expects QAOA angles as first parameter.
    :return: Adapted function that accepts angles in Fourier format as first parameter.
    """
    def fourier_wrapped(*args, **kwargs):
        angles_qaoa = convert_angles_fourier_to_qaoa(args[0])
        return qaoa_func(angles_qaoa, *args[1:], **kwargs)
    return fourier_wrapped


def convert_angles_linear_to_qaoa(params: ndarray, p: int) -> ndarray:
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
    Decorates QAOA-style function with linear angle interface.
    :param qaoa_func: Function that expects QAOA angles as first parameter.
    :param p: Number of QAOA layers.
    :return: Adapted function that accepts angles in linear ramp format.
    """
    def linear_wrapped(*args, **kwargs):
        qaoa_angles = convert_angles_linear_to_qaoa(args[0], p)
        return qaoa_func(qaoa_angles, *args[1:], **kwargs)
    return linear_wrapped


def convert_angles_tqa_to_qaoa(params: ndarray, p: int) -> ndarray:
    """
    Returns QAOA angles defined by the TQA strategy. TQA changes angles linearly from 0 to given final gamma over p layers. Beta is changed from final gamma to 0.
    :param params: 1D array with 1 number: final gamma for the angle progression.
    :param p: Number of QAOA layers.
    :return: 1D array of corresponding QAOA angles.
    """
    gammas = np.linspace(params[0] / p, params[0], p) - params[0] / (2 * p)
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
        qaoa_angles = convert_angles_tqa_to_qaoa(args[0], p)
        return qaoa_func(qaoa_angles, *args[1:], **kwargs)
    return tqa_wrapped


def interp_p_series(angles: ndarray) -> ndarray:
    """
    Interpolates p-series of QAOA angles at level p to generate good initial guess for level p + 1.
    :param angles: A series of angles (gamma or beta) as a function of p.
    :return: Initial guess for the series at level p + 1.
    """
    p = len(angles)
    new_angles = np.zeros((p + 1, ))
    new_angles[0] = angles[0]
    for i in range(1, p):
        new_angles[i] = i / p * angles[i - 1] + (1 - i / p) * angles[i]
    new_angles[-1] = angles[-1]
    return new_angles


def interp_qaoa_angles(angles: ndarray, p: int) -> ndarray:
    """
    Interpolates (MA-)QAOA angles at level p to generate a good initial guess for level p + 1.
    The method is taken from Zhou, Wang, Choi, Pichler, Lukin; Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices.
    https://doi.org/10.1103/PhysRevX.10.021067
    :param angles: Level p angles.
    :param p: Current number of QAOA layers.
    :return: Initial guess for level p + 1.
    """
    angle_series = angles.reshape((p, -1))
    new_angle_series = np.zeros((p + 1, angle_series.shape[1]))
    for j in range(new_angle_series.shape[1]):
        new_angle_series[:, j] = interp_p_series(angle_series[:, j])
    return np.squeeze(new_angle_series.reshape((1, -1)))


def fix_angles(eval_func: callable, num_angles: int, inds: list[int], values: list[float]) -> callable:
    """
    Decorator that returns an evaluator function with specified input elements fixed to the specified values.
    :param eval_func: Original evaluator function that accepts ndarray of angles.
    :param num_angles: Size of array expected by the original evaluator function.
    :param inds: Indices of elements that are to be fixed.
    :param values: Values for the fixed elements.
    :return: New evaluator function that expects smaller input array and augments it with the fixed elements and returns the result of the original evaluator.
    """
    def new_func(angles: ndarray):
        mask = np.zeros(num_angles, dtype=bool)
        mask[inds] = True
        full_angles = np.zeros(num_angles)
        full_angles[mask] = values
        full_angles[~mask] = angles
        return eval_func(full_angles)
    return new_func


# def qaoa_scheme_decorator(ma_qaoa_func: callable, duplication_scheme: list[ndarray]) -> callable:
#     """ Test decorator that uses custom duplication schemes. """
#     def qaoa_wrapped(*args, **kwargs):
#         angles_maqaoa = duplicate_angles(args[0], duplication_scheme)
#         return ma_qaoa_func(angles_maqaoa, *args[1:], **kwargs)
#     return qaoa_wrapped
#
#
# def generate_all_duplication_schemes_p1_22(num_edges: int, num_nodes: int) -> list[list[ndarray]]:
#     """ Test """
#     import itertools as it
#     edge_indices = set(range(num_edges))
#     edge_subsets = list(it.chain.from_iterable(it.combinations(edge_indices, combo_len) for combo_len in range(1, num_edges // 2 + 1)))
#     node_indices = set(range(num_nodes))
#     node_subsets = list(it.chain.from_iterable(it.combinations(node_indices, combo_len) for combo_len in range(1, num_nodes // 2 + 1)))
#     subset_combos = it.product(edge_subsets, node_subsets)
#     duplication_schemes = []
#     for edge_combo, node_combo in subset_combos:
#         next_scheme = [np.array(list(edge_combo)), np.array(list(edge_indices - set(edge_combo))), np.array(list(node_combo)) + num_edges,
#                        np.array(list(node_indices - set(node_combo))) + num_edges]
#         duplication_schemes.append(next_scheme)
#     return duplication_schemes
