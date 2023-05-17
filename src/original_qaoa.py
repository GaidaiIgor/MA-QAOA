"""
Functions that provide regular QAOA interface to MA-QAOA entry points (2 angles per layer)
"""
import numpy as np
from numpy import ndarray


def convert_angles_qaoa_to_multi_angle(angles: ndarray, num_edges: int, num_nodes: int) -> ndarray:
    """
    Repeats each QAOA angle necessary number of times to convert QAOA angle format to MA-QAOA
    :param angles: angles in QAOA format (2 per layer)
    :param num_edges: Number of edges in the graph
    :param num_nodes: Number of nodes in the graph
    :return: angles in MA-QAOA format (individual angle for each node and edge of the graph in each layer)
    """
    maqaoa_angles = []
    for gamma, beta in zip(angles[::2], angles[1::2]):
        maqaoa_angles += [gamma] * num_edges
        maqaoa_angles += [beta] * num_nodes
    return np.array(maqaoa_angles)


def qaoa_decorator(ma_qaoa_func: callable, num_edges: int, num_nodes: int):
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
