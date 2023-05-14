"""
Functions that calculate auxiliary data structures to speed up quantum simulation.
"""
from itertools import product

import numpy as np
from networkx import Graph
from numpy import ndarray


def get_all_binary_labelings(num_qubits: int) -> ndarray:
    """
    :param num_qubits: Number of qubits (=nodes in the graph)
    :return: 2^n x n array of all possible n-bit strings (written in rows)
    """
    return np.array(list(product([0, 1], repeat=num_qubits)))


def check_edge_cut(labeling: ndarray, u: int, v: int) -> int:
    """
    Checks if a particular edge is cut in a given labeling
    :param labeling: 1D array where elements are 0 or 1 - labels on the nodes in the graph
    :param u: First node of the edge
    :param v: Second node of the edge
    :return: 1 if specified edge is between different labels, 0 otherwise
    """
    return 1 if labeling[-u - 1] != labeling[-v - 1] else 0


def find_max_cut(graph: Graph) -> tuple[int, int]:
    all_labelings = get_all_binary_labelings(len(graph))
    all_cuv_vals = np.array([[check_edge_cut(labeling, u, v) for labeling in all_labelings] for (u, v) in graph.edges])
    obj_vals = np.sum(all_cuv_vals, 0)
    max_ind = int(np.argmax(obj_vals))
    return obj_vals[max_ind], max_ind
