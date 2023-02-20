from itertools import product

import bitarray.util as butil
import numpy as np
from numpy import ndarray


def get_neighbour_labelings(num_qubits: int) -> ndarray:
    """
    Computes neighbours of each state on each qubit
    :param num_qubits: Number of qubits (=nodes in the graph)
    :return: 2^n x n array where value of element [i, j] = k s.t. k=i with j-th bit flipped (i.e. k is a neighbour of i on j-th bit)
    """
    neighbours = np.zeros((2 ** num_qubits, num_qubits), dtype=int)
    for i in range(neighbours.shape[0]):
        bits = butil.int2ba(i, length=num_qubits, endian='little')
        for j in range(neighbours.shape[1]):
            bits[j] = 1 - bits[j]
            neighbours[i, j] = butil.ba2int(bits)
            bits[j] = 1 - bits[j]
    return neighbours


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
