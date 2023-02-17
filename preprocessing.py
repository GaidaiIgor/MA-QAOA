from itertools import product
import bitarray.util as butil
import networkx as nx
import numpy as np
from numpy import ndarray


def get_neighbour_labelings(num_qubits: int) -> ndarray:
    # Returns a 2^n x n array where value of element [i, j] is a number k s.t. j-th bit of i is flipped (i.e. k is a neighbour of i on j-th bit)
    neighbours = np.zeros((2 ** num_qubits, num_qubits), dtype=int)
    for i in range(neighbours.shape[0]):
        bits = butil.int2ba(i, length=num_qubits, endian='little')
        for j in range(neighbours.shape[1]):
            bits[j] = 1 - bits[j]
            neighbours[i, j] = butil.ba2int(bits)
            bits[j] = 1 - bits[j]
    return neighbours


def get_all_binary_labelings(num_nodes: int) -> ndarray:
    # Returns all binary labelings on a given number of graph nodes
    return np.array(list(product([0, 1], repeat=num_nodes)))


def calc_cut_weight_total(labeling: ndarray, graph: nx.Graph) -> float:
    # Calculates total cut weight for a given graph labeling
    cut_weight = 0
    for u, v, w in graph.edges.data('weight'):
        if labeling[-u - 1] != labeling[-v - 1]:
            cut_weight += w
    return cut_weight


def check_edge_cut(labeling: ndarray, u: int, v: int) -> int:
    # Checks if a particular edge is cut in a given labeling
    return 1 if labeling[-u - 1] != labeling[-v - 1] else 0

