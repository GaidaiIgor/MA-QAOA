"""
Functions that calculate auxiliary data structures to speed up quantum simulation.
"""
from dataclasses import dataclass

import numpy as np
from networkx import Graph
from numba import njit
from numpy import ndarray
import networkx as nx

from src.graph_utils import get_p_subgraph, find_edge_index, get_index_edge_list


@dataclass
class PSubgraph:
    """
    A class that represents edge-induced p-subgraph for QAOA.
    :var graph: Subgraph itself.
    :var cut_vals: 2D array of size num_edges x 2 ** num_nodes with cut values for each edge in each computational basis.
    :var edge_ind: Index of edge that induced this subgraph in subgraph's edge list.
    :var angle_map: 1D array of indices for overall angles that pertain to this subgraph.
    """
    graph: Graph
    cut_vals: ndarray
    edge_ind: int
    angle_map: ndarray


@njit
def evaluate_z_term(term: ndarray, num_qubits: int) -> ndarray:
    """
    Evaluates a given Z-term of Pauli Z expansion in the computational basis with given number of qubits.
    :param term: Term to evaluate, specified as a 1D array of indices on which Z operators act in big endian format.
    :param num_qubits: Total number of qubits in the system.
    :return: 1D array of size 2 ** num_qubits with the values of the given Z-term in the computational basis.
    """
    term_values = np.zeros(2 ** num_qubits, dtype=np.int8)
    for i in range(len(term_values)):
        bin_vals = [i >> num_qubits - bit_ind - 1 & 1 for bit_ind in term]
        term_values[i] = (-1) ** sum(bin_vals)
    return term_values


@njit
def evaluate_edge_cut(edge: ndarray, num_nodes: int) -> ndarray:
    """
    Evaluates edge cut function for all computational basis states.
    :param edge: edge, specified as a 1D array with 2 indices of the corresponding nodes.
    :param num_nodes: Total number of nodes in the graph.
    :return: 1D array of size 2 ** num_nodes with 1 if the given edge is cut in the corresponding basis or 0 otherwise.
    """
    z_term = evaluate_z_term(edge, num_nodes)
    return (1 - z_term) // 2


@njit
def evaluate_graph_cut_index_edge(index_edge_list: ndarray, num_nodes: int) -> ndarray:
    """
    Evaluates sum of edge cuts for all specified edges.
    :param index_edge_list: 2D array of size num_edges x 2. Each row is an edge specified by indices of nodes in graph.nodes.
    :param num_nodes: Total number of nodes in the graph.
    :return: 1D array of size 2 ** num_nodes with the cut values for each computational basis.
    """
    res = np.zeros(2 ** num_nodes, dtype=np.int32)
    for edge in index_edge_list:
        res += evaluate_edge_cut(edge, num_nodes)
    return res


def evaluate_graph_cut(graph: Graph, edge_list: list[tuple[int, int]] = None) -> ndarray:
    """
    Evaluates sum of edge cuts for all specified edges.
    :param graph: Graph for evaluation.
    :param edge_list: List of edges that should be taken into account. If None, then all edges are taken into account.
    :return: 1D array of size 2 ** num_nodes with the cut values for each computational basis.
    """
    index_edge_list = get_index_edge_list(graph, edge_list)
    return evaluate_graph_cut_index_edge(index_edge_list, len(graph))


def preprocess_subgraphs(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None) -> list[PSubgraph]:
    """
    Calculates p-subgraphs and related structures for each edge in the list.
    :param graph: Considered graph.
    :param p: Number of layers of QAOA.
    :param edge_list: List of considered edges in graph. If None, all edges are considered.
    :return: List of p-subgraphs for all specified edges.
    """
    if edge_list is None:
        edge_list = graph.edges

    nx.set_node_attributes(graph, {node: ind for ind, node in enumerate(graph.nodes)}, name='index')
    nx.set_edge_attributes(graph, {edge: ind for ind, edge in enumerate(graph.edges)}, name='index')
    subgraphs = []
    for edge in edge_list:
        next_subgraph = get_p_subgraph(graph, edge, p)
        cut_vals = np.array([evaluate_edge_cut(edge, len(next_subgraph)) for edge in get_index_edge_list(next_subgraph)])
        edge_ind = find_edge_index(next_subgraph, edge)

        angle_map = []
        for u, v, ind in next_subgraph.edges.data('index'):
            angle_map.append(ind)
        for u, ind in next_subgraph.nodes.data('index'):
            angle_map.append(ind + graph.number_of_edges())
        angle_map = np.array(angle_map)

        subgraphs.append(PSubgraph(next_subgraph, cut_vals, edge_ind, angle_map))
    return subgraphs
