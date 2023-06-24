"""
Functions that calculate auxiliary data structures to speed up quantum simulation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from networkx import Graph
from numba import njit
from numpy import ndarray

from src.graph_utils import get_index_edge_list


@dataclass
class PSubset:
    """
    P-subset of nodes induced by a given target term.
    :var node_subset: 1D array with subset of nodes induced by a given term and driver function.
    :var target_vals: 1D array of size 2 ** len(node_subset) of target function values for all computational basis states of this subset.
    :var driver_term_vals: 2D array of size #terms x 2 ** len(node_subset) with values of all driver terms that belong to this subset for all computational basis states of this
    subset.
    :var angle_map: 1D array with indices of angles corresponding to terms and nodes of this subset in the overall angle array.
    """
    node_subset: ndarray
    target_vals: ndarray
    driver_term_vals: ndarray
    angle_map: ndarray

    @staticmethod
    def create(total_qubits: int, inducing_term: set[int], driver_terms: list[set[int]], p: int) -> PSubset:
        """
        Creates an instance of PSubset.
        :param total_qubits: Total number of qubits in the problem.
        :param inducing_term: Indices of qubits in the inducing term.
        :param driver_terms: Indices of qubits of each term in Z-expansion of the driver function.
        :param p: Number of QAOA layers.
        :return: Instance of PSubset class.
        """
        current_subset = copy.deepcopy(inducing_term)
        for i in range(p):
            next_subset = copy.deepcopy(current_subset)
            for term_ind, term in enumerate(driver_terms):
                if term & current_subset:
                    next_subset |= term
            current_subset = next_subset
        node_subset = np.array(list(current_subset))

        ind_map = {old_ind: new_ind for new_ind, old_ind in enumerate(node_subset)}
        inducing_term_new = np.array([ind_map[old_ind] for old_ind in inducing_term])
        target_vals = evaluate_z_term(inducing_term_new, len(node_subset))

        subset_term_inds = []
        subset_term_vals = []
        for ind, term in enumerate(driver_terms):
            if term.issubset(current_subset):
                subset_term_inds.append(ind)
                term_new = np.array([ind_map[old_ind] for old_ind in term])
                term_new_vals = evaluate_z_term(term_new, len(node_subset))
                subset_term_vals.append(term_new_vals)
        subset_term_inds = np.array(subset_term_inds)
        subset_term_vals = np.array(subset_term_vals)

        angle_map = []
        angles_per_layer = len(driver_terms) + total_qubits
        for i in range(p):
            angle_map.extend(subset_term_inds + i * angles_per_layer)
            angle_map.extend(node_subset + len(driver_terms) + i * angles_per_layer)
        angle_map = np.array(angle_map)

        return PSubset(current_subset, target_vals, subset_term_vals, angle_map)


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
    :return: 1D array of size 2 ** num_qubits with 1 if the given edge is cut in the corresponding basis or 0 otherwise.
    """
    z_term = evaluate_z_term(edge, num_nodes)
    return (1 - z_term) // 2


@njit
def evaluate_graph_cut_index_edge(index_edge_list: ndarray, num_nodes: int) -> ndarray:
    """
    Evaluates sum of edge cuts for all specified edges.
    :param index_edge_list: 2D array of size num_driver_terms x 2. Each row is an edge specified by indices of nodes in graph.nodes.
    :param num_nodes: Total number of nodes in the graph.
    :return: 1D array of size 2 ** num_qubits with the cut values for each computational basis.
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
    :return: 1D array of size 2 ** num_qubits with the cut values for each computational basis.
    """
    index_edge_list = get_index_edge_list(graph, edge_list)
    return evaluate_graph_cut_index_edge(index_edge_list, len(graph))
