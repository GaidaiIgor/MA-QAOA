"""
Functions that provide regular QAOA interface to MA-QAOA entry points (2 angles per layer)
"""
import numpy as np
from networkx import Graph
from numba import njit
from numpy import ndarray, sin, cos

from src.analytical import run_ma_qaoa_analytical_p1
from src.simulation import run_ma_qaoa_simulation
from src.utils import get_all_combinations


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


def run_qaoa_simulation(angles: ndarray, p: int, all_cuv_vals: ndarray, neighbours: ndarray, basis_bin: ndarray, edge_inds: list[int] = None) -> float:
    """
    Runs regular QAOA by direct simulation of quantum evolution. Dumb and slow, but easy to understand and does not require any additional knowledge.
    :param angles: 1D array of all angles for all layers. Format is the same as in run_ma_qaoa_simulation, except there is only one gamma and beta per layer.
    :param p: Number of QAOA layers
    :param all_cuv_vals: 2D array where each row is a diagonal of Cuv operator for each edge in the graph. Size: num_edges x 2^num_nodes
    :param neighbours: Structure calculated by get_neighbour_labelings
    :param basis_bin: Structure calculated by get_all_binary_labelings
    :param edge_inds: Indices of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>
    """
    angles_maqaoa = convert_angles_qaoa_to_multi_angle(angles, all_cuv_vals.shape[0], neighbours.shape[1])
    return run_ma_qaoa_simulation(angles_maqaoa, p, all_cuv_vals, neighbours, basis_bin, edge_inds)


def run_qaoa_analytical_p1(angles: ndarray, graph: Graph, edge_list: list[tuple[int, int]] = None) -> float:
    """
    Runs regular QAOA. All betas and gammas are forced to be the same.
    :param angles: 1D array of all angles for the first layer. Same format as in run_qaoa_simulation.
    :param graph: Graph for which MaxCut problem is being solved
    :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>
    """
    angles_maqaoa = convert_angles_qaoa_to_multi_angle(angles, len(graph.edges), len(graph))
    return run_ma_qaoa_analytical_p1(angles_maqaoa, graph, edge_list)


@njit
def find_operator_expectation_gamma(operator: ndarray, angles: ndarray, p: int, all_edges: ndarray) -> complex:
    """
    Finds expectation of a given Pauli operator after p layers of QAOA. Starts from the gamma layer.
    :param operator: 1D array of ints. Each integer represents a single-qubit Pauli operators. 0 = I, 1 = -X, 2 = -iY, 3 = Z.
    :param angles: 1D array of all angles. Length: 2*p. Order: beta_p, gamma_p, beta_(p-1), gamma_(p-1), etc.
    :param p: How many layers of QAOA to apply.
    :param all_edges: 2D array of ints. Each row is an edge represented by 2 integers.
    :return: Expectation value after p layers.
    """
    action_z = np.array([3, 2, 1, 0])
    valid_edges = np.empty((0, 2), dtype=np.int64)
    for edge in all_edges:
        if ((operator[edge[0]] == 1) | (operator[edge[0]] == 2)) ^ ((operator[edge[1]] == 1) | (operator[edge[1]] == 2)):
            edge = np.ascontiguousarray(edge).reshape((1, -1))
            valid_edges = np.vstack((valid_edges, edge))

    all_combinations = get_all_combinations(valid_edges)
    expectation = 0
    for combination in all_combinations:
        selected_inds = combination.flatten()
        new_operator = operator.copy()
        for ind in selected_inds:
            new_operator[ind] = action_z[new_operator[ind]]
        num_1 = combination.shape[0]
        num_0 = len(valid_edges) - num_1
        expectation_beta = find_operator_expectation_beta(new_operator, angles[1:], p - 1, all_edges)
        expectation += cos(angles[0]) ** num_0 * (-1j * sin(angles[0])) ** num_1 * expectation_beta
    return expectation


@njit
def find_operator_expectation_beta(operator: ndarray, angles: ndarray, p: int, all_edges: ndarray) -> complex:
    """
    Finds expectation of a given Pauli operator after p layers of QAOA. Starts from the beta layer.
    :param operator: 1D array of ints. Each integer represents a single-qubit Pauli operators. 0 = I, 1 = -X, 2 = -iY, 3 = Z.
    :param angles: 1D array of all angles. Length: 2*p. Order: beta_p, gamma_p, beta_(p-1), gamma_(p-1), etc.
    :param p: How many layers of QAOA to apply.
    :param all_edges: 2D array of ints. Each row is an edge represented by 2 integers.
    :return: Expectation value after p layers.
    """
    action_x = np.array([1, 0, 3, 2])
    valid_vertices = np.where((operator == 2) | (operator == 3))[0]
    if p == 0:
        if len(valid_vertices) > 0:
            return 0
        else:
            num_xs = np.count_nonzero(operator == 1)
            return complex((-1) ** num_xs)

    all_combinations = get_all_combinations(valid_vertices)
    expectation = 0
    for combination in all_combinations:
        new_operator = operator.copy()
        new_operator[combination] = action_x[new_operator[combination]]
        num_1 = len(combination)
        num_0 = len(valid_vertices) - num_1
        expectation_gamma = find_operator_expectation_gamma(new_operator, angles[1:], p, all_edges)
        expectation += cos(2 * angles[0]) ** num_0 * (1j * sin(2 * angles[0])) ** num_1 * expectation_gamma
    return expectation


def run_qaoa_analytical(angles: ndarray, p: int, graph: Graph, selected_edges: list[tuple[int, int]] = None) -> float:
    """
    Runs analytical version of QAOA for arbitrary p by explicit iteration over decision tree for all terms.
    :param angles: 1D array of all angles for all layers. Size: 2*p. Order: gamma1, beta1, gamma2, beta2, etc.
    :param p: Number of QAOA layers.
    :param graph: Graph for which MaxCut problem is being solved. Non-weighted edges.
    :param selected_edges: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>
    """
    if selected_edges is None:
        selected_edges = list(graph.edges)

    all_edges = np.array(graph.edges, dtype=np.int64)
    expectation = len(selected_edges) / 2
    angles = angles[::-1]
    for u, v in selected_edges:
        node_labels = np.array([0] * len(graph))
        node_labels[[u, v]] = 3
        next_expectation = find_operator_expectation_beta(node_labels, angles, p, all_edges)
        assert next_expectation.imag < 1e-10, 'Unexpected imaginary part'
        expectation -= next_expectation.real / 2
    return expectation
