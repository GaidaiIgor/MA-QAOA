"""
Functions that provide regular QAOA interface to MA-QAOA entry points (2 angles per layer)
"""
import numpy as np
from numpy import sin, cos
from networkx import Graph
from numpy import ndarray
from enum import Enum
import itertools as it
import copy

from src.analytical import run_ma_qaoa_analytical_p1
from src.simulation import run_ma_qaoa_simulation


class Pauli(Enum):
    """
    Possible Pauli operators acting on a given qubit. Note that X actually means -X, and Y means -iY.
    """
    I = 0  # I
    X = 1  # -X
    Y = 2  # -iY
    Z = 3  # Z

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


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


def apply_x(instance: Pauli) -> Pauli:
    """
    Applies X on the left to the given instance of Pauli operator.
    :param instance: Single-qubit Pauli operator
    :return: Product of X and instance
    """
    if instance == Pauli.Y:
        return Pauli.Z
    elif instance == Pauli.Z:
        return Pauli.Y
    elif instance == Pauli.I:
        return Pauli.X
    elif instance == Pauli.X:
        return Pauli.I


def apply_x_all(operator: ndarray, inds: list[int]) -> ndarray:
    """
    Applies X to a set of qubits of a given Pauli operator defined by inds.
    :param operator: Operator is a product of single-qubit Pauli operators, given in the elements of this array
    :param inds: Indexes where X is applied
    :return: New operator
    """
    new_operator = copy.deepcopy(operator)
    for ind in inds:
        new_operator[ind] = apply_x(new_operator[ind])
    return new_operator


def apply_z(instance: Pauli) -> Pauli:
    """
    Applies Z on the left to the given instance of Pauli operator.
    :param instance: Single-qubit Pauli operator
    :return: Product of Z and instance
    """
    if instance == Pauli.Y:
        return Pauli.X
    elif instance == Pauli.X:
        return Pauli.Y
    elif instance == Pauli.Z:
        return Pauli.I
    elif instance == Pauli.I:
        return Pauli.Z


def apply_z_all(operator: ndarray, inds: list[int]) -> ndarray:
    """
    Applies Z to a set of qubits of operator defined by inds.
    :param operator: Operator is a product of single-qubit Pauli operators, given in the elements of this array
    :param inds: Indexes where Z is applied
    :return: New operator
    """
    new_operator = copy.deepcopy(operator)
    for ind in inds:
        new_operator[ind] = apply_z(new_operator[ind])
    return new_operator


def find_operator_expectation_gamma(operator: ndarray, angles: ndarray, p: int, graph: Graph) -> complex:
    """
    Finds expectation of a given Pauli operator after p layers of QAOA. Starts from the gamma layer.
    :param operator: Operator is a product of single-qubit Pauli operators, given in the elements of this array
    :param angles: 1D array of all angles. Length: 2*p. Order: beta_p, gamma_p, beta_(p-1), gamma_(p-1), etc.
    :param p: How many layers of QAOA to apply
    :param graph: Graph for which MaxCut is being solved
    :return: Expectation value after p layers
    """
    valid_edges = [edge for edge in graph.edges if
                   ((operator[edge[0]] == Pauli.X) | (operator[edge[0]] == Pauli.Y)) ^ ((operator[edge[1]] == Pauli.X) | (operator[edge[1]] == Pauli.Y))]
    all_combinations = it.chain.from_iterable(it.combinations(valid_edges, k) for k in range(len(valid_edges) + 1))
    expectation = 0
    for combination in all_combinations:
        selected_inds = list(it.chain.from_iterable(combination))
        new_operator = apply_z_all(operator, selected_inds)
        num_1 = len(selected_inds) / 2
        num_0 = len(valid_edges) - num_1
        expectation += cos(angles[0]) ** num_0 * (-1j * sin(angles[0])) ** num_1 * find_operator_expectation_beta(new_operator, angles[1:], p - 1, graph)
    return expectation


def find_operator_expectation_beta(operator: ndarray, angles: ndarray, p: int, graph: Graph) -> complex:
    """
    Finds expectation of a given Pauli operator after p layers of QAOA. Starts from the beta layer.
    :param operator: Operator is a product of single-qubit Pauli operators, given in the elements of this array
    :param angles: 1D array of all angles. Length: 2*p. Order: beta_p, gamma_p, beta_(p-1), gamma_(p-1), etc.
    :param p: How many layers of QAOA to apply
    :param graph: Graph for which MaxCut is being solved
    :return: Expectation value after p layers
    """
    valid_vertices = np.where((operator == Pauli.Y) | (operator == Pauli.Z))[0]
    if p == 0:
        if len(valid_vertices) > 0:
            return 0
        else:
            num_xs = np.count_nonzero(operator == Pauli.X)
            return (-1) ** num_xs

    all_combinations = it.chain.from_iterable(it.combinations(valid_vertices, k) for k in range(len(valid_vertices) + 1))
    expectation = 0
    for combination in all_combinations:
        selected_inds = list(combination)
        new_operator = apply_x_all(operator, selected_inds)
        num_1 = len(selected_inds)
        num_0 = len(valid_vertices) - num_1
        expectation += cos(2 * angles[0]) ** num_0 * (1j * sin(2 * angles[0])) ** num_1 * find_operator_expectation_gamma(new_operator, angles[1:], p, graph)
    return expectation


def run_qaoa_analytical(angles: ndarray, p: int, graph: Graph, edge_list: list[tuple[int, int]] = None) -> float:
    """
    Runs analytical version of QAOA for arbitrary p by explicit iteration over decision tree for all terms.
    :param angles: 1D array of all angles for all layers. Size: 2*p. Order: gamma1, beta1, gamma2, beta2, etc.
    :param p: Number of QAOA layers.
    :param graph: Graph for which MaxCut problem is being solved. Non-weighted edges.
    :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>
    """
    if edge_list is None:
        edge_list = graph.edges

    expectation = len(edge_list) / 2
    angles = angles[::-1]
    for u, v in edge_list:
        node_labels = np.array([Pauli.I] * len(graph))
        node_labels[[u, v]] = Pauli.Z
        expectation -= find_operator_expectation_beta(node_labels, angles, p, graph).real / 2
    return expectation
