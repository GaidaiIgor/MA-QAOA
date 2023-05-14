"""
Functions that provide regular QAOA interface to MA-QAOA entry points (2 angles per layer)
"""
import numpy as np
from networkx import Graph
from numpy import ndarray

from src.analytical import run_ma_qaoa_analytical_p1
from src.simulation import run_ma_qaoa_simulation


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


def run_qaoa_simulation(angles: ndarray, p: int, all_cuv_vals: ndarray, edge_inds: list[int] = None) -> float:
    """
    Runs regular QAOA by direct simulation of quantum evolution.
    :param angles: 1D array of all angles for all layers. Format is the same as in run_ma_qaoa_simulation, except there is only one gamma and beta per layer.
    :param p: Number of QAOA layers
    :param all_cuv_vals: 2D array where each row is a diagonal of Cuv operator for each edge in the graph. Size: num_edges x 2^num_nodes
    :param edge_inds: Indices of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>
    """
    num_qubits = all_cuv_vals.shape[1].bit_length() - 1
    angles_maqaoa = convert_angles_qaoa_to_multi_angle(angles, all_cuv_vals.shape[0], num_qubits)
    return run_ma_qaoa_simulation(angles_maqaoa, p, all_cuv_vals, edge_inds)


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
