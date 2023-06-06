"""
Functions that provide analytical formulas for evaluation of the expectation values in QAOA.
"""
import math

import networkx as nx
from networkx import Graph
from numpy import ndarray, sin, cos

from src.graph_utils import get_index_edge_list


def calc_expectation_general_analytical_z1(angles: ndarray, graph: Graph) -> float:
    """
    Calculates target expectation for given angles with Generalized first-order QAOA ansatz via an analytical formula (for p=1).
    :param angles: 1D array of all angles for the first GQAOA layer in the same order as in `get_evaluator_general`.
    :param graph: Graph for which MaxCut problem is being solved.
    :return: Target expectation value for the given angles.
    """
    expectation = len(graph.edges) / 2
    for edge in get_index_edge_list(graph):
        gammas = angles[edge]
        betas = angles[edge + len(graph)]
        expectation -= sin(2 * betas[0]) * sin(2 * betas[1]) * sin(2 * gammas[0]) * sin(2 * gammas[1]) / 2
    return expectation


def calc_expectation_ma_qaoa_analytical_p1(angles: ndarray, graph: Graph, edge_list: list[tuple[int, int]] = None) -> float:
    """
    Calculates target expectation for given angles with MA-QAOA ansatz via an analytical formula for p=1.
    The formula is taken from Vijendran, V., Das, A., Koh, D. E., Assad, S. M. & Lam, P. K. An Expressive Ansatz for Low-Depth Quantum Optimisation. (2023)
    :param angles: 1D array of all angles for the first layer. Same format as in run_ma_qaoa_simulation.
    :param graph: Graph for which MaxCut problem is being solved.
    :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>.
    """
    if edge_list is None:
        edge_list = graph.edges

    gammas = angles[0:len(graph.edges)]
    betas = angles[len(graph.edges):]
    nx.set_edge_attributes(graph, {(u, v): gammas[i] * w for i, (u, v, w) in enumerate(graph.edges.data('weight'))}, name='gamma')
    objective = 0
    for u, v in edge_list:
        w = graph.edges[(u, v)]['weight']
        cuv = w / 2
        d = set(graph[u]) - {v}
        e = set(graph[v]) - {u}
        f = d & e
        cos_prod_d = math.prod([cos(graph.edges[u, m]['gamma']) for m in d - f])
        cos_prod_e = math.prod([cos(graph.edges[v, m]['gamma']) for m in e - f])

        # Triangle terms
        if len(f) != 0:
            cos_prod_f_plus = math.prod([cos(graph.edges[u, m]['gamma'] + graph.edges[v, m]['gamma']) for m in f])
            cos_prod_f_minus = math.prod([cos(graph.edges[u, m]['gamma'] - graph.edges[v, m]['gamma']) for m in f])
            cuv += w / 4 * sin(2 * betas[u]) * sin(2 * betas[v]) * cos_prod_d * cos_prod_e * (cos_prod_f_plus - cos_prod_f_minus)
            cos_prod_d *= math.prod([cos(graph.edges[u, m]['gamma']) for m in f])
            cos_prod_e *= math.prod([cos(graph.edges[v, m]['gamma']) for m in f])

        cuv += w / 2 * sin(graph.edges[u, v]['gamma']) * \
            (sin(2 * betas[u]) * cos(2 * betas[v]) * cos_prod_d + cos(2 * betas[u]) * sin(2 * betas[v]) * cos_prod_e)
        objective += cuv

    return objective
