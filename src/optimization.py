"""
Functions related to optimization of QAOA angles
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.optimize as optimize
from networkx import Graph
from numpy import ndarray

from src.analytical import calc_expectation_ma_qaoa_analytical_p1
from src.graph_utils import get_index_edge_list
from src.original_qaoa import qaoa_decorator
from src.preprocessing import evaluate_graph_cut, preprocess_subgraphs, evaluate_edge_cut
from src.simulation import calc_expectation_general_qaoa, calc_expectation_ma_qaoa_simulation_subgraphs


@dataclass
class Evaluator:
    """
    Class representing evaluator for target function expectation.
    :var func: Function that takes 1D array of input parameters and evaluates target expectation.
    :var num_angles: Number of elements in the 1D array expected by func.
    """
    func: Callable[[ndarray], float]
    num_angles: int

    @staticmethod
    def get_evaluator_general(target_vals: ndarray, driver_term_vals: ndarray, p: int, use_multi_angle: bool = True) -> Evaluator:
        """
        Returns evaluator of target expectation calculated through simulation.
        :param target_vals: Values of the target function at each computational basis.
        :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
        :param p: Number of QAOA layers.
        :param use_multi_angle: True to use individual angles for each term and qubit. False to use 1 angle for all terms and 1 angle for all qubits.
        :return: Simulation evaluator. The order of input parameters: first, driver term angles for 1st layer in the same order as rows of driver_term_vals,
        then mixer angles for 1st layer in the qubits order, then the same format repeats for other layers.
        """
        num_qubits = len(target_vals).bit_length() - 1
        func = lambda angles: calc_expectation_general_qaoa(angles, target_vals, driver_term_vals, p)

        if use_multi_angle:
            num_angles = (driver_term_vals.shape[0] + num_qubits) * p
        else:
            func = qaoa_decorator(func, driver_term_vals.shape[0], num_qubits)
            num_angles = 2 * p
        return Evaluator(change_sign(func), num_angles)

    @staticmethod
    def get_evaluator_standard_maxcut(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, use_multi_angle: bool = True) -> Evaluator:
        """
        Returns an instance of general evaluator where the target function is the cut function and the driver function includes the existing edge terms only.
        :param graph: Graph for MaxCut problem.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param use_multi_angle: True to use individual angles for each term and qubit. False to use 1 angle for all terms and 1 angle for all qubits.
        :return: Simulation evaluator. The order of input parameters: first, edge angles for 1st layer in the order of graph.edges, then node angles for the 1st layer in the order
        of graph.nodes. Then the format repeats for the remaining p - 1 layers.
        """
        target_vals = evaluate_graph_cut(graph, edge_list)
        driver_term_vals = np.array([evaluate_edge_cut(edge, len(graph)) for edge in get_index_edge_list(graph)])
        return Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle)

    @staticmethod
    def get_evaluator_standard_maxcut_subgraphs(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, use_multi_angle: bool = True) -> Evaluator:
        """
        Returns evaluator of standard target expectation calculated through simulation of individual subgraphs.
        This might be faster than simulation of the whole graph if average vertex degree is much smaller than the total number of nodes.
        :param graph: Target graph for MaxCut.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param use_multi_angle: True to use individual angles for each edge and node. False to use 1 angle for all edges and 1 angle for all nodes.
        :return: Simulation evaluator. The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
        """
        subgraphs = preprocess_subgraphs(graph, p, edge_list)
        func = lambda angles: calc_expectation_ma_qaoa_simulation_subgraphs(angles, subgraphs, p)

        if use_multi_angle:
            num_angles = (len(graph.edges) + len(graph)) * p
        else:
            func = qaoa_decorator(func, len(graph.edges), len(graph))
            num_angles = 2 * p
        return Evaluator(change_sign(func), num_angles)

    @staticmethod
    def get_evaluator_analytical(graph: Graph, edge_list: list[tuple[int, int]] = None, use_multi_angle: bool = True) -> Evaluator:
        """
        Returns analytical evaluator of negative target expectation for p=1.
        :param graph: Target graph for MaxCut.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param use_multi_angle: True to use individual angles for each edge and node. False to use 1 angle for all edges and 1 angle for all nodes.
        :return: Analytical evaluator. The input parameters are specified in the following order: all edge angles in the order of graph.edges, then all node angles
        in the order of graph.nodes.
        """
        func = lambda angles: calc_expectation_ma_qaoa_analytical_p1(angles, graph, edge_list)
        if use_multi_angle:
            num_angles = len(graph.edges) + len(graph)
        else:
            func = qaoa_decorator(func, len(graph.edges), len(graph))
            num_angles = 2
        return Evaluator(change_sign(func), num_angles)


def change_sign(func: callable) -> callable:
    """
    Decorator to change sign of the return value of a given function. Useful to carry out maximization instead of minimization.
    :param func: Function whose sign is to be changed.
    :return: Function with changed sign.
    """
    def func_changed_sign(*args, **kwargs):
        return -func(*args, **kwargs)
    return func_changed_sign


def optimize_qaoa_angles(evaluator: Evaluator) -> tuple[float, ndarray]:
    """
    Wrapper around optimize.minimize that restarts optimization from multiple random starting points to minimize evaluator.
    :param evaluator: Evaluator instance.
    :return: Minimum found value and minimizing array of parameters.
    """
    max_no_improvements = 5
    minimum_improvement = 1e-3
    logger = logging.getLogger('QAOA')

    logger.debug('Optimization...')
    time_start = time.perf_counter()
    angles_best = np.zeros(evaluator.num_angles)
    objective_best = 0
    no_improvement_count = 0
    while no_improvement_count < max_no_improvements:
        next_angles = np.random.uniform(-np.pi, np.pi, len(angles_best))
        result = optimize.minimize(evaluator.func, next_angles)
        if -result.fun > objective_best + minimum_improvement:
            no_improvement_count = 0
            objective_best = -result.fun
            angles_best = result.x
        else:
            no_improvement_count += 1

    time_finish = time.perf_counter()
    logger.debug(f'Optimization done. Time elapsed: {time_finish - time_start}')
    return objective_best, angles_best
