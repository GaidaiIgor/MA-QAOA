"""
Functions related to optimization of QAOA angles.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import scipy.optimize as optimize
from networkx import Graph
from numpy import ndarray

import src.analytical
from src.graph_utils import get_index_edge_list
from src.parameter_reduction import qaoa_decorator, qaoa_scheme_decorator, linear_decorator
from src.preprocessing import PSubset, evaluate_graph_cut, evaluate_z_term
from src.simulation import calc_expectation_general_qaoa, calc_expectation_general_qaoa_subsets


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
    def wrap_parameter_strategy(ma_qaoa_func: callable, num_qubits: int, num_driver_terms: int, p: int, angle_strategy: str = 'ma') -> Evaluator:
        """
        Wraps MA-QAOA function input according to the specified angle strategy.
        :param ma_qaoa_func: MA-QAOA function of angles only to be maximized.
        :param num_qubits: Number of qubits.
        :param num_driver_terms: Number of terms in the driver function.
        :param p: Number of QAOA layers.
        :param angle_strategy: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters is according to the angle strategy.
        """
        if angle_strategy == 'ma':
            num_angles = (num_driver_terms + num_qubits) * p
        elif angle_strategy == 'regular':
            num_angles = 2 * p
            ma_qaoa_func = qaoa_decorator(ma_qaoa_func, num_driver_terms, num_qubits)
        elif angle_strategy == 'linear':
            num_angles = 4
            ma_qaoa_func = linear_decorator(qaoa_decorator(ma_qaoa_func, num_driver_terms, num_qubits), p)
        return Evaluator(change_sign(ma_qaoa_func), num_angles)

    @staticmethod
    def get_evaluator_general(target_vals: ndarray, driver_term_vals: ndarray, p: int, angle_strategy: str = 'ma') -> Evaluator:
        """
        Returns evaluator of target expectation calculated through simulation.
        :param target_vals: Values of the target function at each computational basis.
        :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
        :param p: Number of QAOA layers.
        :param angle_strategy: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters: first, driver term angles for 1st layer in the same order as rows of driver_term_vals,
        then mixer angles for 1st layer in the qubits order, then the same format repeats for other layers.
        """
        func = lambda angles: calc_expectation_general_qaoa(angles, driver_term_vals, p, target_vals)
        num_qubits = len(target_vals).bit_length() - 1
        return Evaluator.wrap_parameter_strategy(func, num_qubits, driver_term_vals.shape[0], p, angle_strategy)

    @staticmethod
    def get_evaluator_standard_maxcut(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, angle_strategy: str = 'ma') -> Evaluator:
        """
        Returns an instance of general evaluator where the target function is the cut function and the driver function includes the existing edge terms only.
        :param graph: Graph for MaxCut problem.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param angle_strategy: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters: first, edge angles for 1st layer in the order of graph.edges, then node angles for the 1st layer in the order
        of graph.nodes. Then the format repeats for the remaining p - 1 layers.
        """
        target_vals = evaluate_graph_cut(graph, edge_list)
        driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
        return Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, angle_strategy)

    @staticmethod
    def get_evaluator_general_subsets(num_qubits: int, target_terms: list[set[int]], target_coeffs: list[float], driver_terms: list[set[int]], p: int,
                                      angle_strategy: str = 'ma') -> Evaluator:
        """
        Returns general evaluator that evaluates by separation into subsets corresponding to each target term.
        :param num_qubits: Total number of qubits in the problem.
        :param target_terms: Indices of qubits of each term in Z-expansion of the target function.
        :param target_coeffs: List of size len(subsets) + 1 with multiplier coefficients for each subset given in the same order. The last extra coefficient is a shift.
        :param driver_terms: Indices of qubits of each term in Z-expansion of the driver function.
        :param p: Number of QAOA layers.
        :param angle_strategy: Name of the strategy to choose the number of variable parameters.
        :return: Evaluator that accepts 1D array of angles and returns the corresponding target expectation value. The order of angles is the same as in `get_evaluator_general`.
        """
        subsets = [PSubset.create(num_qubits, inducing_term, driver_terms, p) for inducing_term in target_terms]
        func = lambda angles: calc_expectation_general_qaoa_subsets(angles, subsets, target_coeffs, p)
        return Evaluator.wrap_parameter_strategy(func, num_qubits, len(driver_terms), p, angle_strategy)

    @staticmethod
    def get_evaluator_standard_maxcut_subgraphs(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, angle_strategy: str = 'ma') -> Evaluator:
        """
        Returns an instance of general subset evaluator where the target function is the cut function and the driver function includes the existing edge terms only.
        :param graph: Target graph for MaxCut.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param angle_strategy: Name of the strategy to choose the number of variable parameters.
        :return: Simulation subgraph evaluator. The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
        """
        target_terms = [set(edge) for edge in get_index_edge_list(graph, edge_list)]
        target_term_coeffs = [-1 / 2] * len(target_terms) + [len(target_terms) / 2]
        driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
        return Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p, angle_strategy)

    @staticmethod
    def get_evaluator_standard_maxcut_analytical(graph: Graph, edge_list: list[tuple[int, int]] = None, use_multi_angle: bool = True) -> Evaluator:
        """
        Returns analytical evaluator of negative target expectation for p=1 with MA-QAOA ansatz.
        :param graph: Target graph for MaxCut.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param use_multi_angle: True to use individual angles for each edge and node. False to use 1 angle for all edges and 1 angle for all nodes.
        :return: Analytical evaluator. The input parameters are specified in the following order: all edge angles in the order of graph.edges, then all node angles
        in the order of graph.nodes.
        """
        func = lambda angles: src.analytical.calc_expectation_ma_qaoa_analytical_p1(angles, graph, edge_list)
        if use_multi_angle:
            num_angles = len(graph.edges) + len(graph)
        else:
            func = qaoa_decorator(func, len(graph.edges), len(graph))
            num_angles = 2
        return Evaluator(change_sign(func), num_angles)

    @staticmethod
    def get_evaluator_general_scheme(target_vals: ndarray, driver_term_vals: ndarray, p: int, duplication_scheme: list[ndarray]) -> Evaluator:
        """ Test method """
        func = lambda angles: calc_expectation_general_qaoa(angles, driver_term_vals, p, target_vals)
        func = qaoa_scheme_decorator(func, duplication_scheme)
        num_angles = len(duplication_scheme)
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


def optimize_qaoa_angles(evaluator: Evaluator, starting_point: ndarray = None, num_restarts: int = 1) -> tuple[float, ndarray]:
    """
    Wrapper around minimizer function that restarts optimization from multiple random starting points to minimize evaluator.
    :param evaluator: Evaluator instance.
    :param starting_point: Starting point for optimization. Chosen randomly if None.
    :param num_restarts: Number of random starting points to try. Has no effect if specific starting point is provided.
    :return: Minimum found value and minimizing array of parameters.
    """
    if starting_point is not None:
        num_restarts = 1

    logger = logging.getLogger('QAOA')
    logger.debug('Optimization...')
    time_start = time.perf_counter()

    angles_best = np.zeros(evaluator.num_angles)
    objective_best = 0
    for i in range(num_restarts):
        if starting_point is not None:
            next_angles = starting_point
        else:
            next_angles = np.random.uniform(-np.pi, np.pi, len(angles_best))

        result = optimize.minimize(evaluator.func, next_angles)
        if -result.fun > objective_best:
            objective_best = -result.fun
            angles_best = result.x

    time_finish = time.perf_counter()
    logger.debug(f'Optimization done. Time elapsed: {time_finish - time_start}')
    return objective_best, angles_best
