"""
Functions related to optimization of QAOA angles.
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
from scipy.optimize import OptimizeResult

# from qiskit_aer.primitives import Estimator as AerEstimator

# from qiskit.primitives import Estimator

from src.analytical import calc_expectation_ma_qaoa_analytical_p1
from src.angle_strategies import qaoa_decorator, linear_decorator, tqa_decorator, fix_angles, fourier_decorator
from src.graph_utils import get_index_edge_list
from src.preprocessing import PSubset, evaluate_graph_cut, evaluate_z_term
from src.simulation.plain import calc_expectation_general_qaoa, calc_expectation_general_qaoa_subsets
# from src.simulation.qiskit_backend import get_observable_maxcut, get_ma_ansatz, evaluate_angles_ma_qiskit_fast, get_ma_ansatz_alt

# from src.simulation.qiskit_backend import evaluate_angles_ma_qiskit, get_observable_maxcut, get_ma_ansatz, evaluate_angles_ma_qiskit_fast


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
    def wrap_parameter_strategy(ma_qaoa_func: callable, num_qubits: int, num_driver_terms: int, p: int, search_space: str = 'ma') -> Evaluator:
        """
        Wraps MA-QAOA function input according to the specified angle strategy.
        :param ma_qaoa_func: MA-QAOA function of angles only to be maximized.
        :param num_qubits: Number of qubits.
        :param num_driver_terms: Number of terms in the driver function.
        :param p: Number of QAOA layers.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters is according to the angle strategy.
        """
        if search_space == 'ma' or search_space == 'xqaoa':
            num_angles = (num_driver_terms + num_qubits) * p
        elif search_space == 'qaoa' or search_space == 'fourier':
            num_angles = 2 * p
            ma_qaoa_func = qaoa_decorator(ma_qaoa_func, num_driver_terms, num_qubits)
            if search_space == 'fourier':
                ma_qaoa_func = fourier_decorator(ma_qaoa_func)
        elif search_space == 'linear':
            num_angles = 4
            ma_qaoa_func = linear_decorator(qaoa_decorator(ma_qaoa_func, num_driver_terms, num_qubits), p)
        elif search_space == 'tqa':
            num_angles = 1
            ma_qaoa_func = tqa_decorator(qaoa_decorator(ma_qaoa_func, num_driver_terms, num_qubits), p)
        else:
            raise 'Unknown search space'
        return Evaluator(change_sign(ma_qaoa_func), num_angles)

    @staticmethod
    def get_evaluator_general(target_vals: ndarray, driver_term_vals: ndarray, p: int, search_space: str = 'ma') -> Evaluator:
        """
        Returns evaluator of target expectation calculated through simulation.
        :param target_vals: Values of the target function at each computational basis.
        :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
        :param p: Number of QAOA layers.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters: first, driver term angles for 1st layer in the same order as rows of driver_term_vals,
        then mixer angles for 1st layer in the qubits order, then the same format repeats for other layers.
        """
        apply_y = search_space == 'xqaoa'
        func = lambda angles: calc_expectation_general_qaoa(angles, driver_term_vals, p, target_vals, apply_y)
        num_qubits = len(target_vals).bit_length() - 1
        return Evaluator.wrap_parameter_strategy(func, num_qubits, driver_term_vals.shape[0], p, search_space)

    @staticmethod
    def get_evaluator_standard_maxcut(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, search_space: str = 'ma') -> Evaluator:
        """
        Returns an instance of general evaluator where the target function is the cut function and the driver function includes the existing edge terms only.
        :param graph: Graph for MaxCut problem.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters: first, edge angles for 1st layer in the order of graph.edges, then node angles for the 1st layer in the order
        of graph.nodes. Then the format repeats for the remaining p - 1 layers.
        """
        target_vals = evaluate_graph_cut(graph, edge_list)
        driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
        return Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, search_space)

    @staticmethod
    def get_evaluator_general_subsets(num_qubits: int, target_terms: list[set[int]], target_coeffs: list[float], driver_terms: list[set[int]], p: int,
                                      search_space: str = 'ma') -> Evaluator:
        """
        Returns general evaluator that evaluates by separation into subsets corresponding to each target term.
        :param num_qubits: Total number of qubits in the problem.
        :param target_terms: Indices of qubits of each term in Z-expansion of the target function.
        :param target_coeffs: List of size len(subsets) + 1 with multiplier coefficients for each subset given in the same order. The last extra coefficient is a shift.
        :param driver_terms: Indices of qubits of each term in Z-expansion of the driver function.
        :param p: Number of QAOA layers.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Evaluator that accepts 1D array of angles and returns the corresponding target expectation value. The order of angles is the same as in `get_evaluator_general`.
        """
        subsets = [PSubset.create(num_qubits, inducing_term, driver_terms, p) for inducing_term in target_terms]
        func = lambda angles: calc_expectation_general_qaoa_subsets(angles, subsets, target_coeffs, p)
        return Evaluator.wrap_parameter_strategy(func, num_qubits, len(driver_terms), p, search_space)

    @staticmethod
    def get_evaluator_standard_maxcut_subgraphs(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, search_space: str = 'ma') -> Evaluator:
        """
        Returns an instance of general subset evaluator where the target function is the cut function and the driver function includes the existing edge terms only.
        :param graph: Target graph for MaxCut.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Simulation subgraph evaluator. The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
        """
        target_terms = [set(edge) for edge in get_index_edge_list(graph, edge_list)]
        target_term_coeffs = [-1 / 2] * len(target_terms) + [len(target_terms) / 2]
        driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
        return Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p, search_space)

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
        func = lambda angles: calc_expectation_ma_qaoa_analytical_p1(angles, graph, edge_list)
        if use_multi_angle:
            num_angles = len(graph.edges) + len(graph)
        else:
            func = qaoa_decorator(func, len(graph.edges), len(graph))
            num_angles = 2
        return Evaluator(change_sign(func), num_angles)

    # @staticmethod
    # def get_evaluator_qiskit(graph: Graph, p: int, search_space: str = 'ma') -> Evaluator:
    #     """
    #     Returns qiskit evaluator of maxcut expectation.
    #     :param graph: Graph for maxcut.
    #     :param p: Number of QAOA layers.
    #     :param search_space: Name of the strategy to choose the number of variable parameters.
    #     :return: Evaluator that computes maxcut expectation achieved by MA-QAOA with given angles (implemented with qiskit).
    #     The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
    #     """
    #     func = lambda angles: evaluate_angles_ma_qiskit(angles, graph, p)
    #     return Evaluator.wrap_parameter_strategy(func, len(graph), len(graph.edges), p, search_space)

    # @staticmethod
    # def get_evaluator_qiskit_fast(graph: Graph, p: int, search_space: str = 'ma') -> Evaluator:
    #     """
    #     Returns qiskit evaluator of maxcut expectation.
    #     :param graph: Graph for maxcut.
    #     :param p: Number of QAOA layers.
    #     :param search_space: Name of the strategy to choose the number of variable parameters.
    #     :return: Evaluator that computes maxcut expectation achieved by MA-QAOA with given angles (implemented with qiskit).
    #     The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
    #     """
    #     maxcut_hamiltonian = get_observable_maxcut(graph)
    #     estimator = AerEstimator(approximation=True, run_options={'shots': None})
    #     # estimator = Estimator()
    #     # ansatz = get_ma_ansatz(graph, p)
    #     ansatz = get_ma_ansatz_alt(graph, p)
    #     func = lambda angles: evaluate_angles_ma_qiskit_fast(angles, ansatz, estimator, maxcut_hamiltonian)
    #     return Evaluator.wrap_parameter_strategy(func, len(graph), len(graph.edges), p, search_space)

    def fix_params(self, inds, values):
        """
        Fixes specified parameters to specified values.
        :param inds: Indices to fix.
        :param values: Fixed values.
        :return: None.
        """
        self.func = fix_angles(self.func, self.num_angles, inds, values)
        self.num_angles -= len(inds)

    # @staticmethod
    # def get_evaluator_general_scheme(target_vals: ndarray, driver_term_vals: ndarray, p: int, duplication_scheme: list[ndarray]) -> Evaluator:
    #     """ Test method """
    #     func = lambda angles: calc_expectation_general_qaoa(angles, driver_term_vals, p, target_vals)
    #     func = qaoa_scheme_decorator(func, duplication_scheme)
    #     num_angles = len(duplication_scheme)
    #     return Evaluator(change_sign(func), num_angles)


def change_sign(func: callable) -> callable:
    """
    Decorator to change sign of the return value of a given function. Useful to carry out maximization instead of minimization.
    :param func: Function whose sign is to be changed.
    :return: Function with changed sign.
    """
    def func_changed_sign(*args, **kwargs):
        return -func(*args, **kwargs)
    return func_changed_sign


def optimize_qaoa_angles(evaluator: Evaluator, starting_point: ndarray = None, num_restarts: int = 1, objective_max: float = None, objective_tolerance: float = 0.9995, **kwargs) \
        -> OptimizeResult:
    """
    Wrapper around minimizer function that restarts optimization from multiple random starting points to minimize evaluator.
    :param evaluator: Evaluator instance.
    :param starting_point: Starting point for optimization. Chosen randomly if None.
    :param num_restarts: Number of random starting points to try. Has no effect if specific starting point is provided.
    :param objective_max: Maximum achievable objective. Optimization stops if answer sufficiently close to max_objective is achieved.
    :param objective_tolerance: Fraction of 1 that controls how close the result need to be to objective_max before optimization can be stopped.
    :param kwargs: Keyword arguments for optimizer.
    :return: Minimization result.
    """
    if starting_point is not None:
        num_restarts = 1

    logger = logging.getLogger('QAOA')
    logger.debug('Optimization...')
    time_start = time.perf_counter()

    result_best = None
    for i in range(num_restarts):
        if starting_point is not None:
            next_angles = starting_point
        else:
            next_angles = np.random.uniform(-np.pi, np.pi, evaluator.num_angles)

        result = optimize.minimize(evaluator.func, next_angles, **kwargs)
        if not result.success:
            print(result)
            raise Exception('Optimization failed')
        result.x = np.arctan2(np.sin(result.x), np.cos(result.x))  # Normalize angle range

        if result_best is None or result.fun < result_best.fun:
            result_best = result

        if objective_max is not None and -result_best.fun / objective_max > objective_tolerance:
            break

    time_finish = time.perf_counter()
    logger.debug(f'Optimization done. Time elapsed: {time_finish - time_start}')
    return result_best
