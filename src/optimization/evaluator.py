""" Module with Evaluator class. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from networkx import Graph
from numpy import ndarray

from src.analytical import calc_expectation_ma_qaoa_analytical_p1
from src.angle_strategies.direct import tqa_decorator, qaoa_decorator, linear_decorator, fourier_decorator, fix_angles
from src.angle_strategies.search_space import SearchSpaceGeneral, SearchSpaceControlled, SearchSpace
from src.graph_utils import get_index_edge_list
from src.preprocessing import evaluate_all_cuts, evaluate_z_term, PSubset
from src.simulation.plain import calc_expectation_general_qaoa, calc_expectation_general_qaoa_subsets


@dataclass
class Evaluator:
    """
    Class representing evaluator for target function expectation.
    :var func: Function that takes 1D array of input parameters and evaluates target expectation.
    :var num_angles: Number of elements in the 1D array expected by func.
    :var search_space: Name of the search space or instance of SearchSpace.
    :var p: Number of QAOA layers.
    :var num_qubits: Number of qubits.
    :var num_phase_terms: Number of terms in the phase operator.
    :var fixed_inds: Fixed indices.
    """
    func: Callable[[ndarray], float]
    num_angles: int
    search_space: str | SearchSpace
    p: int
    num_qubits: int
    num_phase_terms: int
    fixed_inds: Sequence = None

    @staticmethod
    def wrap_parameter_strategy(ma_qaoa_func: callable, num_qubits: int, num_phase_terms: int, p: int, search_space: str | SearchSpace = 'ma') -> Evaluator:
        """
        Wraps MA-QAOA function input according to the specified angle strategy.
        :param ma_qaoa_func: MA-QAOA function of angles only to be maximized.
        :param num_qubits: Number of qubits.
        :param num_phase_terms: Number of terms in the phase function.
        :param p: Number of QAOA layers.
        :param search_space: Name of the search space to use or SearchSpace instance for custom search spaces.
        :return: Simulation evaluator. The order of input parameters is according to the angle strategy.
        """
        if search_space == 'tqa':
            num_angles = 1
            ma_qaoa_func = tqa_decorator(qaoa_decorator(ma_qaoa_func, num_phase_terms, num_qubits), p)
        elif search_space == 'linear':
            num_angles = 4
            ma_qaoa_func = linear_decorator(qaoa_decorator(ma_qaoa_func, num_phase_terms, num_qubits), p)
        elif search_space == 'qaoa' or search_space == 'fourier':
            num_angles = 2 * p
            ma_qaoa_func = qaoa_decorator(ma_qaoa_func, num_phase_terms, num_qubits)
            if search_space == 'fourier':
                ma_qaoa_func = fourier_decorator(ma_qaoa_func)
        elif search_space == 'ma' or search_space == 'general' or search_space == 'xqaoa':
            num_angles = (num_phase_terms + num_qubits) * p
        elif isinstance(search_space, SearchSpace):
            num_angles = search_space.get_num_angles(num_phase_terms, num_qubits, p)
            ma_qaoa_func = search_space.apply_interface(ma_qaoa_func, num_phase_terms, num_qubits, p)
        else:
            raise 'Unknown search space'
        return Evaluator(ma_qaoa_func, num_angles, search_space, p, num_qubits, num_phase_terms)

    @staticmethod
    def get_evaluator_general(target_vals: ndarray, driver_term_vals: ndarray, p: int, search_space: str | SearchSpace = 'general') -> Evaluator:
        """
        Returns evaluator of target expectation calculated through simulation.
        :param target_vals: Values of the target function at each computational basis.
        :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
        :param p: Number of QAOA layers.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters: first, driver term angles for 1st layer in the same order as rows of driver_term_vals,
        then mixer angles for 1st layer in the qubits order, then the same format repeats for other layers.
        """
        if search_space == 'xqaoa':
            mixer_type = 'x+y'
        elif isinstance(search_space, SearchSpaceControlled):
            mixer_type = 'controlled'
        else:
            mixer_type = 'standard'

        func = lambda angles: calc_expectation_general_qaoa(angles, driver_term_vals, p, target_vals, mixer_type)
        num_qubits = len(target_vals).bit_length() - 1
        return Evaluator.wrap_parameter_strategy(func, num_qubits, driver_term_vals.shape[0], p, search_space)

    @staticmethod
    def get_evaluator_standard_maxcut(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None, search_space: str | SearchSpace = 'ma') -> Evaluator:
        """
        Returns an instance of general evaluator where the target function is the cut function and the driver function includes the existing edge terms only.
        :param graph: Graph for MaxCut problem.
        :param p: Number of QAOA layers.
        :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Simulation evaluator. The order of input parameters: first, edge angles for 1st layer in the order of graph.edges, then node angles for the 1st layer in the order
        of graph.nodes. Then the format repeats for the remaining p - 1 layers.
        """
        target_vals = evaluate_all_cuts(graph, edge_list)
        driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
        return Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, search_space)

    @staticmethod
    def get_evaluator_general_subsets(num_qubits: int, target_terms: list[set[int]], target_coeffs: list[float], driver_terms: list[set[int]], p: int,
                                      search_space: str = 'general') -> Evaluator:
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
            search_space = 'ma'
        else:
            func = qaoa_decorator(func, len(graph.edges), len(graph))
            num_angles = 2
            search_space = 'qaoa'
        return Evaluator(func, num_angles, search_space, 1, len(graph), len(edge_list))

    def evaluate(self, angles: ndarray) -> float:
        """
        Evaluates expectation at the given angles.
        :param angles: Array of angles.
        :return: Expectation.
        """
        if len(angles) != self.num_angles:
            raise Exception('Wrong number of angles')
        result = self.func(angles)
        return result

    def fix_params(self, inds: Sequence, values: Sequence):
        """
        Fixes specified parameters to specified values.
        :param inds: Indices to fix.
        :param values: Fixed values.
        :return: None.
        """
        self.fixed_inds = inds
        self.func = fix_angles(self.func, self.num_angles, inds, values)
        self.num_angles -= len(inds)
