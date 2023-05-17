"""
Functions related to optimization of QAOA angles
"""
import logging
import time

import numpy as np
import scipy.optimize as optimize
from networkx import Graph
from numpy import ndarray

from src.analytical import calc_expectation_ma_qaoa_analytical_p1
from src.original_qaoa import qaoa_decorator
from src.preprocessing import get_edge_cut, preprocess_subgraphs
from src.simulation import calc_expectation_ma_qaoa_simulation, calc_expectation_ma_qaoa_simulation_subgraphs


def change_sign(func: callable) -> callable:
    """
    Decorator to change sign of the return value of a given function. Useful to carry out maximization instead of minimization.
    :param func: Function whose sign is to be changed.
    :return: Function with changed sign.
    """
    def func_changed_sign(*args, **kwargs):
        return -func(*args, **kwargs)
    return func_changed_sign


def create_evaluator(use_multi_angle: bool, use_analytical: bool, use_subgraphs: bool, p: int, graph: Graph, edge_list: list[tuple[int, int]]) -> callable:
    """
    Creates expectation evaluator function for maximization.
    :param use_multi_angle: True to use individual angles for each node and edge of the graph (MA-QAOA) or false for regular QAOA ansatz.
    :param use_analytical: True to use analytical expression to evaluate expectation value (available for p=1 only), false for simulation.
    :param use_subgraphs: True to simulate p-subgraphs induced by each edge separately. False to simulate the entire graph once.
    :param p: Number of QAOA layers.
    :param graph: Graph for which MaxCut problem is being solved.
    :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Returns a function that takes QAOA angles as input and evaluates negative target expectation.
    """
    if edge_list is None:
        edge_list = list(graph.edges)
    assert not use_analytical or p == 1, "Analytical mode is only available for p = 1"
    logger = logging.getLogger('QAOA')

    if use_analytical:
        evaluator = lambda angles: calc_expectation_ma_qaoa_analytical_p1(angles, graph, edge_list)
    else:
        logger.debug('Preprocessing...')
        time_start = time.perf_counter()

        if use_subgraphs:
            subgraphs = preprocess_subgraphs(graph, p, edge_list)
            evaluator = lambda angles: calc_expectation_ma_qaoa_simulation_subgraphs(angles, p, subgraphs)
        else:
            cut_vals = np.array([get_edge_cut(edge, len(graph)) for edge in graph.edges])
            all_edges = list(graph.edges)
            edge_inds = np.array([all_edges.index(edge) for edge in edge_list])
            evaluator = lambda angles: calc_expectation_ma_qaoa_simulation(angles, p, cut_vals, edge_inds)

        time_finish = time.perf_counter()
        logger.debug(f'Preprocessing done. Time elapsed: {time_finish - time_start}')

    if not use_multi_angle:
        evaluator = qaoa_decorator(evaluator, len(graph.edges), len(graph))
    return change_sign(evaluator)


def optimize_qaoa_angles(use_multi_angle: bool, use_analytical: bool, use_subgraphs: bool, p: int, graph: Graph, edge_list: list[tuple[int, int]] = None) -> \
        tuple[float, ndarray]:
    """
    Optimizes QAOA angles. Starts from random angles multiple times until better solution can no longer be found.
    :param use_multi_angle: True to use individual angles for each node and edge of the graph (MA-QAOA). False to use regular QAOA ansatz.
    :param use_analytical: True to use analytical expression to evaluate expectation value (available for p=1 only). False to use simulation.
    :param use_subgraphs: True to simulate p-subgraphs induced by each edge separately. False to simulate the entire graph once.
    Has no effect in analytical mode. If p-subgraphs are small, setting this to True could exponentially speed up simulation.
    However, if sizes of p-subgraphs are equal or close to the original graph, then it might be faster to set this to False.
    :param p: Number of QAOA layers.
    :param graph: Graph for which MaxCut problem is being solved.
    :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: 1) Maximum expectation value achieved during optimization.
    2) Set of angles that result in the returned expectation value. Format: angles are specified in the order of application for simulation, i.e.
    all gammas for 1st layer (in the edge order), then all betas for 1st layer (in the nodes order), then the same format repeats for all other layers.
    """
    max_no_improvements = 5
    minimum_improvement = 1e-3
    logger = logging.getLogger('QAOA')

    evaluator = create_evaluator(use_multi_angle, use_analytical, use_subgraphs, p, graph, edge_list)
    logger.debug('Optimization...')
    time_start = time.perf_counter()
    num_angles_per_layer = len(graph.edges) + len(graph) if use_multi_angle else 2
    angles_best = np.zeros(num_angles_per_layer * p)
    objective_best = 0
    no_improvement_count = 0
    while no_improvement_count < max_no_improvements:
        next_angles = np.random.uniform(-np.pi, np.pi, len(angles_best))
        result = optimize.minimize(evaluator, next_angles)
        if -result.fun > objective_best + minimum_improvement:
            no_improvement_count = 0
            objective_best = -result.fun
            angles_best = result.x
        else:
            no_improvement_count += 1

    time_finish = time.perf_counter()
    logger.debug(f'Optimization done. Time elapsed: {time_finish - time_start}')
    return objective_best, angles_best
