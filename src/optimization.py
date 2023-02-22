import logging
import time
from typing import Callable, Any

import numpy as np
import scipy.optimize as optimize
from networkx import Graph

import src.preprocessing as pr
from src.analytical import run_ma_qaoa_analytical_p1
from src.classical_qaoa import run_qaoa_analytical_p1, run_qaoa_simulation
from src.simulation import run_ma_qaoa_simulation


def change_sign(func: Callable[[Any, ...], int | float]) -> Callable[[Any, ...], int | float]:
    """
    Decorator to change sign of the return value of a given function. Useful to carry out maximization instead of minimization.
    :param func: Function whose sign is to be changed
    :return: Function with changed sign
    """
    def func_changed_sign(*args, **kwargs):
        return -func(*args, **kwargs)

    return func_changed_sign


def optimize_qaoa_angles(multi_angle: bool, use_analytical: bool, p: int, graph: Graph, edge_list: list[tuple[int, int]] = None) -> float:
    """
    Runs QAOA angle optimization
    :param multi_angle: True to use individual angles for each node and edge of the graph (MA-QAOA)
    :param use_analytical: True to use analytical expression to evaluate expectation value (available for p=1 only)
    :param p: Number of QAOA layers
    :param graph: Graph for which MaxCut problem is being solved
    :param edge_list: List of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Maximum expectation value achieved during optimization
    """
    max_no_improvements = 3
    assert not use_analytical or p == 1, "Cannot use analytical for p != 1"

    if not use_analytical:
        logging.debug('Preprocessing...')
        time_start = time.perf_counter()
        neighbours = pr.get_neighbour_labelings(len(graph))
        all_labelings = pr.get_all_binary_labelings(len(graph))
        all_cuv_vals = np.array([[pr.check_edge_cut(labeling, u, v) for labeling in all_labelings] for (u, v) in graph.edges])
        all_edge_list = list(graph.edges)
        edge_inds = None if edge_list is None else [all_edge_list.index(edge) for edge in edge_list]
        time_finish = time.perf_counter()
        logging.debug(f'Preprocessing done. Time elapsed: {time_finish - time_start}')

    logging.debug('Optimization...')
    time_start = time.perf_counter()
    num_angles_per_layer = len(graph.edges) + len(graph) if multi_angle else 2
    angles_best = np.zeros(num_angles_per_layer * p)
    objective_best = 0
    no_improvement_count = 0

    while no_improvement_count < max_no_improvements:
        next_angles = np.random.uniform(-np.pi, np.pi, len(angles_best))
        if use_analytical:
            if multi_angle:
                result = optimize.minimize(change_sign(run_ma_qaoa_analytical_p1), next_angles, (graph, edge_list))
            else:
                result = optimize.minimize(change_sign(run_qaoa_analytical_p1), next_angles, (graph, edge_list))
        else:
            if multi_angle:
                result = optimize.minimize(change_sign(run_ma_qaoa_simulation), next_angles, (p, all_cuv_vals, neighbours, all_labelings, edge_inds))
            else:
                result = optimize.minimize(change_sign(run_qaoa_simulation), next_angles, (p, all_cuv_vals, neighbours, all_labelings, edge_inds))

        if -result.fun > objective_best:
            no_improvement_count = 0
            objective_best = -result.fun
            angles_best = next_angles / np.pi
        else:
            no_improvement_count += 1

    time_finish = time.perf_counter()
    logging.debug(f'Optimization done. Runtime: {time_finish - time_start}')
    return objective_best
