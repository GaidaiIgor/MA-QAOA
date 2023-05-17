"""
Example uses of the library.
"""
import logging
import time

import networkx as nx
import numpy as np

from src.optimization import optimize_qaoa_angles
from src.original_qaoa import qaoa_decorator
from src.preprocessing import preprocess_subgraphs, get_edge_cut
from src.simulation import calc_expectation_ma_qaoa_simulation, calc_expectation_ma_qaoa_simulation_subgraphs


def add_graph():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1)
    g.add_edge(0, 3, weight=1)
    g.add_edge(1, 4, weight=1)
    g.add_edge(1, 5, weight=1)
    g.add_edge(2, 6, weight=1)
    g.add_edge(2, 7, weight=1)
    g.add_edge(3, 8, weight=1)
    g.add_edge(3, 9, weight=1)
    g.add_edge(4, 10, weight=1)
    g.add_edge(4, 11, weight=1)
    g.add_edge(5, 12, weight=1)
    g.add_edge(5, 13, weight=1)
    g.add_edge(6, 14, weight=1)
    g.add_edge(6, 15, weight=1)
    g.add_edge(7, 16, weight=1)
    g.add_edge(7, 17, weight=1)
    g.add_edge(8, 18, weight=1)
    g.add_edge(8, 19, weight=1)
    g.add_edge(9, 20, weight=1)
    g.add_edge(9, 21, weight=1)

    g.graph['max_cut'] = 21
    nx.write_gml(g, 'graphs/simple/reg3_sub_tree_p2.5.gml')


def run_point_subgraphs():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree_p2.gml', destringizer=int)
    p = 2
    edge_list = None
    angles = np.array([np.pi / 8] * 2 * p)

    time_start = time.perf_counter()
    subgraphs = preprocess_subgraphs(graph, p, edge_list)
    time_finish = time.perf_counter()
    logger.debug(f'Preprocessing time: {time_finish - time_start}')

    time_start = time.perf_counter()
    qaoa_evaluator = qaoa_decorator(calc_expectation_ma_qaoa_simulation_subgraphs, len(graph.edges), len(graph))
    res = qaoa_evaluator(angles, p, subgraphs)
    time_finish = time.perf_counter()
    logger.debug(f'Expectation value: {res}')
    logger.debug(f'Evaluation time: {time_finish - time_start}')


def run_point():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree_p2.gml', destringizer=int)
    p = 2
    edge_list = None
    angles = np.array([np.pi / 8] * 2 * p)

    time_start = time.perf_counter()
    all_cuv_vals = np.array([get_edge_cut(edge, len(graph)) for edge in graph.edges])
    time_finish = time.perf_counter()
    logger.debug(f'Preprocessing time: {time_finish - time_start}')

    time_start = time.perf_counter()
    qaoa_evaluator = qaoa_decorator(calc_expectation_ma_qaoa_simulation, len(graph.edges), len(graph))
    res = qaoa_evaluator(angles, p, all_cuv_vals, edge_list)
    time_finish = time.perf_counter()
    logger.debug(f'Expectation value: {res}')
    logger.debug(f'Evaluation time: {time_finish - time_start}')


def run_optimization():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree_p3.gml', destringizer=int)
    p = 1
    edge_list = None
    use_multi_angle = False
    use_analytical = False
    use_subgraphs = True

    objective_best, angles_best = optimize_qaoa_angles(use_multi_angle, use_analytical, use_subgraphs, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.DEBUG)
    # add_graph()
    # run_point()
    # run_point_subgraphs()
    run_optimization()
