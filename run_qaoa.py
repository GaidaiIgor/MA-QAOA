"""
Example uses of the library.
"""
import itertools as it
import logging
import time

import networkx as nx
import numpy as np

from src.graph_utils import get_index_edge_list
from src.optimization import optimize_qaoa_angles, Evaluator
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def add_graph():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1)
    g.add_edge(0, 4, weight=1)
    g.add_edge(0, 6, weight=1)
    g.add_edge(0, 7, weight=1)
    g.add_edge(1, 2, weight=1)
    g.add_edge(1, 3, weight=1)
    g.add_edge(1, 5, weight=1)
    g.add_edge(1, 7, weight=1)
    g.add_edge(2, 3, weight=1)
    g.add_edge(2, 4, weight=1)
    g.add_edge(2, 6, weight=1)
    g.add_edge(3, 4, weight=1)
    g.add_edge(3, 5, weight=1)
    g.add_edge(3, 7, weight=1)
    g.add_edge(4, 5, weight=1)
    g.add_edge(4, 6, weight=1)
    g.add_edge(5, 6, weight=1)
    g.add_edge(5, 7, weight=1)
    g.add_edge(6, 7, weight=1)

    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, 'graphs/simple/reg5_n8_e20.gml')


def run_point():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree_p2.gml', destringizer=int)
    p = 2
    angles = np.array([np.pi / 8] * 2 * p)

    logger.debug('Preprocessing started...')
    time_start = time.perf_counter()
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=False)
    time_finish = time.perf_counter()
    logger.debug(f'Preprocessing finished. Time elapsed: {time_finish - time_start}')

    logger.debug('Evaluation started...')
    time_start = time.perf_counter()
    res = evaluator.func(angles)
    time_finish = time.perf_counter()
    logger.debug(f'Evaluation finished. Expectation value: {-res}. Time elapsed: {time_finish - time_start}')


def run_optimization():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree.gml', destringizer=int)
    p = 1
    target_vals = evaluate_graph_cut(graph)
    # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    driver_term_vals = np.array([evaluate_z_term(term, len(graph)) for term in it.combinations(range(len(graph)), 2)])
    use_multi_angle = True

    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle)
    objective_best, angles_best = optimize_qaoa_angles(evaluator)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.DEBUG)
    # add_graph()
    # run_point()
    run_optimization()
