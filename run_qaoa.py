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
import matplotlib.pyplot as plt


def add_graph():
    k = 8
    n = 15
    g = nx.random_regular_graph(k, n)
    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/simple/reg{k}_n{n}_e{len(g.edges)}.gml')


def run_point():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree.gml', destringizer=int)
    p = 1
    num_angles = (len(graph) + len(graph.edges)) * p
    angles = np.linspace(np.pi / 16, np.pi / 8, num_angles)

    logger.debug('Preprocessing started...')
    time_start = time.perf_counter()

    # target_vals = evaluate_graph_cut(graph)
    # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # use_multi_angle = True
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
    # evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(graph, p)
    # evaluator = Evaluator.get_evaluator_analytical(graph)

    time_finish = time.perf_counter()
    logger.debug(f'Preprocessing finished. Time elapsed: {time_finish - time_start}')

    logger.debug('Evaluation started...')
    time_start = time.perf_counter()
    res = evaluator.func(angles)
    time_finish = time.perf_counter()
    logger.debug(f'Evaluation finished. Expectation value: {-res}. Time elapsed: {time_finish - time_start}')


def run_draw():
    graph = nx.read_gml('graphs/nodes_8/4.gml', destringizer=int)
    nx.draw(graph)
    plt.show()


def run_optimization():
    graph = nx.read_gml('graphs/simple/reg4_n7_e14.gml', destringizer=int)
    p = 1

    # target_vals = evaluate_graph_cut(graph)
    # # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # # driver_term_vals = np.array([evaluate_z_term(term, len(graph)) for term in it.combinations(range(len(graph)), 2)])
    # driver_term_vals = np.array([evaluate_z_term(term, len(graph)) for term in it.combinations(range(len(graph)), 1)])
    # use_multi_angle = True
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # # driver_terms = [{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}]
    # # driver_terms = [{0, 1, 2}, {0, 3}, {0, 4}, {0, 5}]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=False)

    # evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(graph, p, use_multi_angle=False)

    objective_best, angles_best = optimize_qaoa_angles(evaluator)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.DEBUG)
    # add_graph()
    run_point()
    # run_optimization()
    # run_draw()
