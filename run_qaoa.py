"""
Example uses of the library.
"""
import itertools as it
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.graph_utils import get_index_edge_list
from src.optimization import optimize_qaoa_angles, Evaluator
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def add_graph():
    # k = 8
    # n = 15
    # g = nx.random_regular_graph(k, n)

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    g.add_edge(1, 5)
    g.add_edge(1, 6)
    g.add_edge(1, 7)

    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/simple/reg4_sub_tree.gml')


def run_point():
    graph = nx.read_gml('graphs/simple/reg3_sub_tree.gml', destringizer=int)
    p = 1
    # num_angles = (len(graph) + len(graph.edges)) * p
    # angles = np.array([-0.25, -0.25000006, 0.75000002, 0.25000006, 0.25000001, -0.25, -0.75000001, -0.74999999, -0.25000004, -0.25000003, 0.75000004, 0.25, 0.25000005,
    #                    0.75000006, 0.24999999, -0.25]) * np.pi
    angles = np.array([np.pi / 4] * 12)
    angles[[1, 2, 3]] *= -1

    logger.debug('Preprocessing started...')
    time_start = time.perf_counter()

    # target_vals = evaluate_graph_cut(graph)
    # driver_term_vals = np.array([evaluate_z_term(np.array([term]), len(graph)) for term in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

    target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
    # evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(graph, p)
    # evaluator = Evaluator.get_evaluator_standard_maxcut_analytical(graph)

    time_finish = time.perf_counter()
    logger.debug(f'Preprocessing finished. Time elapsed: {time_finish - time_start}')

    logger.debug('Evaluation started...')
    time_start = time.perf_counter()
    res = evaluator.func(angles)
    time_finish = time.perf_counter()
    logger.debug(f'Evaluation finished. Expectation value: {-res}. Time elapsed: {time_finish - time_start}')


def run_draw():
    # graph = nx.read_gml('graphs/nodes_8/4575.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/4633.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/2914.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/749.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/5956.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/7921.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/1000.gml', destringizer=int)  # maxcut = 8
    graph = nx.read_gml('graphs/simple/reg4_n6_e12.gml', destringizer=int)
    nx.draw(graph, with_labels=True)
    plt.show()


def run_optimization():
    # graph = nx.read_gml('graphs/simple/reg4_sub_tree.gml', destringizer=int)
    graph = nx.read_gml('graphs/nodes_8/1000.gml', destringizer=int)  # 6.712, 7.901
    # graph = read_graph_xqaoa('graphs/xqaoa/G4#128_24.csv')  # maxcut = 224, xqaoa = 222, full = 216 (516?), reduced = 210 (366)
    p = 6
    use_multi_angle = False

    target_vals = evaluate_graph_cut(graph)
    # edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]])
    edges = [edge for edge in get_index_edge_list(graph) if not (all(edge == [1, 5]))]
    # edges = [edge for edge in get_index_edge_list(graph)]
    driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in edges])
    # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # driver_term_vals = np.array([evaluate_z_term(term, len(graph)) for term in it.combinations(range(len(graph)), 2)])
    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle=use_multi_angle)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # # driver_terms = [set(term) for term in it.chain(it.combinations(range(len(graph)), 1), it.combinations(range(len(graph)), 2))]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    # evaluator = Evaluator.get_evaluator_general_z1_analytical(graph)
    # evaluator = Evaluator.get_evaluator_general_z1_analytical_reduced(graph)

    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, edge_list=[(0, 1)], use_multi_angle=False)

    # evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(graph, p)

    objective_best, angles_best = optimize_qaoa_angles(evaluator, num_restarts=60)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')

    # expectations = calc_per_edge_expectation(angles_best, driver_term_vals, p, graph, use_multi_angle=use_multi_angle)
    # print('Done')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.DEBUG)

    # add_graph()
    # run_point()
    run_optimization()
    # run_draw()
    # my_test()
