"""
Example uses.
"""
import itertools as it
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm

from src.graph_utils import get_index_edge_list
from src.optimization import optimize_qaoa_angles, Evaluator
from src.parameter_reduction import generate_all_duplication_schemes_p1_22, convert_angles_qaoa_to_multi_angle
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def add_graph():
    # k = 8
    # n = 15
    # g = nx.random_regular_graph(k, n)

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 4)
    g.add_edge(1, 5)
    g.add_edge(4, 5)
    g.add_edge(2, 6)

    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/simple/n7_e7.gml')


def run_point():
    graph = nx.read_gml('graphs/simple/n6_e6.gml', destringizer=int)
    p = 1
    # num_angles = (len(graph) + len(graph.edges)) * p
    # angles = np.array([-0.25, -0.25, -0.25, 0, 0, -0.25, 0, 0.25, 0.25, 0.25, 0.25, 0]) * np.pi
    angles = np.array([-0.250, -0.084, 0.250, 0.000, 0.000, -0.180, 0.250, 0.000, 0.250, 0.159, -0.250, 0.111, -0.361, 0.102]) * np.pi

    logger.debug('Preprocessing started...')
    time_start = time.perf_counter()

    # target_vals = evaluate_graph_cut(graph)
    # driver_term_vals = np.array([evaluate_z_term(np.array([term]), len(graph)) for term in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
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
    graph = nx.read_gml('graphs/nodes_8/1000.gml', destringizer=int)  # maxcut = 8
    # graph = nx.read_gml('graphs/simple/reg4_n6_e12.gml', destringizer=int)
    nx.draw(graph, with_labels=True)
    plt.show()


def run_angle_grouping():
    graph = nx.read_gml('graphs/nodes_8/1000.gml', destringizer=int)  # MC=8, MA1=7.577, Q2=7.342
    # graph = read_graph_xqaoa('graphs/xqaoa/G4#128_24.csv')  # maxcut = 224, xqaoa = 222, full = 216 (516?), reduced = 210 (366)
    p = 1

    target_vals = evaluate_graph_cut(graph)
    edges = [edge for edge in get_index_edge_list(graph)]
    driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in edges])

    # duplication_scheme = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
    # edge_groups = [[0, 1, 2, 5], [3, 4]]
    # node_groups = [[0, 4], [1, 2, 3, 5]]
    # shift = len(edges)
    # node_groups = [np.array(item) + shift for item in node_groups]
    # duplication_scheme = [*edge_groups, *node_groups]
    # duplication_scheme = [np.array(item) for item in duplication_scheme]
    # evaluator = Evaluator.get_evaluator_general_scheme(target_vals, driver_term_vals, p, duplication_scheme)

    dup_schemes = generate_all_duplication_schemes_p1_22(len(graph.edges), len(graph))
    optimized = []
    for scheme in tqdm.tqdm(dup_schemes, smoothing=0):
        evaluator = Evaluator.get_evaluator_general_scheme(target_vals, driver_term_vals, p, scheme)
        objective_best, _ = optimize_qaoa_angles(evaluator, num_restarts=10)
        optimized.append(objective_best)
    print(max(optimized))


def run_optimization():
    graph = nx.read_gml('graphs/nodes_8/1000.gml', destringizer=int)  # MC=8, MA1=7.577, Q2=7.342
    # graph = read_graph_xqaoa('graphs/xqaoa/G4#128_24.csv')  # maxcut = 224, xqaoa = 222, full = 216 (516?), reduced = 210 (366)
    p = 1
    use_multi_angle = True

    # target_vals = evaluate_graph_cut(graph)
    # edges = [edge for edge in get_index_edge_list(graph)]
    # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in edges])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle=use_multi_angle)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # # driver_terms = [set(term) for term in it.chain(it.combinations(range(len(graph)), 1), it.combinations(range(len(graph)), 2))]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=use_multi_angle)

    # evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(graph, p)

    objective_best, angles_best = optimize_qaoa_angles(evaluator, num_restarts=1)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {repr(angles_best / np.pi)}')

    # expectations = calc_per_edge_expectation(angles_best, driver_term_vals, p, graph, use_multi_angle=use_multi_angle)
    print('Done')


def run_optimization_combo():
    graph = nx.read_gml('graphs/nodes_8/1000.gml', destringizer=int)  # MC=8, MA1=7.577, Q1=6.712, Q2=7.342
    p = 2

    evaluator_qaoa = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=False)
    objective_qaoa, angles_qaoa = optimize_qaoa_angles(evaluator_qaoa, num_restarts=10)

    evaluator_ma = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=True)
    starting_point = convert_angles_qaoa_to_multi_angle(angles_qaoa, len(graph.edges), len(graph))
    objective_ma, angles_ma = optimize_qaoa_angles(evaluator_ma, starting_point=starting_point)

    print(f'QAOA: {objective_qaoa}')
    print(f'MA-QAOA: {objective_ma}')
    print('Done')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    # logger.setLevel(logging.DEBUG)
    np.set_printoptions(linewidth=160, formatter={'float': lambda x: '{:.3f}'.format(x)})

    # add_graph()
    # run_point()
    run_optimization()
    # run_optimization_combo()
    # run_draw()
