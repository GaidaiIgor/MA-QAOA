"""
Entry points for test single core uses.
"""
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data_processing import numpy_str_to_array
from src.optimization import optimize_qaoa_angles, Evaluator
from src.preprocessing import evaluate_graph_cut


def run_add_graph():
    # k = 8
    # n = 15
    # g = nx.random_regular_graph(k, n)

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    # g.add_edge(0, 3)
    # g.add_edge(1, 4)
    # g.add_edge(1, 5)
    # g.add_edge(4, 5)
    # g.add_edge(2, 6)

    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/simple/n7_e7.gml')


def run_point():
    # graph = nx.complete_graph(5)
    graph = nx.read_gml('graphs/nodes_8/ed_4/20.gml', destringizer=int)
    p = 3

    starting_point = np.array([-0.000, -0.000, 0.000, -0.000, 0.250, 0.000, 0.000, 0.000, 0.000, -0.000, 0.000, -0.250, 0.250, -0.000, 0.000, 0.500, 0.000, 0.000, -0.250, 0.000,
                               -0.250, 0.000, 0.250, -0.000, 0.250, 0.000, 0.250, 0.250, 0.000, 0.250, 0.000, -0.250, -0.000, -0.000, 0.250, 0.250, 0.000, 0.000, 0.250, 0.000,
                               0.250, 0.000, 0.000, 0.000, -0.250, 0.000, -0.000, 0.000, -0.000, -0.250, -0.250]) * np.pi

    starting_point = np.array([-0.25, 0.5, 0.25, -0., 0.5, 0.5, 0.5, 0., 0.5, -0.25, 0.5, 0.5, -0.25, 0.5, 0., 0., -0.25, 0.25, 0.5, -0., 0.5, 0., -0.25, -0.25, 0.25, 0.25, 0.,
                               0.25, -0.25, 0.5, 0., -0.25, 0.5, -0., 0., 0., 0., -0.25, 0.5, -0.25, -0., -0.25, 0.5, 0.5, 0., 0., -0.25, -0.25, -0.25, 0.5, 0.25]) * np.pi

    # target_vals = evaluate_graph_cut(graph)
    # driver_term_vals = np.array([evaluate_z_term(np.array([term]), len(graph)) for term in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
    res = -evaluator.func(starting_point)

    # res = evaluate_angles_ma_qiskit(angles, graph, p)

    print(f'Expectation: {res}')


def run_optimization():
    graph = nx.read_gml('graphs/new/nodes_9/depth_3/0.gml', destringizer=int)
    # graph = nx.complete_graph(3)
    # graph = read_graph_xqaoa('graphs/xqaoa/G6#128_1.csv')
    p = 2
    search_space = 'qaoa'

    # target_vals = evaluate_graph_cut(graph)
    # edges = [edge for edge in get_index_edge_list(graph)]
    # # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in edges])
    # driver_term_vals = np.array([evaluate_z_term(np.array([node]), len(graph)) for node in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle=use_multi_angle)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # # driver_terms = [set(term) for term in it.chain(it.combinations(range(len(graph)), 1), it.combinations(range(len(graph)), 2))]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    # evaluator = Evaluator.get_evaluator_qiskit_fast(graph, p, search_space)
    # evaluator = Evaluator.get_evaluator_standard_maxcut_analytical(graph, use_multi_angle=True)

    starting_point = numpy_str_to_array('[-0.17993277 -1.30073361 -1.08469108 -1.59744761]')
    # starting_point = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))

    result = optimize_qaoa_angles(evaluator, starting_angles=starting_point)

    print(f'Best achieved objective: {-result.fun}')
    print(f'Maximizing angles: {repr(result.x / np.pi)}')

    # expectations = calc_per_edge_expectation(angles_best, driver_term_vals, p, graph, use_multi_angle=use_multi_angle)
    print('Done')


def run_draw_graph():
    graph = nx.read_gml('graphs/nodes_8/ed_4/1.gml', destringizer=int)
    nx.draw(graph, with_labels=True)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    np.set_printoptions(linewidth=160, formatter={'float': lambda x: '{:.3f}'.format(x)})

    # Select procedure to run below
    start = time.perf_counter()
    # run_add_graph()
    # run_point()
    run_optimization()
    # run_draw_graph()
    end = time.perf_counter()
    print(f'Elapsed time: {end - start}')
