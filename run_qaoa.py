"""
Entry points for test single core uses.
"""
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.angle_strategies import SearchSpace
from src.graph_utils import get_index_edge_list
from src.optimization import optimize_qaoa_angles, Evaluator
from src.preprocessing import evaluate_graph_cut, evaluate_z_term

import numdifftools as nd
import src.optimization as optimization
import scipy.linalg as linalg


def run_add_graph():
    g = nx.Graph()
    for i in range(6):
        g.add_node(i)
    g.add_edge(0, 1)
    g.add_edge(0, 3)
    g.add_edge(0, 5)
    g.add_edge(2, 1)
    g.add_edge(2, 3)
    g.add_edge(2, 5)
    g.add_edge(4, 1)
    g.add_edge(4, 3)
    g.add_edge(4, 5)

    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/other/simple/bipartite_full_n6.gml')


def run_point():
    graph = nx.read_gml('graphs/other/simple/n3_e3.gml', destringizer=int)
    p = 1

    for i in range(10):
        print(f'Run: {i}')
        # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space='ma')
        evaluator = Evaluator.get_evaluator_standard_maxcut_qiskit_simulator(graph, p, search_space='ma')
        # evaluator = Evaluator.get_evaluator_standard_maxcut_qiskit_hardware(graph, p, search_space='ma')
        point = np.array(np.linspace(-np.pi, np.pi, evaluator.num_angles))
        res = evaluator.evaluate(point)
        print(f'Result: {res}')


def run_optimization():
    graph = nx.read_gml('graphs/other/simple/n8_crand_example.gml', destringizer=int)
    p = 1

    # search_space = 'qaoa'
    # starting_point = np.array([0.2, -0.2] * p)

    search_space = 'ma'
    starting_point = np.array([0.2] * 10 + [-0.2] * 8)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)

    gradient = nd.Gradient(evaluator.evaluate, 0.01)(starting_point)
    # result = optimize_qaoa_angles(evaluator, starting_angles=starting_point, options={'maxiter': 1})
    # print(f'Budget: {result.nfev}')
    # gradient = result.x - starting_point

    num_attempts = 1000
    stats = np.zeros((2, num_attempts))
    for i in range(num_attempts):
        search_space = SearchSpace(np.array([gradient]), starting_point)
        search_space.augment_basis(14)
        evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
        starting_point_new = np.array([0] * evaluator.num_angles)
        result = optimize_qaoa_angles(evaluator, starting_angles=starting_point_new)
        stats[0, i] = -result.fun / graph.graph["max_cut"]
        stats[1, i] = result.nfev
    stats_avg = [np.mean(stats[0, :]), np.min(stats[0, :]), np.max(stats[0, :]), np.mean(stats[1, :]), np.min(stats[1, :]), np.max(stats[1, :])]
    print(stats_avg)

    # target_vals = evaluate_graph_cut(graph)
    # driver_edges = np.array([[0, 1], [2, 3], [4, 5]])
    # driver_term_vals = np.array([evaluate_z_term(driver_edge, len(graph)) for driver_edge in driver_edges])
    # # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # # driver_term_vals = np.array([evaluate_z_term(np.array([node]), len(graph)) for node in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, search_space)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # # driver_terms = [set(term) for term in it.chain(it.combinations(range(len(graph)), 1), it.combinations(range(len(graph)), 2))]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    # result = optimize_qaoa_angles(evaluator, starting_angles=starting_point)
    # print(f'Budget: {result.nfev} + Preprocessing: {optimization.counter}')
    # print(f'Best AR: {-result.fun / graph.graph["max_cut"]}')
    # print(f'Maximizing angles: {repr(result.x)}')
    # print('Done')

    # evaluator = Evaluator.get_evaluator_standard_maxcut_qiskit_simulator(graph, p, search_space)
    # results = [None] * 1
    # for i in range(len(results)):
    #     print(f'\rIteration: {i}', end='')
    #     results[i] = optimize_qaoa_angles(evaluator, starting_angles=starting_point, method='cobyla')
    # print(f'\nAvg func: {np.mean([-res.fun for res in results])}')
    # print(f'Std func: {np.std([-res.fun for res in results])}')
    # print(f'Avg budget: {np.mean([res.nfev for res in results])}')


def run_draw_graph():
    graph = nx.read_gml('graphs/main/nodes_9/depth_3/0.gml', destringizer=int)
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
