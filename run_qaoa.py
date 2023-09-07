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
from src.angle_strategies import generate_all_duplication_schemes_p1_22, convert_angles_qaoa_to_ma, convert_angles_tqa_qaoa
from src.preprocessing import evaluate_graph_cut, evaluate_z_term
from src.simulation.qiskit_backend import evaluate_angles_ma_qiskit, optimize_angles_ma_qiskit


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


def run_draw():
    graph = nx.read_gml('graphs/nodes_8/ed_4/1.gml', destringizer=int)
    nx.draw(graph, with_labels=True)
    plt.show()


def run_angle_grouping():
    graph = nx.read_gml('graphs/all_8/1000.gml', destringizer=int)  # MC=8, MA1=7.577, Q2=7.342
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


def run_gradient():
    graph = nx.read_gml('graphs/all_8/3054.gml', destringizer=int)
    angles = np.array([0.250, 0.250, -0.250, -0.418, -0.250, 1.121, 0.250, 0.750]) * np.pi

    grad = np.zeros_like(angles)
    for i in range(len(grad)):
        for edge in graph.edges(i):
            grad[i] += np.sin(2 * angles[edge[1]])
        grad[i] *= np.cos(2 * angles[i])

    hessian = np.zeros((len(angles), len(angles)))
    for i in range(len(angles)):
        for j in range(len(angles)):
            if i == j:
                for edge in graph.edges(i):
                    hessian[i, i] += np.sin(2 * angles[edge[1]])
                hessian[i, i] *= -2 * np.sin(2 * angles[i])
            else:
                if not graph.has_edge(i, j):
                    continue
                hessian[i, j] = 2 * np.cos(2 * angles[i]) * np.cos(2 * angles[j])
    print('Done')


def run_optimization():
    graph = nx.read_gml('graphs/nodes_8/ed_4/20.gml', destringizer=int)
    # graph = nx.complete_graph(3)
    # graph.add_edge(1, 4)
    p = 3
    search_space = 'ma'

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

    # starting_point = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    #                            -0.2, -0.2, -0.2, -0, -0.2, -0.2, -0.2, -0.2]) * np.pi

    # starting_point = np.array([0.25, 0.5, 0.5, 0, 0.5, 0, 0.25, -0.25, 0.5, -0.25, 0.5, -0.25, -0.25,
    #                            -0.25, 0.25, 0.25, 0, -0.25, -0.25, 0.25, 0]) * np.pi

    starting_point = np.array(
        [2.73049202e+00, -1.57079557e+00, 8.23669261e-01, -5.37994478e-08, -1.57079787e+00, 1.57079536e+00, 1.75875056e+00, 3.01317851e+00, 1.57079581e+00, -5.13436257e-01,
         -1.36538678e+00, -1.42626267e+00, 2.52977539e+00, -1.19945168e+00, -3.14769144e+00, 2.53652760e-01, -1.12245459e+00, 7.26598795e-01, 1.57079710e+00, -2.12580519e-01,
         1.57079655e+00, 9.46241383e-07, -7.85398067e-01, -1.06895980e+00, -2.51324309e+00, 9.46493096e-01, 2.78648794e+00, 9.63720308e-01, 2.35619501e+00, 1.92767952e+00,
         3.58012439e-01, 2.11307651e+00, 1.57079670e+00, -1.29482715e-01, -2.86662339e+00, 3.04281729e+00, 2.77724479e+00, -5.43541231e-01, -1.95934930e+00, -4.51499814e-01,
         -1.11106016e-01, 2.32252708e+00, 1.26064702e+00, 1.57079668e+00, 2.85394767e+00, 1.58945341e-06, 2.66596677e+00, 2.35619394e+00, 2.30458336e+00, -1.57079535e+00,
         9.67225912e-01])
    starting_point = np.array(
        [-0.25, 0.5, 0.25, -0., 0.5, 0.5, 0.5, 0., 0.5, -0.25, 0.5, 0.5, -0.25, 0.5, 0., 0., -0.25, 0.25, 0.5, -0., 0.5, 0., -0.25, -0.25, 0.25, 0.25, 0., 0.25, -0.25, 0.5, 0.,
         -0.25, 0.5, -0., 0., 0., 0., -0.25, 0.5, -0.25, -0., -0.25, 0.5, 0.5, 0., 0., -0.25, -0.25, -0.25, 0.5, 0.25]) * np.pi

    # starting_point = np.array([-0.005, -0.072, 0.000, -0.000, 0.250, 0.000, 0.081, 0.048, 0.024, -0.085, 0.079, -0.250, 0.131, -0.000, 0.001, -0.461, 0.009, 0.005, -0.187, 0.000,
    #                            -0.250, 0.071, 0.128, -0.045, 0.184, 0.019, 0.286, 0.163, 0.000, 0.250, 0.000, -0.228, -0.072, -0.119, 0.251, 0.240, 0.054, 0.110, 0.141, 0.033, 0.250,
    #                            0.091, 0.045, 0.000, -0.249, 0.000, -0.062, 0.000, -0.030, -0.199, -0.250])

    # starting_point = np.array([-0.000, -0.000, 0.000, -0.000, 0.250, 0.000, 0.000, 0.000, 0.000, -0.000, 0.000, -0.250, 0.250, -0.000, 0.000, -0.500, 0.000, 0.000, -0.250, 0.000,
    #    -0.250, 0.000, 0.250, -0.000, 0.250, 0.000, 0.250, 0.250, 0.000, 0.250, 0.000, -0.250, -0.000, -0.000, 0.250, 0.250, 0.000, 0.000, 0.250, 0.000, 0.250,
    #    0.000, 0.000, 0.000, -0.250, 0.000, -0.000, 0.000, -0.000, -0.250, -0.250])

    # fix_inds = [0, 1, 2]
    # fix_vals = starting_point[fix_inds]
    # evaluator.fix_params(fix_inds, fix_vals)
    # mask = np.ones_like(starting_point, dtype=bool)
    # mask[fix_inds] = False
    # starting_point = starting_point[mask]

    objective_best, angles_best = optimize_qaoa_angles(evaluator, starting_point=starting_point, method='BFGS', options={'disp': True})

    # objective_best = optimize_angles_ma_qiskit(graph, p)

    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {repr(angles_best / np.pi)}')

    # expectations = calc_per_edge_expectation(angles_best, driver_term_vals, p, graph, use_multi_angle=use_multi_angle)
    print('Done')


def run_optimization_combo():
    graph = nx.read_gml('graphs/all_8/3054.gml', destringizer=int)
    p = 1

    evaluator_1 = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space='tqa')
    objective_1, angles_1 = optimize_qaoa_angles(evaluator_1, starting_point=np.array([0.5]))
    print(f'obj: {objective_1}; angles: {angles_1}')

    evaluator_2 = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space='regular')
    starting_point = convert_angles_tqa_qaoa(angles_1, p)
    objective_2, angles_2 = optimize_qaoa_angles(evaluator_2, starting_point=starting_point)

    print(f'TQA: {objective_1}; {angles_1}')
    print(f'QAOA: {objective_2}; {angles_2}')
    print('Done')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    # logger.setLevel(logging.DEBUG)
    np.set_printoptions(linewidth=160, formatter={'float': lambda x: '{:.3f}'.format(x)})

    start = time.perf_counter()
    # add_graph()
    run_point()
    # run_optimization()
    # run_optimization_combo()
    # run_draw()
    # run_gradient()
    end = time.perf_counter()
    print(f'Elapsed time: {end - start}')
