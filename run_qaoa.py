"""
Example uses of the library.
"""
import logging
import time

import networkx as nx
import numpy as np
import csv

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
    graph = nx.read_gml('graphs/nodes_8/0.gml', destringizer=int)
    p = 1
    # num_angles = (len(graph) + len(graph.edges)) * p
    angles = np.array([0.74999998, -0.25000006, 0.75000002, 0.25000006, 0.25000001, -0.25, -0.75000001, -0.74999999, -0.25000004, -0.25000003, 0.75000004, 0.25, 0.25000005,
                       0.75000006, 0.24999999, -0.25]) * np.pi

    logger.debug('Preprocessing started...')
    time_start = time.perf_counter()

    target_vals = evaluate_graph_cut(graph)
    driver_term_vals = np.array([evaluate_z_term(np.array([term]), len(graph)) for term in range(8)])
    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
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
    # graph = nx.read_gml('graphs/nodes_8/4575.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/4633.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/2914.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/749.gml', destringizer=int)
    # graph = nx.read_gml('graphs/nodes_8/5956.gml', destringizer=int)
    graph = nx.read_gml('graphs/nodes_8/7921.gml', destringizer=int)
    nx.draw(graph)
    plt.show()


def read_graph(base, D, N, I):
    edges = []
    with open(f"{base}/G{D}#{N}_{I}.csv") as graph_file:
        graph_reader = csv.reader(graph_file, delimiter=',')
        for i, row in enumerate(graph_reader):
            if i == 0:
                _, M, C_opt, MIPGap = row
            elif i == 1:
                solution = row
            else:
                i, j = row
                edges.append((int(i), int(j)))
    solution = [int(x) for x in solution]
    graph_attr = {"Nodes": N, "Edges": int(M), "Cost": float(C_opt),
                  "MIPGap": float(MIPGap), "Solution": solution}
    return graph_attr, edges


def run_optimization():
    graph_attr, edges = read_graph('graphs/simple', 10, 128, 1)
    graph = nx.from_edgelist(edges)
    # graph = nx.read_gml('graphs/nodes_8/0.gml', destringizer=int)
    p = 1

    # target_vals = evaluate_graph_cut(graph)
    # # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # # driver_term_vals = np.array([evaluate_z_term(term, len(graph)) for term in it.combinations(range(len(graph)), 2)])
    # driver_term_vals = np.array([evaluate_z_term(np.array([term]), len(graph)) for term in range(8)])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

    target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # driver_terms = [{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}]
    # driver_terms = [{0, 1, 2}, {0, 3}, {0, 4}, {0, 5}]
    driver_terms = [{term} for term in range(len(graph))]
    evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)

    # evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(graph, p)

    objective_best, angles_best = optimize_qaoa_angles(evaluator, num_restarts=100)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.DEBUG)

    # add_graph()
    # run_point()
    run_optimization()
    # run_draw()
