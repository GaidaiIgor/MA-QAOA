"""
Main entry point
"""
import logging
import time

import networkx as nx
import numpy as np

import src.preprocessing as pr
from src.optimization import optimize_qaoa_angles
from src.original_qaoa import run_qaoa_simulation


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


def run_point():
    p = 2
    graph = nx.read_gml('graphs/simple/reg3_sub_tree_p2.5.gml', destringizer=int)
    edge_list = None
    angles = np.array([np.pi / 8] * 2 * p)

    time_start = time.perf_counter()
    all_labelings = pr.get_all_binary_labelings(len(graph))
    all_cuv_vals = np.array([[pr.check_edge_cut(labeling, u, v) for labeling in all_labelings] for (u, v) in graph.edges])
    time_finish = time.perf_counter()
    logger.debug(f'Preprocessing time: {time_finish - time_start}')

    time_start = time.perf_counter()
    res = run_qaoa_simulation(angles, p, all_cuv_vals, edge_list)
    time_finish = time.perf_counter()
    logger.debug(f'Expectation value: {res}')
    logger.debug(f'Evaluation time: {time_finish - time_start}')


def run_optimization():
    multi_angle = False
    p = 2
    graph = nx.read_gml('graphs/simple/reg3_sub_tree_p2.gml', destringizer=int)
    edge_list = None

    use_analytical = p == 1

    objective_best, angles_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.DEBUG)
    # add_graph()
    run_point()
    # run_optimization()
