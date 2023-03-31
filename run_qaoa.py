"""
Main entry point
"""
import networkx as nx
import numpy as np
import src.plot

from src.optimization import optimize_qaoa_angles
import logging

from src.original_qaoa import run_qaoa_analytical, run_qaoa_analytical_p1, run_qaoa_simulation


def run_main():
    multi_angle = False
    use_analytical = True
    p = 2
    graph = nx.read_gml('graphs/simple/reg3_sub_tree.gml', destringizer=int)
    edge_list = [(0, 1)]
    # edge_list = None

    # angles = np.array([np.pi / 8, np.pi / 8, np.pi / 8, np.pi / 8])
    # ans1 = run_qaoa_analytical(angles, p, graph, edge_list)
    # ans2 = run_qaoa_simulation(angles, p, graph, edge_list)
    # print(ans1, ans2)

    objective_best, angles_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')

    # src.plot.plot_qaoa_expectation_p1(graph, edge_list)


# TODO: starting angles range?
if __name__ == '__main__':
    logging.basicConfig()
    # logger = logging.getLogger('QAOA')
    # logger.setLevel(logging.DEBUG)
    run_main()
