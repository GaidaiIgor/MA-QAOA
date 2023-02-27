"""
Main entry point
"""
import networkx as nx
import numpy as np

from src.optimization import optimize_qaoa_angles


def run_main():
    multi_angle = False
    use_analytical = True
    p = 1
    graph = nx.read_gml('graphs/simple/reg3_sub_triangle.gml', destringizer=int)
    edge_list = [(0, 1)]
    # edge_list = None

    objective_best, angles_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')
    print(f'Maximizing angles: {angles_best / np.pi}')


# TODO: starting angles range?
if __name__ == '__main__':
    run_main()
