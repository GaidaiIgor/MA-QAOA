"""
Examples of basic uses.
"""
import networkx as nx
import numpy as np

from src.optimization import Evaluator, optimize_qaoa_angles


def run_point():
    """ Evaluates cost expectation at the given QAOA angles. """
    graph = nx.read_gml('graphs/main/nodes_9/depth_3/0.gml', destringizer=int)
    p = 2
    # Could also be set to 'ma' for MA-QAOA. Other options are available as well, see Evaluator class for more details.
    search_space = 'qaoa'
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    # Angles are given in the order of application, i.e. gamma_1, beta_1, gamma_2, beta_2, etc.
    # In case of MA-QAOA the order of gammas and betas matches the order of graph.edges and graph.nodes
    angles = np.array([0.1] * evaluator.num_angles) * np.pi
    cut_expectation = -evaluator.func(angles)
    print(f'Expectation: {cut_expectation}')


def run_optimization():
    """ Optimizes cost expectation. """
    graph = nx.read_gml('graphs/main/nodes_9/depth_3/0.gml', destringizer=int)
    p = 2
    search_space = 'qaoa'
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    # Picks random starting angles if None are given
    result = optimize_qaoa_angles(evaluator, starting_angles=None, method='L-BFGS-B')
    print(f'Best achieved objective: {-result.fun}')
    print(f'Maximizing angles: {repr(result.x / np.pi)}')


if __name__ == '__main__':
    run_optimization()
