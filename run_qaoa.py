""" Entry points for test single core uses. """
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.angle_strategies.direct import convert_angles_ma_to_controlled_ma
from src.angle_strategies.guess_provider import GuessProviderConstant
from src.angle_strategies.search_space import SearchSpaceControlled
from src.optimization.evaluator import Evaluator
from src.optimization.evaluator_qiskit import EvaluatorQiskit
from src.optimization.optimization import optimize_qaoa_angles
from src.preprocessing import evaluate_all_cuts


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

    cut_vals = evaluate_all_cuts(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/other/simple/bipartite_full_n6.gml')


def run_point():
    graph = nx.read_gml('graphs/main/nodes_12/depth_3/0.gml', destringizer=int)
    # graph = nx.graph_atlas(14)

    p = 1
    search_space = 'qaoa'

    point = np.array([0.059, -0.094]) * np.pi
    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    evaluator = EvaluatorQiskit.get_evaluator_standard_maxcut_qiskit_simulator(graph, p, search_space)
    print(evaluator.evaluate(point))


def run_point_ibm():
    # graph = nx.read_gml('graphs/other/simple/n3_e3.gml', destringizer=int)
    graph = nx.complete_graph(24)
    p = 1
    point = np.array([0.2, -0.2])
    search_space = 'qaoa'
    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    # evaluator = EvaluatorQiskit.get_evaluator_standard_maxcut_qiskit_simulator(graph, p, search_space=search_space)
    evaluator = EvaluatorQiskit.get_evaluator_standard_maxcut_qiskit_hardware(graph, p, search_space=search_space)

    for i in range(1):
        print(evaluator.evaluate(point))


def run_optimization():
    graph = nx.read_gml('graphs/main/nodes_12/depth_3/0.gml', destringizer=int)
    # graph = nx.read_gml('graphs/other/simple/line_n3.gml', destringizer=int)
    # graph = nx.graph_atlas(14)

    # target_vals = np.array([0, 5, 1, 0])
    # driver_term_vals = np.diag(target_vals)
    p = 1
    search_space = 'qaoa'

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, search_space)
    # fixed_angles = np.array([-0.25] * len(graph.edges) + [0, 0, 0.25, 0.25]) * np.pi
    # evaluator.fix_params(list(range(len(fixed_angles))), fixed_angles)

    guess_provider = GuessProviderConstant()
    starting_angles = guess_provider.provide_guess(evaluator)[0]
    # starting_angles = np.array([np.pi / 4, -np.pi / 4])
    # starting_angles = np.array([-0.25] * len(graph.edges) + [0.25] * len(graph) ** 2) * np.pi
    # starting_angles = None

    res = optimize_qaoa_angles(evaluator, starting_angles, method='SLSQP')
    print(res)
    print(repr(res.x / np.pi))


def run_draw_graph():
    graph = nx.read_gml('graphs/main/nodes_9/depth_3/0.gml', destringizer=int)
    nx.draw(graph, with_labels=True)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    logger.setLevel(logging.WARNING)
    np.set_printoptions(linewidth=160, formatter={'float': lambda x: '{:.3f}'.format(x)})

    # Select procedure to run below
    start = time.perf_counter()
    # run_add_graph()
    # run_point_ibm()
    run_point()
    # run_optimization()
    # run_draw_graph()
    end = time.perf_counter()
    print(f'Elapsed time: {end - start}')
