import time

import networkx as nx
import numpy as np
import pandas as pd
import itertools as it

from src.optimization import Evaluator, optimize_qaoa_angles
from src.graph_utils import get_edge_diameter
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def extend_csv():
    df = pd.read_csv('output.csv', index_col=0)

    graph_count = 11117
    edge_diameters = []
    for i in range(graph_count):
        print(i)
        graph = nx.read_gml(f'graphs/nodes_8/{i}.gml', destringizer=int)
        edge_diameter = get_edge_diameter(graph)
        edge_diameters.append(edge_diameter)

    df.insert(3, 'edge_diameter2', edge_diameters)
    df.to_csv('output2.csv')


def run_node_8_graphs_general():
    graph_count = 11117
    p = 1
    df = pd.read_csv('output.csv', index_col=0)

    num_nodes = 8
    driver_term_vals = np.array([evaluate_z_term(term, num_nodes) for term in it.combinations(range(num_nodes), 1)])
    time_start = time.perf_counter()
    for i in range(graph_count):
        key = f'nodes_8/{i}.gml'
        # if df.loc[key, 'maxcut'] - df.loc[key, 'exp_p2'] < 1e-2:
        #     df.loc[key, 'exp_p3'] = df.loc[key, 'exp_p2']
        #     continue

        print(i)
        graph = nx.read_gml(f'graphs/{key}', destringizer=int)
        target_vals = evaluate_graph_cut(graph)
        evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

        objective_best, angles_best = optimize_qaoa_angles(evaluator)
        df.loc[key, 'exp_general_p1'] = objective_best
        df.loc[key, 'angles_general_p1'] = str(angles_best)

    time_finish = time.perf_counter()
    print(f'Done. Time elapsed: {time_finish - time_start}')
    df.to_csv('output2.csv')


def run_node_8_graphs_standard():
    graph_count = 11117
    p = 4
    df = pd.read_csv('output.csv', index_col=0)

    time_start = time.perf_counter()
    for i in range(graph_count):
        key = f'nodes_8/{i}.gml'
        if df.loc[key, 'maxcut'] - df.loc[key, 'exp_p3'] < 1e-3:
            df.loc[key, 'exp_p4'] = df.loc[key, 'exp_p3']
            continue

        print(i)
        graph = nx.read_gml(f'graphs/{key}', destringizer=int)
        evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
        objective_best, angles_best = optimize_qaoa_angles(evaluator)
        df.loc[key, 'exp_p4'] = objective_best
        df.loc[key, 'angles_p4'] = str(angles_best)

    time_finish = time.perf_counter()
    print(f'Done. Time elapsed: {time_finish - time_start}')
    df.to_csv('output2.csv')


# extend_csv()
run_node_8_graphs_general()
# run_node_8_graphs_standard()
