import time

import networkx as nx
import numpy as np
import pandas as pd
import itertools as it
from multiprocessing import Pool
from functools import partial
from pandas import DataFrame
from tqdm import tqdm

from src.optimization import Evaluator, optimize_qaoa_angles
from src.graph_utils import get_edge_diameter, get_index_edge_list
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


def calculate_general_qaoa(path, driver_term_vals, p):
    graph = nx.read_gml(path, destringizer=int)
    target_vals = evaluate_graph_cut(graph)
    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)
    return optimize_qaoa_angles(evaluator)


def run_node_8_general():
    p = 1
    num_workers = 20
    graph_size = 8

    df = pd.read_csv('output.csv', index_col=0)
    # keys = list(df.index)
    keys = list(df.loc[abs(df['maxcut'] - df[f'exp_general_p{p}']) > 1e-3, :].index)
    paths = [f'graphs/{key}' for key in keys]

    driver_term_vals = np.array([evaluate_z_term(np.array([term]), graph_size) for term in range(graph_size)])
    pool_func = partial(calculate_general_qaoa, driver_term_vals=driver_term_vals, p=p)

    result = []
    with Pool(num_workers) as pool:
        for item in tqdm(pool.imap(pool_func, paths), total=len(keys), smoothing=0, ascii=' █'):
            result.append(item)

    result = DataFrame(result, index=keys, columns=[f'exp_general_p{p}', f'angles_general_p{p}'])
    # df = df.join(result)
    df.update(result)
    df.to_csv('output2.csv')


def calculate_general_qaoa_sub(path, driver_terms, p):
    graph = nx.read_gml(path, destringizer=int)
    target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)
    return optimize_qaoa_angles(evaluator)


def run_node_8_general_sub():
    p = 1
    num_workers = 20
    graph_size = 8

    df = pd.read_csv('output.csv', index_col=0)
    keys = list(df.index)
    # keys = list(df.loc[abs(df['maxcut'] - df[f'exp_qaoa_p{p - 1}']) > 1e-3, :].index)
    paths = [f'graphs/{key}' for key in keys]

    driver_terms = [{term} for term in range(graph_size)]
    pool_func = partial(calculate_general_qaoa_sub, driver_terms=driver_terms, p=p)

    result = []
    with Pool(num_workers) as pool:
        for item in tqdm(pool.imap(pool_func, paths), total=len(keys), smoothing=0):
            result.append(item)

    result = DataFrame(result, index=keys, columns=[f'exp_sub_p{p}', f'angles_sub_p{p}'])
    df = df.join(result)
    df.to_csv('output2.csv')


def calculate_standard_qaoa(path, p):
    graph = nx.read_gml(path, destringizer=int)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=False)
    return optimize_qaoa_angles(evaluator)


def run_node_8_standard_qaoa():
    p = 9
    num_workers = 20

    df = pd.read_csv('output.csv', index_col=0)
    keys = list(df.loc[abs(df['maxcut'] - df[f'exp_qaoa_p{p - 1}']) > 1e-3, :].index)
    paths = [f'graphs/{key}' for key in keys]

    time_start = time.perf_counter()
    result = []
    pool_func = partial(calculate_standard_qaoa, p=p)
    with Pool(num_workers) as pool:
        for item in tqdm(pool.imap(pool_func, paths), total=len(keys)):
            result.append(item)
    time_finish = time.perf_counter()

    result = DataFrame(result, index=keys, columns=[f'exp_qaoa_p{p}', f'angles_qaoa_p{p}'])
    df = df.join(result)
    print(f'Done. Time elapsed: {time_finish - time_start}')
    df.to_csv('output2.csv')


if __name__ == '__main__':
    # extend_csv()
    run_node_8_general()
    # run_node_8_standard_pool()
