import time

import networkx as nx
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from pandas import DataFrame
from tqdm import tqdm
import os.path as path
import itertools as it
import glob
from pathlib import Path


from src.optimization import Evaluator, optimize_qaoa_angles
from src.graph_utils import get_edge_diameter, get_index_edge_list, read_graph_xqaoa
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def collect_results_from(base_path, columns, aggregator):
    paths = glob.glob(f'{base_path}/*.csv')
    stat = []
    for path in paths:
        df = pd.read_csv(path)
        stat.append(aggregator(df[columns], axis=0))
    index_keys = [Path(path).parts[-1] for path in paths]
    summary_df = DataFrame(stat, columns=columns, index=index_keys)
    return summary_df


def collect_results_xqaoa():
    aggregator = np.mean
    df_gqaoa = collect_results_from('graphs/xqaoa/output', ['GQAOA'], aggregator)
    df_xqaoa = collect_results_from('simulation_data', ['XQAOA', 'Geomans_Williamson'], aggregator)
    return df_gqaoa.join(df_xqaoa)


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


def worker_general_qaoa(path, reader, p):
    graph = reader(path)
    target_vals = evaluate_graph_cut(graph)
    driver_term_vals = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 1)])
    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)
    return path, *optimize_qaoa_angles(evaluator, num_restarts=1)


def worker_general_qaoa_sub(path, reader, p):
    graph = reader(path)
    target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)
    return path, *optimize_qaoa_angles(evaluator, num_restarts=1)


def worker_standard_qaoa(path, reader, p):
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, use_multi_angle=False)
    return path, *optimize_qaoa_angles(evaluator, num_restarts=1)


def select_worker_func(method, reader, p):
    if method == 'general':
        worker_func = partial(worker_general_qaoa, reader=reader, p=p)
    elif method == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif method == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p)
    return worker_func


def optimize_expectation_parallel(paths, method, reader, p, num_workers):
    worker_func = select_worker_func(method, reader, p)
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(worker_func, paths), total=len(paths), smoothing=0, ascii=' █'):
            path_tokens = path.split(result[0])
            output_path = path.join(path_tokens[0], 'output', path_tokens[1])
            result_df = DataFrame({'GQAOA': [result[1]]})
            existing_df = pd.read_csv(output_path, index_col=0) if path.exists(output_path) else DataFrame()
            new_df = pd.concat([existing_df, result_df], ignore_index=True)
            new_df.to_csv(output_path)


def run_graphs_parallel():
    paths = glob.glob(f'graphs/xqaoa/*.csv')
    method = 'general_sub'
    reader = read_graph_xqaoa
    p = 1
    num_workers = 20
    num_iterations = 10

    for i in range(num_iterations):
        print(f'Iteration {i}')
        optimize_expectation_parallel(paths, method, reader, p, num_workers)


if __name__ == '__main__':
    collect_results_xqaoa()
    # extend_csv()
    # run_graphs_parallel()
