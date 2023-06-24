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
from src.parameter_reduction import convert_angles_qaoa_to_ma
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def collect_results_from(base_path: str, columns: list[str], aggregator: callable):
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


def worker_general_qaoa(path: str, reader: callable, p: int):
    graph = reader(path)
    target_vals = evaluate_graph_cut(graph)
    driver_term_vals = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 1)])

    driver_term_vals_2 = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # driver_term_vals_2 = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 2)])
    driver_term_vals = np.append(driver_term_vals, driver_term_vals_2, axis=0)

    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)
    expectation, angles = optimize_qaoa_angles(evaluator, num_restarts=1)
    return path, expectation / graph.graph['maxcut'], angles


def worker_general_qaoa_sub(path: str, reader: callable, p: int):
    graph = reader(path)
    target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)
    return path, *optimize_qaoa_angles(evaluator, num_restarts=1)


def worker_standard_qaoa(data: tuple, reader: callable, p: int, angle_strategy: str):
    graph = reader(data[0])
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, angle_strategy=angle_strategy)
    if data[1] is None:
        expectation, angles = optimize_qaoa_angles(evaluator, num_restarts=1)
    else:
        ma_angles = convert_angles_qaoa_to_ma(data[1], len(graph.edges), len(graph))
        expectation, angles = optimize_qaoa_angles(evaluator, starting_point=ma_angles)
    return path, expectation / graph.graph['maxcut'], angles


def select_worker_func(method: str, reader: callable, p: int, angle_strategy: str):
    if method == 'general':
        worker_func = partial(worker_general_qaoa, reader=reader, p=p)
    elif method == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif method == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p, angle_strategy=angle_strategy)
    return worker_func


def prepare_worker_data(input_df: DataFrame, angles_col: str):
    paths = [f'graphs/nodes_8/{item}.gml' for item in input_df.index]
    if angles_col is None:
        angles_qaoa = [None] * len(paths)
    else:
        angles_qaoa = [[float(angle) for angle in angles_str[1:-1].split()] for angles_str in input_df[angles_col]]
    return list(zip(paths, angles_qaoa))


def optimize_expectation_parallel(input_df: DataFrame, num_workers: int, reader: callable, method: str, p: int, angle_strategy: str, out_path: str, out_col_name: str,
                                  angles_col: str = None):
    worker_data = prepare_worker_data(input_df, angles_col)
    worker_func = select_worker_func(method, reader, p, angle_strategy)
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(worker_func, worker_data), total=len(worker_data), smoothing=0, ascii=' █'):
            key = int(path.split(result[0])[1][:-4])
            results.append([key, *result[1:]])
    out_df = DataFrame(results).set_axis(['key', f'{out_col_name}_p{p}', f'{out_col_name}_p{p}_angles'], axis=1).set_index('key').sort_index()
    out_df = input_df.join(out_df)
    out_df.to_csv(out_path)


def run_graphs_parallel():
    input_df = pd.read_csv('output.csv', index_col=0)
    num_workers = 20
    reader = partial(nx.read_gml, destringizer=int)
    method = 'standard'
    p = 1
    angle_strategy = 'ma'
    out_path = 'output2.csv'
    col_name = 'ma_qaoa'
    angles_col = f'qaoa_rand_p{p}_angles'
    num_iterations = 1

    for i in range(num_iterations):
        print(f'Iteration {i}')
        optimize_expectation_parallel(input_df, num_workers, reader, method, p, angle_strategy, out_path, col_name, angles_col)


if __name__ == '__main__':
    # collect_results_xqaoa()
    # extend_csv()
    run_graphs_parallel()
