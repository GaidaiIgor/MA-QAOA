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
from src.parameter_reduction import convert_angles_qaoa_to_ma, linear_ramp
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


def worker_standard_qaoa(data: tuple, reader: callable, p: int, angle_strategy: str, num_restarts: int = 1):
    path, starting_point = data
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, angle_strategy=angle_strategy)
    if starting_point is None:
        expectation, angles = optimize_qaoa_angles(evaluator, num_restarts=num_restarts, objective_max=graph.graph['maxcut'])
        if angle_strategy == 'linear':
            evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, angle_strategy='regular')
            qaoa_angles = linear_ramp(angles, p)
            expectation, angles = optimize_qaoa_angles(evaluator, starting_point=qaoa_angles)
    else:
        ma_angles = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))
        expectation, angles = optimize_qaoa_angles(evaluator, starting_point=ma_angles)
    return path, expectation / graph.graph['maxcut'], angles


def worker_relaxation(data: tuple, reader: callable, p: int, angle_strategy: str):
    path, starting_point = data
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, angle_strategy=angle_strategy)
    if angle_strategy == 'regular':
        starting_angles = linear_ramp(starting_point, p)
    elif angle_strategy == 'ma':
        starting_angles = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))

    expectation, angles = optimize_qaoa_angles(evaluator, starting_point=starting_angles)
    return path, expectation / graph.graph['maxcut'], angles


def select_worker_func(worker: str, reader: callable, p: int, angle_strategy: str, num_restarts: int = 1):
    if worker == 'general':
        worker_func = partial(worker_general_qaoa, reader=reader, p=p)
    elif worker == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif worker == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p, angle_strategy=angle_strategy, num_restarts=num_restarts)
    elif worker == 'relax':
        worker_func = partial(worker_relaxation, reader=reader, p=p, angle_strategy=angle_strategy)
    return worker_func


def prepare_worker_data(input_df: DataFrame, angles_col: str, skip_col: str):
    df = input_df.loc[input_df[skip_col] < 0.99, :]
    paths = df.index
    if angles_col is None:
        starting_angles = [None] * len(paths)
    else:
        starting_angles = [[float(angle) for angle in angles_str[1:-1].split()] for angles_str in df[angles_col]]
    return list(zip(paths, starting_angles))


def get_angle_col_name(col_name: str) -> str:
    return f'{col_name}_angles'


def optimize_expectation_parallel(input_df: DataFrame, num_workers: int, reader: callable, worker: str, p: int, angle_strategy: str, num_restarts: int, out_path: str, out_col: str,
                                  angles_col: str = None, skip_col: str = None, comparison_col: str = None):
    worker_data = prepare_worker_data(input_df, angles_col, skip_col)
    worker_func = select_worker_func(worker, reader, p, angle_strategy, num_restarts)
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(worker_func, worker_data), total=len(worker_data), smoothing=0, ascii=' █'):
            results.append(result)

    out_angle_col = get_angle_col_name(out_col)
    new_df = DataFrame(results).set_axis(['path', out_col, out_angle_col], axis=1).set_index('path').sort_index()
    out_df = pd.read_csv(out_path, index_col=0)
    out_df.update(new_df)
    out_df = out_df.join(new_df[new_df.columns.difference(out_df.columns)])

    if comparison_col is not None:
        comparison_angle_col = get_angle_col_name(comparison_col)
        for row_ind in out_df.index:
            if np.isnan(out_df.loc[row_ind, out_col]) or out_df.loc[row_ind, out_col] < out_df.loc[row_ind, comparison_col]:
                out_df.loc[row_ind, out_col] = out_df.loc[row_ind, comparison_col]
                out_df.loc[row_ind, out_angle_col] = out_df.loc[row_ind, comparison_angle_col]

    out_df.to_csv(out_path)


def run_graphs_parallel():
    p = 2
    input_path = f'graphs/nodes_8/output/qaoa/random/out.csv'
    num_workers = 20
    worker = 'standard'
    angle_strategy = 'regular'
    num_restarts = 5
    out_path = f'graphs/nodes_8/output/qaoa/random/out.csv'
    col_name = f'p_{p}'
    angles_col = None
    skip_col = f'p_{p - 1}'
    comparison_col = f'p_{p - 1}'

    reader = partial(nx.read_gml, destringizer=int)
    input_df = pd.read_csv(input_path, index_col=0)
    optimize_expectation_parallel(input_df, num_workers, reader, worker, p, angle_strategy, num_restarts, out_path, col_name, angles_col, skip_col, comparison_col)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # collect_results_xqaoa()
    # extend_csv()
    run_graphs_parallel()
