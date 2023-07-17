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
from numpy import ndarray

from src.optimization import Evaluator, optimize_qaoa_angles
from src.graph_utils import get_edge_diameter, get_index_edge_list, read_graph_xqaoa
from src.parameter_reduction import convert_angles_qaoa_to_ma, linear_ramp, convert_angles_tqa_qaoa, interp_qaoa_angles
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


def calculate_edge_diameter(df: DataFrame):
    edge_diameters = [0] * df.shape[0]
    for i in range(len(edge_diameters)):
        path = df.index[i]
        graph = nx.read_gml(path, destringizer=int)
        edge_diameters[i] = get_edge_diameter(graph)

    df['edge_diameter'] = edge_diameters
    return df


def calculate_min_p(df: DataFrame):
    min_p = [0] * df.shape[0]
    cols = [col for col in df.columns if col[:2] == 'p_' and col[-1].isdigit()]
    p_vals = [int(col.split('_')[1]) for col in cols]
    for i in range(len(min_p)):
        row = df.iloc[i, :][cols]
        min_p[i] = p_vals[np.where(row > 0.9995)[0][0]]
    df['min_p'] = min_p
    df['p_rel_ed'] = df['min_p'] - df['edge_diameter']
    return df


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


def worker_standard_qaoa(data: tuple, reader: callable, p: int, search_space: str, num_restarts: int = 1):
    path, starting_point = data
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    if starting_point is None:
        expectation, angles = optimize_qaoa_angles(evaluator, num_restarts=num_restarts, objective_max=graph.graph['maxcut'])
    else:
        expectation, angles = optimize_qaoa_angles(evaluator, starting_point=starting_point)
        # angles = starting_point

    if search_space == 'linear' or search_space == 'tqa':
        if search_space == 'linear':
            qaoa_angles = linear_ramp(angles, p)
        elif search_space == 'tqa':
            qaoa_angles = convert_angles_tqa_qaoa(angles, p)
        evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space='regular')
        expectation, angles = optimize_qaoa_angles(evaluator, starting_point=qaoa_angles)

    return path, expectation / graph.graph['maxcut'], angles


def worker_relaxation(data: tuple, reader: callable, p: int, search_space: str):
    path, starting_point = data
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    if search_space == 'regular':
        starting_angles = linear_ramp(starting_point, p)
    elif search_space == 'ma':
        starting_angles = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))

    expectation, angles = optimize_qaoa_angles(evaluator, starting_point=starting_angles)
    return path, expectation / graph.graph['maxcut'], angles


def worker_maxcut(data: tuple, reader: callable):
    path = data[0]
    graph = reader(path)
    cut_vals = evaluate_graph_cut(graph)
    max_cut = int(max(cut_vals))
    graph.graph['maxcut'] = max_cut
    nx.write_gml(graph, path)


def select_worker_func(worker: str, reader: callable, search_space: str, p: int, num_restarts: int = 1):
    if worker == 'general':
        worker_func = partial(worker_general_qaoa, reader=reader, p=p)
    elif worker == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif worker == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p, search_space=search_space, num_restarts=num_restarts)
    elif worker == 'relax':
        worker_func = partial(worker_relaxation, reader=reader, p=p, search_space=search_space)
    elif worker == 'maxcut':
        worker_func = partial(worker_maxcut, reader=reader)
    return worker_func


def numpy_str_to_array(array_string: str) -> ndarray:
    return np.array([float(item) for item in array_string[1:-1].split()])


def prepare_worker_data(input_df: DataFrame, initial_guess: str, angles_col: str = None, rows: ndarray = None):
    if rows is None:
        rows = input_df.index

    df = input_df.loc[rows, :]
    paths = df.index
    if initial_guess == 'explicit' or initial_guess == 'interp':
        starting_angles = [numpy_str_to_array(angles_str) for angles_str in df[angles_col]]
        if initial_guess == 'interp':
            starting_angles = [interp_qaoa_angles(angles) for angles in starting_angles]
    else:
        starting_angles = [None] * len(paths)

    return list(zip(paths, starting_angles))


def get_angle_col_name(col_name: str) -> str:
    return f'{col_name}_angles'


def optimize_expectation_parallel(input_df: DataFrame, num_workers: int, worker: str, reader: callable, search_space: str, p: int, num_restarts: int, out_path: str, out_col: str,
                                  initial_guess: str = None, angles_col: str = None, rows: ndarray = None, comparison_col: str = None):
    worker_data = prepare_worker_data(input_df, initial_guess, angles_col, rows)
    worker_func = select_worker_func(worker, reader, search_space, p, num_restarts)
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(worker_func, worker_data), total=len(worker_data), smoothing=0, ascii=' █'):
            if result is not None:
                results.append(result)

    if len(results) == 0:
        return

    out_angle_col = get_angle_col_name(out_col)
    new_df = DataFrame(results).set_axis(['path', out_col, out_angle_col], axis=1).set_index('path').sort_index()
    out_df = input_df.copy()
    out_df.update(new_df)
    out_df = out_df.join(new_df[new_df.columns.difference(out_df.columns)])

    if comparison_col is not None:
        comparison_angle_col = get_angle_col_name(comparison_col)
        rows = np.isnan(out_df[out_col]) | (out_df[out_col] < input_df[comparison_col])
        out_df.loc[rows, out_angle_col] = input_df.loc[rows, comparison_angle_col]
        out_df.loc[rows, out_col] = input_df.loc[rows, comparison_col]

    print(f'p: {p}; mean: {np.mean(out_df[out_col])}; converged: {sum(out_df[out_col] > 0.9995)}\n')
    out_df.to_csv(out_path)


def run_graphs_parallel():
    input_path = 'graphs/nodes_8/ed_4/output/qaoa/tqa/out.csv'
    num_workers = 20
    worker = 'standard'
    search_space = 'tqa'
    initial_guess = None
    num_restarts = 1
    out_path = input_path
    reader = partial(nx.read_gml, destringizer=int)

    for p in range(2, 11):
        input_df = pd.read_csv(input_path, index_col=0)
        col_name = f'p_{p}'
        starting_angles_col = f'p_{p - 1}_angles'
        # rows = (input_df['p_rel_ed'] == 1) & (input_df['edge_diameter'] == 3)
        rows = None if p == 1 else input_df[f'p_{p - 1}'] < 0.9995
        comparison_col = None if p == 1 else f'p_{p - 1}'
        optimize_expectation_parallel(input_df, num_workers, worker, reader, search_space, p, num_restarts, out_path, col_name, initial_guess, starting_angles_col, rows,
                                      comparison_col)

    # p = 10
    # for r in range(1, 11):
    #     col_name = f'ar_{r}'
    #     skip_col = None if r == 1 else f'ar_{r - 1}'
    #     comparison_col = None if r == 1 else f'ar_{r - 1}'


def generate_graphs():
    out_path = 'graphs/nodes_8/ed_5.5'
    num_graphs = 10000
    nodes = 8
    edge_prob = 0.05
    graphs = []
    diameters = []
    for i in range(num_graphs):
        connected = False
        while not connected:
            next_graph = nx.gnp_random_graph(nodes, edge_prob)
            connected = nx.is_connected(next_graph)
        graphs.append(next_graph)
        diameters.append(get_edge_diameter(next_graph))
    print(np.mean(diameters))

    # if abs(np.mean(diameters) - 5) < 5e-3:
    #     for i in range(len(graphs)):
    #         nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # collect_results_xqaoa()
    # extend_csv()
    run_graphs_parallel()
    #
    # for i in range(1):
    #     generate_graphs()
