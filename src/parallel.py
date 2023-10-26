import itertools as it
import re
from functools import partial
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from src.data_processing import numpy_str_to_array, copy_expectation_column
from src.graph_utils import get_index_edge_list
from src.optimization import Evaluator, optimize_qaoa_angles
from src.angle_strategies import convert_angles_qaoa_to_ma, convert_angles_linear_qaoa, convert_angles_tqa_qaoa, interp_qaoa_angles
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def worker_general_qaoa(data: tuple, reader: callable, p: int):
    """
    Worker function for Generalized QAOA.
    :param data: Tuple of input data for the worker. Includes 1) Path to the input file; 2) Starting point for optimization (or None).
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles.
    """
    path, starting_point = data
    graph = reader(path)

    target_vals = evaluate_graph_cut(graph)
    driver_term_vals = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 1)])

    driver_term_vals_2 = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # driver_term_vals_2 = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 2)])
    driver_term_vals = np.append(driver_term_vals, driver_term_vals_2, axis=0)

    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)
    result = optimize_qaoa_angles(evaluator)
    return path, -result.fun / graph.graph['maxcut'], result.x, result.nfev


def worker_general_qaoa_sub(path: str, reader: callable, p: int):
    """
    Worker function for Generalized QAOA working through subsets.
    :param path: Path to graph file for the worker.
    :param reader: Function that reads graph from the file.
    :param p:  Number of QAOA layers.
    :return: 1) Path to processed file; 2) Cut value 3) Corresponding angles.
    """
    graph = reader(path)
    target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)
    return path, *optimize_qaoa_angles(evaluator, num_restarts=1)


def worker_standard_qaoa(data: tuple, reader: callable, p: int, search_space: str) -> tuple:
    """
    Worker function for non-generalized QAOA.
    :param data: Tuple of input data for the worker. Includes 1) Path to the input file; 2) Starting point for optimization (or None for random) in the format of search space.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param search_space: Name of angle search space (ma, qaoa, linear, tqa).
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles; 4) Number of function evaluations.
    """
    path, starting_point = data
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    method = 'COBYLA' if starting_point is not None and any(starting_point == 0) else 'BFGS'
    try:
        result = optimize_qaoa_angles(evaluator, starting_point=starting_point, method=method, options={'maxiter': np.iinfo(np.int32).max})
    except str:
        raise f'Optimization failed at {path}'
    nfev = result.nfev
    return path, -result.fun / graph.graph['maxcut'], result.x, nfev


def worker_interp(data: tuple, reader: callable, p: int) -> tuple:
    """
    Worker that implements interp strategy for QAOA (interpolates angles from p - 1 for a guess at p).
    :param data: Tuple of 1) Path to the graph file; 2) Best angles found at level p - 1.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles; 4) Number of function evaluations.
    """
    path, prev_angles = data
    starting_angles = interp_qaoa_angles(prev_angles, p - 1)
    return worker_standard_qaoa((path, starting_angles), reader, p, 'qaoa')


def worker_linear(data: tuple, reader: callable, p: int, search_space: str) -> tuple:
    """
    Worker that implements linear strategies for QAOA (4 or 1 parameters search followed by relaxation).
    :param data: Tuple of 1) Path to the graph file.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param search_space: Name of angle search space (linear or tqa).
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles; 4) Number of function evaluations.
    """
    path, = data
    _, _, linear_angles, total_nfev = worker_standard_qaoa((path, None), reader, p, search_space)

    if search_space == 'tqa':
        qaoa_angles = convert_angles_tqa_qaoa(linear_angles, p)
    elif search_space == 'linear':
        qaoa_angles = convert_angles_linear_qaoa(linear_angles, p)
    else:
        raise 'Unknown search space'

    _, ar, angles, nfev = worker_standard_qaoa((path, qaoa_angles), reader, p, 'qaoa')
    total_nfev += nfev
    return path, ar, angles, total_nfev


def worker_greedy(data: tuple, reader: callable, p: int) -> tuple:
    """
    Worker that implements greedy strategy for QAOA (p + 1 optimizations from transition states).
    :param data: Tuple of 1) Path to the graph file; 2) Best angles found at level p - 1.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles; 4) Number of function evaluations.
    """
    path, prev_angles = data
    best_ar = 0
    best_angles = [0] * p
    total_nfev = 0
    for insert_layer in range(p):
        starting_angles = np.concatenate((prev_angles[:2 * insert_layer], [0] * 2, prev_angles[2 * insert_layer:]))
        path, next_ar, next_angles, nfev = worker_standard_qaoa((path, starting_angles), reader, p, 'qaoa')
        total_nfev += nfev
        if best_ar < next_ar:
            best_ar = next_ar
            best_angles = next_angles
    return path, best_ar, best_angles, total_nfev


def worker_combined_qaoa(data: tuple, reader: callable, p: int) -> tuple:
    """
    Worker that tries multiple angle strategies for QAOA.
    :param data: Tuple of 1) Path to the graph file; 2) Best AR found at level p - 1; 3) Corresponding angles
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles.
    """
    path, prev_angles = data
    num_strategies = 4
    ars = [0] * num_strategies
    angles = [0] * num_strategies
    nfevs = [0] * num_strategies

    _, ars[0], angles[0], nfevs[0] = worker_interp((path, prev_angles), reader, p)
    _, ars[1], angles[1], nfevs[1] = worker_linear((path,), reader, p, 'tqa')
    _, ars[2], angles[2], nfevs[2] = worker_standard_qaoa((path, None), reader, p, 'qaoa')
    _, ars[3], angles[3], nfevs[3] = worker_greedy((path, prev_angles), reader, p)

    best_index = np.argmax(ars)
    return path, ars[best_index], angles[best_index], sum(nfevs)


def worker_maqaoa(data: tuple, reader: callable, p: int, guess_format: str) -> tuple:
    """
    Worker that executes MA-QAOA optimization (independent angles on each term).
    :param data: Tuple of 1) Path to the graph file; 2) Starting angles, or None for random.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param guess_format: Name of format of starting point (ma or qaoa).
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles; 4) Number of function evaluations.
    """
    path, starting_point = data
    graph = reader(path)
    if guess_format == 'qaoa':
        if starting_point is None:
            starting_point = np.random.uniform(-np.pi, np.pi, 2 * p)
        starting_point = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))
    return worker_standard_qaoa((path, starting_point), reader, p, 'ma')


def worker_maxcut(path: str, reader: callable):
    """
    Worker that evaluates maxcut by brute-force and writes it to the input file as graph property.
    :param path: Path to the input file.
    :param reader: Function that reads graph from the file.
    :return: None.
    """
    graph = reader(path)
    cut_vals = evaluate_graph_cut(graph)
    max_cut = int(max(cut_vals))
    graph.graph['maxcut'] = max_cut
    nx.write_gml(graph, path)


def select_worker_func(worker: str, reader: callable, p: int, search_space: str, guess_format: str):
    """
    Selects worker function based on arguments and binds all arguments except the input path.
    :param worker: Name of worker.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param search_space: Name of angle search space (xqaoa, ma, qaoa, linear, tqa).
    :param guess_format: Name of format of starting point (same options as for search space).
    :return: Bound worker function.
    """
    if worker == 'general':
        worker_func = partial(worker_general_qaoa, reader=reader, p=p)
    elif worker == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif worker == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p, search_space=search_space)
    elif worker == 'interp':
        worker_func = partial(worker_interp, reader=reader, p=p)
    elif worker == 'linear':
        worker_func = partial(worker_linear, reader=reader, p=p, search_space=search_space)
    elif worker == 'greedy':
        worker_func = partial(worker_greedy, reader=reader, p=p)
    elif worker == 'combined':
        worker_func = partial(worker_combined_qaoa, reader=reader, p=p)
    elif worker == 'ma':
        worker_func = partial(worker_maqaoa, reader=reader, p=p, guess_format=guess_format)
    else:
        raise 'Unknown worker'
    return worker_func


def prepare_worker_data(input_df: DataFrame, rows: ndarray, initial_guess: str, angles_col: str) -> list[tuple]:
    """
    Prepares input data for workers.
    :param input_df: Dataframe with input data.
    :param rows: Array that identifies which rows of input_df should be considered.
    :param initial_guess: Initial guess strategy (random, explicit or interp).
    :param angles_col: Name of column in input_df with initial angles for optimization.
    :return: List of tuples with worker data.
    """
    if rows is None:
        rows = input_df.index

    df = input_df.loc[rows, :]
    paths = df.index
    if initial_guess == 'explicit':
        starting_angles = [numpy_str_to_array(angles_str) for angles_str in df[angles_col]]
    elif initial_guess == 'random':
        starting_angles = [None] * len(paths)
    else:
        raise 'Unknown guess'

    return list(zip(paths, starting_angles))


def optimize_expectation_parallel(dataframe_path: str, rows_func: callable, num_workers: int, worker: str, reader: callable, search_space: str, p: int, initial_guess: str,
                                  guess_format: str, angles_col: str | None, copy_col: str | None, copy_p: int, copy_better: bool, out_col: str):
    """
    Optimizes cut expectation for a given set of graphs in parallel and writes the output dataframe.
    :param dataframe_path: Path to input dataframe with information about jobs.
    :param rows_func: Function that accepts dataframe and returns boolean array identifying which rows of the dataframe should be considered.
    :param num_workers: Number of parallel workers.
    :param worker: Worker name.
    :param reader: Function that reads graph from the file.
    :param search_space: Name of angle search space (ma, qaoa, linear, tqa).
    :param p: Number of QAOA layers.
    :param initial_guess: Initial guess strategy (random, explicit or interp).
    :param guess_format: Name of format of starting point (same options as for search space).
    :param angles_col: Name of column in df with initial angles for optimization.
    :param copy_col: Name of the column from where expectation should be copied if it was not calculated in the current round (=None).
    :param copy_p: Value of p for the copy column. If current p is different, the copied angles will be appended with 0 to keep the angle format consistent.
    :param copy_better: If true, better expectation values will also be copied from copy_col.
    :param out_col: Name of the column in output dataframe with calculated expectation values.
    :return: False if no new results were calculated.
    """
    df = pd.read_csv(dataframe_path, index_col=0)
    rows = rows_func(df)
    cols = ['path', out_col, f'{out_col}_angles', f'{out_col}_nfev']
    worker_data = prepare_worker_data(df, rows, initial_guess, angles_col)

    if len(worker_data) == 0:
        if out_col in df:
            return
        results = [(path, *[np.nan] * (len(cols) - 1)) for path in df.index]
    else:
        worker_func = select_worker_func(worker, reader, p, search_space, guess_format)
        results = []
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(worker_func, worker_data), total=len(worker_data), smoothing=0, ascii=' █'):
                if result is not None:
                    results.append(result)

    new_df = DataFrame(results).set_axis(cols, axis=1).set_index('path').sort_index()
    df.update(new_df)
    df = df.join(new_df[new_df.columns.difference(df.columns)])
    df = copy_expectation_column(df, copy_better, copy_col, out_col, copy_p, p)

    dataset_id = re.search(r'nodes_\d+/depth_\d+', dataframe_path)[0]
    print(f'dataset: {dataset_id}; p: {p}; mean: {np.mean(df[out_col]):.3f}; min: {min(df[out_col]):.3f}; converged: {sum(df[out_col] > 0.9995)}; '
          f'nfev: {np.mean(df[f"{out_col}_nfev"]):.0f}\n')
    df.to_csv(dataframe_path)


def calculate_maxcut_parallel(paths: list[str], num_workers: int, reader: callable):
    """
    Calculates maxcut for all graphs specified in paths are writes it to the graph file.
    :param paths: List of paths to graphs to calculate maxcut.
    :param num_workers: Number of parallel workers.
    :param reader: Function that reads graph from the file.
    :return: None
    """
    worker_func = partial(worker_maxcut, reader=reader)
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap(worker_func, paths), total=len(paths), smoothing=0, ascii=' █'):
            pass
