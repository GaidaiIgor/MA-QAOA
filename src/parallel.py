import itertools as it
from functools import partial
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from src.data_processing import numpy_str_to_array, get_angle_col_name, copy_expectation_column
from src.graph_utils import get_index_edge_list
from src.optimization import Evaluator, optimize_qaoa_angles
from src.angle_strategies import convert_angles_qaoa_to_ma, linear_ramp, convert_angles_tqa_qaoa, interp_qaoa_angles
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def worker_general_qaoa(data: tuple, reader: callable, p: int, num_restarts: int):
    """
    Worker function for Generalized QAOA.
    :param data: Tuple of input data for the worker. Includes 1) Path to the input file; 2) Starting point for optimization (or None).
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param num_restarts: Number of optimization restarts.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles.
    """
    path, starting_point = data
    graph = reader(path)
    target_vals = evaluate_graph_cut(graph)
    driver_term_vals = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 1)])

    # driver_term_vals_2 = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
    # # driver_term_vals_2 = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 2)])
    # driver_term_vals = np.append(driver_term_vals, driver_term_vals_2, axis=0)

    evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)
    expectation, angles = optimize_qaoa_angles(evaluator, num_restarts=num_restarts)
    return path, expectation / graph.graph['maxcut'], angles


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


def worker_standard_qaoa(data: tuple, reader: callable, p: int, search_space: str, guess_format: str = None, num_restarts: int = 1):
    """
    Worker function for non-generalized QAOA.
    :param data: Tuple of input data for the worker. Includes 1) Path to the input file; 2) Starting point for optimization (or None).
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param search_space: Name of angle search space (ma, qaoa, linear, tqa).
    :param guess_format: Name of format of starting point (same options as for search space).
    :param num_restarts: Number of restarts for optimization when starting from random angles.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles.
    """
    path, starting_point = data
    graph = reader(path)
    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    if starting_point is None:
        result = optimize_qaoa_angles(evaluator, num_restarts=num_restarts, objective_max=graph.graph['maxcut'])

    else:
        if search_space == 'ma' and guess_format == 'qaoa':
            starting_point = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))
        method = 'Nelder-Mead' if starting_point[-1] == 0 else 'BFGS'
        result = optimize_qaoa_angles(evaluator, starting_point=starting_point, method=method)

    if search_space == 'linear' or search_space == 'tqa':
        if search_space == 'linear':
            qaoa_angles = linear_ramp(result.x, p)
        elif search_space == 'tqa':
            qaoa_angles = convert_angles_tqa_qaoa(result.x, p)
        evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space='regular')
        result = optimize_qaoa_angles(evaluator, starting_point=qaoa_angles)

    return path, -result.fun / graph.graph['maxcut'], result.x, result.nfev


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


def select_worker_func(worker: str, reader: callable, p: int, search_space: str, guess_format: str, num_restarts: int = 1):
    """
    Selects worker function based on arguments and binds all arguments except the input path.
    :param worker: Name of worker.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :param search_space: Name of angle search space (ma, qaoa, linear, tqa).
    :param guess_format: Name of format of starting point (same options as for search space).
    :param num_restarts: Number of restarts for optimization when starting from random angles.
    :return: Bound worker function.
    """
    if worker == 'general':
        worker_func = partial(worker_general_qaoa, reader=reader, p=p, num_restarts=num_restarts)
    elif worker == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif worker == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p, search_space=search_space, guess_format=guess_format, num_restarts=num_restarts)
    return worker_func


def prepare_worker_data(input_df: DataFrame, rows: ndarray, initial_guess: str, angles_col: str, p: int) -> list[tuple]:
    """
    Prepares input data for workers.
    :param input_df: Dataframe with input data.
    :param rows: Array that identifies which rows of input_df should be considered.
    :param initial_guess: Initial guess strategy (random, explicit or interp).
    :param angles_col: Name of column in input_df with initial angles for optimization.
    :param p: Value of p for the initial angles.
    :return: List of tuples with worker data.
    """
    if rows is None:
        rows = input_df.index

    df = input_df.loc[rows, :]
    paths = df.index
    if initial_guess == 'explicit' or initial_guess == 'interp':
        starting_angles = [numpy_str_to_array(angles_str) for angles_str in df[angles_col]]
        if initial_guess == 'interp':
            starting_angles = [interp_qaoa_angles(angles, p) for angles in starting_angles]
    elif initial_guess == 'random':
        starting_angles = [None] * len(paths)
    else:
        raise 'Unknown guess'

    return list(zip(paths, starting_angles))


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


def optimize_expectation_parallel(dataframe_path: str, rows_func: callable, num_workers: int, worker: str, reader: callable, search_space: str, p: int, initial_guess: str,
                                  guess_format: str, angles_col: str | None, num_restarts: int, copy_col: str | None, copy_p: int, copy_better: bool, out_col: str):
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
    :param num_restarts: Number of restarts for optimization when starting from random angles.
    :param copy_col: Name of the column from where expectation should be copied if it was not calculated in the current round (=None).
    :param copy_p: Value of p for the copy column. If current p is different, the copied angles will be appended with 0 to keep the angle format consistent.
    :param copy_better: If true, better expectation values will also be copied from copy_col.
    :param out_col: Name of the column in output dataframe with calculated expectation values.
    :return: False if no new results were calculated.
    """
    df = pd.read_csv(dataframe_path, index_col=0)
    rows = rows_func(df)
    cols = ['path', out_col, f'{out_col}_angles', f'{out_col}_nfev']
    worker_data = prepare_worker_data(df, rows, initial_guess, angles_col, p - 1)

    if len(worker_data) == 0:
        results = [(path, *[np.nan] * (len(cols) - 1)) for path in df.index]
    else:
        worker_func = select_worker_func(worker, reader, p, search_space, guess_format, num_restarts)
        results = []
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(worker_func, worker_data), total=len(worker_data), smoothing=0, ascii=' █'):
                if result is not None:
                    results.append(result)

    new_df = DataFrame(results).set_axis(cols, axis=1).set_index('path').sort_index()
    df.update(new_df)
    df = df.join(new_df[new_df.columns.difference(df.columns)])
    df = copy_expectation_column(df, copy_better, copy_col, out_col, copy_p, p)

    print(f'p: {p}; mean: {np.mean(df[out_col])}; converged: {sum(df[out_col] > 0.9995)}; nfev: {np.mean(df[f"{out_col}_nfev"])}\n')
    df.to_csv(dataframe_path)
