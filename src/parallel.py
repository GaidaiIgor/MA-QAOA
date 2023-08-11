import itertools as it
from functools import partial
from multiprocessing import Pool

import networkx as nx
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from src.graph_utils import get_index_edge_list
from src.optimization import Evaluator, optimize_qaoa_angles
from src.angle_strategies import convert_angles_qaoa_to_ma, linear_ramp, convert_angles_tqa_qaoa, interp_qaoa_angles
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


def worker_general_qaoa(path: str, reader: callable, p: int):
    """
    Worker function for Generalized QAOA.
    :param path: Path to graph file for the worker.
    :param reader: Function that reads graph from the file.
    :param p: Number of QAOA layers.
    :return: 1) Path to processed file; 2) Approximation ratio; 3) Corresponding angles.
    """
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
        expectation, angles = optimize_qaoa_angles(evaluator, num_restarts=num_restarts, objective_max=graph.graph['maxcut'])
    else:
        if search_space == 'ma' and guess_format == 'qaoa':
            starting_point = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))
        expectation, angles = optimize_qaoa_angles(evaluator, starting_point=starting_point)

    if search_space == 'linear' or search_space == 'tqa':
        if search_space == 'linear':
            qaoa_angles = linear_ramp(angles, p)
        elif search_space == 'tqa':
            qaoa_angles = convert_angles_tqa_qaoa(angles, p)
        evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space='regular')
        expectation, angles = optimize_qaoa_angles(evaluator, starting_point=qaoa_angles)

    return path, expectation / graph.graph['maxcut'], angles


def worker_maxcut(data: tuple, reader: callable):
    """
    Worker that evaluates maxcut by brute-force and writes it to the input file as graph property.
    :param data: Tuple of input data for the worker. Includes 1) Path to the input file.
    :param reader: Function that reads graph from the file.
    :return: None.
    """
    path = data[0]
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
        worker_func = partial(worker_general_qaoa, reader=reader, p=p)
    elif worker == 'general_sub':
        worker_func = partial(worker_general_qaoa_sub, reader=reader, p=p)
    elif worker == 'standard':
        worker_func = partial(worker_standard_qaoa, reader=reader, p=p, search_space=search_space, guess_format=guess_format, num_restarts=num_restarts)
    elif worker == 'maxcut':
        worker_func = partial(worker_maxcut, reader=reader)
    return worker_func


def numpy_str_to_array(array_string: str) -> ndarray:
    """
    Converts numpy array string representation back to array.
    :param array_string: Numpy array string.
    :return: Numpy array.
    """
    return np.array([float(item) for item in array_string[1:-1].split()])


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
    else:
        starting_angles = [None] * len(paths)

    return list(zip(paths, starting_angles))


def get_angle_col_name(col_name: str) -> str:
    """
    Returns column name with angles corresponding to expectations in a given column.
    :param col_name: Expectation column name.
    :return: Column name with the corresponding angles.
    """
    return f'{col_name}_angles'


def optimize_expectation_parallel(input_df: DataFrame, rows: ndarray, num_workers: int, worker: str, reader: callable, search_space: str, p: int, initial_guess: str,
                                  guess_format: str, angles_col: str, num_restarts: int, copy_col: str | None, copy_better: bool, out_path: str, out_col: str):
    """
    Optimizes cut expectation for a given set of graphs in parallel and writes the output dataframe.
    :param input_df: Input dataframe with information about jobs.
    :param rows: Array that identifies which rows of input_df should be considered.
    :param num_workers: Number of parallel workers.
    :param worker: Worker name.
    :param reader: Function that reads graph from the file.
    :param search_space: Name of angle search space (ma, qaoa, linear, tqa).
    :param p: Number of QAOA layers.
    :param initial_guess: Initial guess strategy (random, explicit or interp).
    :param guess_format: Name of format of starting point (same options as for search space).
    :param angles_col: Name of column in input_df with initial angles for optimization.
    :param num_restarts: Number of restarts for optimization when starting from random angles.
    :param copy_col: Name of the column from where expectation should be copied if it was not calculated in the current round (=None).
    :param copy_better: If true, better expectation values will also be copied from copy_col.
    :param out_path: Path where the output dataframe should be written.
    :param out_col: Name of the column in output dataframe with calculated expectation values.
    :return: None.
    """
    worker_data = prepare_worker_data(input_df, rows, initial_guess, angles_col, p - 1)
    worker_func = select_worker_func(worker, reader, p, search_space, guess_format, num_restarts)
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

    if copy_col is not None:
        copy_rows = np.isnan(out_df[out_col])
        if copy_better:
            copy_rows = copy_rows | (out_df[out_col] < input_df[copy_col])
        comparison_angle_col = get_angle_col_name(copy_col)
        out_df.loc[copy_rows, out_angle_col] = input_df.loc[copy_rows, comparison_angle_col]
        out_df.loc[copy_rows, out_col] = input_df.loc[copy_rows, copy_col]

    print(f'p: {p}; mean: {np.mean(out_df[out_col])}; converged: {sum(out_df[out_col] > 0.9995)}\n')
    out_df.to_csv(out_path)
