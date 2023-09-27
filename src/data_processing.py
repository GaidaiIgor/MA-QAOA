import glob
import re
from math import copysign
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from src.graph_utils import get_edge_diameter


def collect_results_from(base_path: str, columns: list[str], aggregator: callable) -> DataFrame:
    """
    Collects results from multiple dataframes.
    :param base_path: Path to folder with dataframes.
    :param columns: List of columns that should be taken from each dataframe.
    :param aggregator: Aggregator function that will be applied to each column of the input dataframes.
    :return: Aggregated dataframe, where each input dataframe is reduced to a row in the output dataframe.
    """
    paths = glob.glob(f'{base_path}/*.csv')
    stat = []
    for path in paths:
        df = pd.read_csv(path)
        stat.append(aggregator(df[columns], axis=0))
    index_keys = [Path(path).parts[-1] for path in paths]
    summary_df = DataFrame(stat, columns=columns, index=index_keys)
    return summary_df


def extract_numbers(str_arr: list[str]) -> list[int]:
    """ Extracts numbers after _ from column names. """
    return [int(name.split('_')[1]) for name in str_arr]


def get_column_average(df_path: str, col_regex: str) -> tuple[list, list]:
    """
    Returns average value of columns whose name matches specified regular expression and column header values.
    :param df_path: Path to dataframe.
    :param col_regex: Regular expression for column names.
    :return: 1) List of column header values. 2) List of corresponding column averages.
    """
    df = pd.read_csv(df_path, index_col=0)
    df = df.filter(regex=col_regex)
    header_values = extract_numbers(df.columns)
    averages = np.mean(df, axis=0)
    return header_values, averages


def calculate_edge_diameter(df: DataFrame):
    """
    Calculates edge diameters for all graphs specified in input dataframe and adds the result to the same dataframe.
    :param df: Input dataframe.
    :return: Modified dataframe with added values of edge diameter.
    """
    edge_diameters = [0] * df.shape[0]
    for i in range(len(edge_diameters)):
        path = df.index[i]
        graph = nx.read_gml(path, destringizer=int)
        edge_diameters[i] = get_edge_diameter(graph)

    df['edge_diameter'] = edge_diameters
    return df


def calculate_edge_probabilities(paths: list[str]) -> list[float]:
    """
    Calculates edge probability for all graphs specified in input dataframe.
    :param paths: List of paths to graphs.
    :return: List of edge probabilities.
    """
    edge_probabilities = [0] * len(paths)
    for i in range(len(edge_probabilities)):
        graph = nx.read_gml(paths[i], destringizer=int)
        max_edges = len(graph) * (len(graph) - 1) // 2
        edge_probabilities[i] = len(graph.edges) / max_edges
    return edge_probabilities


def calculate_property(paths: list[str], property_func: callable) -> list[float]:
    """
    Reads specified graphs and calculates specified properties.
    :param paths: List of paths to graphs.
    :param property_func: Function that accepts graph as argument and returns its property.
    :return: List of edge probabilities.
    """
    properties = [0] * len(paths)
    for i in range(len(properties)):
        graph = nx.read_gml(paths[i], destringizer=int)
        properties[i] = property_func(graph)
    return properties


def calculate_min_p(df: DataFrame, target_convergence: float):
    """
    Finds minimum value of p necessary to achieve maxcut and appends it to the given dataframe.
    :param df: Input dataframe with graphs and calculation results for multiple p values.
    :param target_convergence: Target approximation ratio at which a graph is considered fully converged.
    :return: Modified dataframe with added minimum values of p.
    """
    min_p = [0] * df.shape[0]
    cols = [col for col in df.columns if col[:2] == 'p_' and col[-1].isdigit()]
    p_vals = extract_numbers(cols)
    for i in range(df.shape[0]):
        row = df.iloc[i, :][cols]
        p_index = np.where(row > target_convergence)[0]
        min_p[i] = p_vals[p_index[0]] if len(p_index) > 0 else np.inf
    df['min_p'] = min_p
    return df


def calculate_extra(df_path: str):
    """
    Calculates edge diameters and minimum values of p for specified dataframe.
    :param df_path: Path to a dataframe with calculations.
    :return: None.
    """
    df = pd.read_csv(df_path, index_col=0)
    df = calculate_edge_diameter(df)
    df = calculate_min_p(df, 0.9995)
    df.to_csv(df_path)


def numpy_str_to_array(array_string: str) -> ndarray:
    """
    Converts numpy array string representation back to array.
    :param array_string: Numpy array string.
    :return: Numpy array.
    """
    return np.array([float(item) for item in array_string[1:-1].split()])


def get_angle_col_name(col_name: str) -> str:
    """
    Returns column name with angles corresponding to expectations in a given column.
    :param col_name: Expectation column name.
    :return: Column name with the corresponding angles.
    """
    return f'{col_name}_angles'


def copy_expectation_column(df: DataFrame, copy_better: bool, copy_col: str, out_col: str, copy_p: int, out_p: int) -> DataFrame:
    """
    Copies expectations and angles from the specified column if they are not present (or smaller) in the given column.
    :param df: Dataframe with QAOA expectation results.
    :param copy_better: True if larger expectations are to be copied. False to copy only when a value is missing from the out_col (None).
    :param copy_col: Column from where expectations should be copied.
    :param out_col: Column to where expectations should be copied.
    :param copy_p: The value of p for the copy_col.
    :param out_p: The value of p for the out_col.
    :return: Modified dataframe with copied expectations.
    """
    if copy_col is not None:
        copy_angles_col = get_angle_col_name(copy_col)
        out_angles_col = get_angle_col_name(out_col)

        copy_rows = np.isnan(df[out_col])
        if copy_better:
            copy_rows = copy_rows | (df[out_col] < df[copy_col])

        if not any(copy_rows):
            return df

        copy_angles = df.loc[copy_rows, copy_angles_col].apply(lambda x: numpy_str_to_array(x))
        if copy_p != out_p:
            angles_per_layer = len(copy_angles[0]) // copy_p
            copy_angles = [str(np.concatenate((angles, [0] * angles_per_layer))) for angles in copy_angles]
        df.loc[copy_rows, out_angles_col] = copy_angles
        df.loc[copy_rows, out_col] = df.loc[copy_rows, copy_col]
    return df


def copy_expectation_dataframe(df: DataFrame) -> DataFrame:
    """
    Applies copy_expectation_column for all pairs of adjacent values of p.
    :param df: Dataframe with QAOA calculations.
    :return: Modified dataframe.
    """
    copy_better = True
    exp_col_names = [col for col in df.columns if re.match(r'p_\d+$', col)]
    p_vals = extract_numbers(exp_col_names)
    max_p = max(p_vals)
    for p in range(2, max_p + 1):
        df = copy_expectation_column(df, copy_better, f'p_{p - 1}', f'p_{p}', p - 1, p)
    return df


def merge_dfs(base_path: str, ps: list[int], restarts: int, convergence_threshold: float, out_name: str, copy_better: bool):
    """
    Merges dataframes corresponding to specified values of p with specified number of restarts and writes a new merged dataframe.
    :param base_path: Path to the folder with individual p calculations.
    :param ps: List of p-values to merge
    :param restarts: How many restarts should be taken from each p.
    If the specified number of restarts is not available, but lower number of restarts is completely converged, then this number of restarts will be taken as equivalent.
    :param convergence_threshold: Threshold AR for complete convergence.
    :param out_name: Name of the merged dataframe.
    :param copy_better: True to copy expectations from smaller p if better answer was found.
    :return: None.
    """
    merged_df = pd.DataFrame()
    for p_ind, p in enumerate(ps):
        next_df = pd.read_csv(f'{base_path}/p_{p}/out.csv', index_col=0)
        exp_col_name = f'r_{restarts}'
        calculated_restarts = extract_numbers(next_df.filter(regex=r'r_\d+$').columns)
        if restarts > calculated_restarts[-1] and sum(next_df[f'r_{calculated_restarts[-1]}'] > convergence_threshold) == next_df.shape[0]:
            exp_col_name = f'r_{calculated_restarts[-1]}'

        angle_col_name = get_angle_col_name(exp_col_name)
        new_exp_col_name = f'p_{p}'
        new_angle_col_name = get_angle_col_name(new_exp_col_name)
        next_df = next_df[[exp_col_name, angle_col_name]].set_axis([new_exp_col_name, new_angle_col_name], axis=1)
        if merged_df.empty:
            merged_df = next_df
        else:
            merged_df = merged_df.join(next_df)
    if copy_better:
        merged_df = copy_expectation_dataframe(merged_df)
    merged_df.to_csv(out_name)


def normalize_angles(angles: ndarray) -> ndarray:
    """
    Adds +-pi to angles to move them into +-pi/2 range.
    :param angles: QAOA angles array given in fractions of pi.
    :return: Normalized angles array.
    """
    normalized = angles.copy()
    for i in range(len(normalized)):
        normalized[i] -= int(normalized[i])
        if normalized[i] > 0.5 or normalized[i] <= -0.5:
            normalized[i] -= copysign(1, normalized[i])
    return normalized


def round_angles(angles: ndarray) -> ndarray:
    """
    Rounds angles to the nearest multiples of pi/4.
    :param angles: Input angles given in fractions of pi.
    :return: Rounded.
    """
    return np.round(angles * 4) / 4
