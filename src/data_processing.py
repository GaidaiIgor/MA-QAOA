"""
Functions related to data processing.
"""

import re
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from src.graph_utils import get_edge_diameter


def extract_numbers(str_arr: list[str]) -> list[int]:
    """
    Extracts numbers after _ from given string.
    :param str_arr: List of strings to process.
    :return: List of numbers after _ in the strings.
    """
    return [int(name.split('_')[1]) for name in str_arr]


def exponential_form(ps, c1, c2):
    """
    Fitting funtion for AR vs p data.
    :param ps: Values of p.
    :param c1: Fitting coefficient.
    :param c2: Fitting coefficient.
    :return: Fitted value.
    """
    return 1 - c1 * np.exp(-c2 * ps)


def linear_function(ps, c1, c2):
    return c1 + c2 * ps


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


def transfer_expectation_columns(df: DataFrame, transfer_from: str, transfer_to: str, angle_suffixes: Sequence[str], p_from: int, p_to: int, transfer_better: bool = True) -> \
        DataFrame:
    """
    Transfers expectations and angles between the columns if they are not present (or smaller) in copy_to.
    :param df: Dataframe with QAOA expectation results.
    :param transfer_from: Column from where expectations should be copied.
    :param transfer_to: Column to where expectations should be copied.
    :param angle_suffixes: Suffixes that define the names of the columns that store the angles associated with the primary expectation columns.
    :param p_from: The value of p for the transfer_from column.
    :param p_to: The value of p for the transfer_to column.
    :param transfer_better: True if larger expectations are to be copied. False to copy only when a value is missing from the copy_to (nan).
    :return: Modified dataframe with copied expectations.
    """
    transfer_rows = np.isnan(df[transfer_to])
    if transfer_better:
        transfer_rows = transfer_rows | (df[transfer_to] < df[transfer_from])

    if not any(transfer_rows):
        return df

    for suffix in angle_suffixes:
        transfer_angles_from = transfer_from + suffix
        transfer_angles_to = transfer_to + suffix
        if p_to - p_from == 0:
            df.loc[transfer_rows, transfer_angles_to] = df.loc[transfer_rows, transfer_angles_from]
        else:
            original_angles = df.loc[transfer_rows, transfer_angles_from].apply(lambda x: numpy_str_to_array(x))
            transformed_angles = [str(np.concatenate((angles, [0] * (len(angles) // p_from) * (p_to - p_from)))) for angles in original_angles]
            df.loc[transfer_rows, transfer_angles_to] = transformed_angles

    df.loc[transfer_rows, transfer_to] = df.loc[transfer_rows, transfer_from]
    return df


def transfer_expectation_dataframe(df: DataFrame) -> DataFrame:
    """
    Transfers expectations between all pairs of adjacent values of p assuming the standard angle suffixes.
    :param df: Dataframe with QAOA calculations.
    :return: Modified dataframe.
    """
    transfer_better = True
    exp_col_names = [col for col in df.columns if re.match(r'p_\d+$', col)]
    p_vals = extract_numbers(exp_col_names)
    max_p = max(p_vals)
    for p in range(2, max_p + 1):
        df = transfer_expectation_columns(df, f'p_{p - 1}', f'p_{p}', ['_angles'], p - 1, p, transfer_better)
    return df


def merge_dfs(base_path: str, ps: Sequence[int], restarts: Sequence[int], convergence_threshold: float, out_name: str, copy_better: bool):
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
        next_restarts = restarts[p_ind]
        exp_col_name = f'r_{next_restarts}'
        calculated_restarts = extract_numbers(next_df.filter(regex=r'r_\d+$').columns)[-1]
        if next_restarts > calculated_restarts and sum(next_df[f'r_{calculated_restarts}'] > convergence_threshold) == next_df.shape[0]:
            exp_col_name = f'r_{calculated_restarts}'

        angle_col_name = get_angle_col_name(exp_col_name)
        new_exp_col_name = f'p_{p}'
        new_angle_col_name = get_angle_col_name(new_exp_col_name)
        next_df = next_df[[exp_col_name, angle_col_name]].set_axis([new_exp_col_name, new_angle_col_name], axis=1)
        if merged_df.empty:
            merged_df = next_df
        else:
            merged_df = merged_df.join(next_df)
    if copy_better:
        merged_df = transfer_expectation_dataframe(merged_df)
    merged_df.to_csv(out_name)


def normalize_qaoa_angles(angles: ndarray) -> ndarray:
    """
    Normalizes angles to the [-pi; pi] range.
    :param angles: QAOA angles.
    :return: Normalized angles.
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def round_angles(angles: ndarray) -> ndarray:
    """
    Rounds angles to the nearest multiples of pi/4.
    :param angles: Input angles given in fractions of pi.
    :return: Rounded.
    """
    return np.round(angles * 4) / 4


class DiscreteTransform:
    """
    Base class implementing discrete sine and cosine transformations.
    :var func: sin or cos.
    """
    func: callable = None

    @classmethod
    def transform(cls, xs: ndarray) -> ndarray:
        """
        Calculates discrete sine or cosine transform of type 4 with the given sequence.
        :param xs: Sequence for transform.
        :return: Transformed sequence.
        """
        ys = np.zeros_like(xs, dtype=float)
        mult_factors = np.arange(len(xs)) + 0.5
        for i in range(len(ys)):
            ys[i] = np.dot(xs, cls.func(mult_factors * (i + 0.5) * np.pi / len(xs)))
        return ys

    @classmethod
    def inverse(cls, ys: ndarray) -> ndarray:
        """
        Inverses discrete transform.
        :param ys: Sequence transformed by the transform method.
        :return: Original input to the transform method.
        """
        return cls.transform(ys) * 2 / len(ys)


class DiscreteSineTransform(DiscreteTransform):
    func: callable = np.sin


class DiscreteCosineTransform(DiscreteTransform):
    func: callable = np.cos


class DataExtractor:
    """ Class that defines data extraction (rearrangement) methods. """

    def __init__(self, df_path: str):
        self.df = pd.read_csv(df_path, index_col=0)

    def get_ps(self) -> list:
        """
        Returns the values of computed p in this dataset.
        :return: p values.
        """
        ps = [int(col.split('_')[1]) for col in self.df.columns if re.match(r'p_\d+$', col)]
        return ps

    def get_ar_aggregated(self, aggregator: callable) -> Sequence:
        """
        Applies given aggregator to AR columns.
        :param aggregator: Aggregator function.
        :return: Aggregated result.
        """
        return self.df.filter(regex=r'p_\d+$').apply(aggregator)

    def get_cost_all(self) -> DataFrame:
        """
        Returns total cost for each graph.
        :return: Total cost for each graph.
        """
        ps = self.get_ps()
        cost_all = self.df.filter(regex=r'p_\d+_nfev$').cumsum(axis=1) * ps
        # cost_all = self.df.filter(regex=r'p_\d+_nfev$').cumsum(axis=1)
        return cost_all

    def get_cost_average(self) -> Sequence:
        """
        Returns average cost for each p.
        :return: Average cost for each p.
        """
        costs_all = self.get_cost_all()
        return costs_all.mean()

    def get_cost_worst_case(self) -> Sequence:
        """
        Returns cost for the worst case graphs.
        :return: Cost for the worst case graphs.
        """
        costs_all = self.get_cost_all()
        min_inds = self.get_ar_aggregated(np.argmin)
        costs_worst_case = [costs_all.iloc[min_inds[col_ind], col_ind] for col_ind in range(costs_all.shape[1])]
        return costs_worst_case
