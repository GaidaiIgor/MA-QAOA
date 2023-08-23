import glob
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


def calculate_min_p(df: DataFrame):
    """
    Finds minimum value of p necessary to achieve maxcut and appends it to the given dataframe.
    :param df: Input dataframe with graphs and calculation results for multiple p values.
    :return: Modified dataframe with added minimum values of p.
    """
    min_p = [0] * df.shape[0]
    cols = [col for col in df.columns if col[:2] == 'p_' and col[-1].isdigit()]
    p_vals = [int(col.split('_')[1]) for col in cols]
    for i in range(len(min_p)):
        row = df.iloc[i, :][cols]
        p_index = np.where(row > 0.9995)[0]
        min_p[i] = p_vals[p_index[0]] if len(p_index) > 0 else np.inf

    df['min_p'] = min_p
    df['p_rel_ed'] = df['min_p'] - df['edge_diameter']
    return df


def numpy_str_to_array(array_string: str) -> ndarray:
    """
    Converts numpy array string representation back to array.
    :param array_string: Numpy array string.
    :return: Numpy array.
    """
    return np.array([float(item) for item in array_string[1:-1].split()])
