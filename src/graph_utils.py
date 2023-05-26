"""
Graph utilities.
"""

import numpy as np
from networkx import Graph
from numpy import ndarray


def get_index_edge_list(graph: Graph, edge_list: list[tuple[int, int]] = None) -> ndarray:
    """
    Returns 2D array of edges specified by pairs of node indices in the order of graph.nodes instead of node labels.
    :param graph: Graph to consider.
    :param edge_list: List of edges that should be taken into account. If None, then all edges are taken into account.
    :return: 2D array of size graph.edges x 2, where each edge is specified by node indices instead of labels.
    """
    if edge_list is None:
        edge_list = graph.edges

    node_indices = {node: ind for ind, node in enumerate(graph.nodes)}
    index_edge_list = []
    for edge in edge_list:
        index_edge_list.append([node_indices[edge[0]], node_indices[edge[1]]])
    return np.array(index_edge_list)
