"""
Graph utilities.
"""
import networkx as nx
import numpy as np
from networkx import Graph
from numpy import ndarray


def get_edge_diameter(graph: Graph) -> int:
    """
    Returns edge diameter of the graph, i.e. maximum number of BFS layers necessary to discover all edges.
    :param graph: Graph.
    :return: Edge diameter.
    """
    peripheral_nodes = nx.periphery(graph)
    diameter = nx.diameter(graph)
    for node in peripheral_nodes:
        last_edge = list(nx.edge_bfs(graph, node))[-1]
        if nx.shortest_path_length(graph, node, last_edge[0]) == diameter and nx.shortest_path_length(graph, node, last_edge[1]) == diameter:
            return diameter + 1
    return diameter


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
