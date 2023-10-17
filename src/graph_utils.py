"""
Graph utilities.
"""
from queue import SimpleQueue
from typing import Sequence

import networkx as nx
import numpy as np
from networkx import Graph
from numpy import ndarray


def edge_bfs(graph: Graph, starting_edge: tuple) -> dict[tuple, int]:
    """
    Carries out edge BFS from the specified edge and returns distances to all other edges.
    :param graph: Graph for BFS.
    :param starting_edge: Starting edge.
    :return: Distances to all edges starting from the given edge.
    """
    if starting_edge[0] > starting_edge[1]:
        starting_edge = starting_edge[::-1]
    distances = {starting_edge: 0}
    bfs_queue = SimpleQueue()
    bfs_queue.put(starting_edge)
    while not bfs_queue.empty():
        parent_edge = bfs_queue.get()
        for node in parent_edge:
            for edge in graph.edges(node):
                if edge[0] > edge[1]:
                    edge = edge[::-1]
                if edge not in distances:
                    distances[edge] = distances[parent_edge] + 1
                    bfs_queue.put(edge)
    return distances


def get_max_edge_depth(graph: Graph) -> int:
    """
    Returns worst case depth of edge BFS.
    :param graph: Graph in question.
    :return: Worst case depth of edge BFS.
    """
    depths = []
    for edge in graph.edges:
        distances = edge_bfs(graph, edge)
        depths.append(max(distances.values()))
    return max(depths)


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


def get_node_indices(graph: Graph) -> dict:
    """
    Returns a dict that maps node labels to their indices.
    :param graph: Graph.
    :return: Dict that maps node labels to their indices.
    """
    return {node: i for i, node in enumerate(graph.nodes)}


def get_index_edge_list(graph: Graph, edge_list: list[tuple[int, int]] = None) -> ndarray:
    """
    Returns 2D array of edges specified by pairs of node indices in the order of graph.nodes instead of node labels.
    :param graph: Graph to consider.
    :param edge_list: List of edges that should be taken into account. If None, then all edges are taken into account.
    :return: 2D array of size graph.edges x 2, where each edge is specified by node indices instead of labels.
    """
    if edge_list is None:
        edge_list = graph.edges

    node_indices = get_node_indices(graph)
    index_edge_list = []
    for edge in edge_list:
        index_edge_list.append([node_indices[edge[0]], node_indices[edge[1]]])
    return np.array(index_edge_list)


def read_graph_xqaoa(path):
    """
    Reads a graph in XQAOA format
    :param path: Path to graph file.
    :return: Read graph.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = lines[2:]
    graph = nx.read_edgelist(lines, delimiter=',', nodetype=int)
    nx.set_edge_attributes(graph, 1, 'weight')
    return graph


def is_isomorphic(graph: Graph, other_graphs: Sequence) -> bool:
    """
    Checks if given graph is isomorphic to any of the other graphs.
    :param graph: Graph to check.
    :param other_graphs: Graphs to check against.
    :return: True if the graph is isomorphic to any of the graphs, False otherwise.
    """
    for i in range(len(other_graphs)):
        if nx.is_isomorphic(graph, other_graphs[i]):
            return True
    return False


def find_non_isomorphic(graphs: Sequence) -> list[bool]:
    """
    Finds non-isomorphic graphs among the given iterable.
    :param graphs: Graphs to search.
    :return: Boolean list such that if True elements are taken from graphs, then none of them will be isomorphic.
    """
    res = [True] * len(graphs)
    for pivot in range(len(graphs)):
        if not res[pivot]:
            continue
        for i in range(pivot + 1, len(graphs)):
            if nx.is_isomorphic(graphs[pivot], graphs[i]):
                res[i] = False
    return res
