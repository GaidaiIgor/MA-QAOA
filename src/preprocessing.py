"""
Functions that calculate auxiliary data structures to speed up quantum simulation.
"""
from dataclasses import dataclass

import numpy as np
from networkx import Graph
from numba import njit
from numpy import ndarray
import networkx as nx

from src.graph_utils import get_p_subgraph, find_edge_index


@dataclass
class PSubgraph:
    """
    A class that represents edge-induced p-subgraph for QAOA.
    :var graph: Subgraph itself.
    :var cut_vals: 2D array of size num_edges x 2 ** num_nodes with cut values for each edge in each node labeling.
    :var edge_inds: 1D array of indices for edges that should be taken into account when calculating expectation.
    :var angle_map: 1D array of indices for overall angles that pertain to this subgraph.
    """
    graph: Graph
    cut_vals: ndarray
    edge_inds: ndarray
    angle_map: ndarray


@njit
def get_edge_cut(edge: tuple[int, int], num_nodes: int) -> ndarray:
    """
    Returns edge cut values for all binary labelings.
    :param edge: Edge whose cut is being considered.
    :param num_nodes: Total number of nodes in the graph.
    :return: 1D array of size 2 ** num_nodes, where each element corresponds to a binary labeling (in lexicographical order) separating the nodes of the graph
    into 2 disjoint sets, and is equal to 1 if the considered edge is between the sets of a given labeling and 0 otherwise.
    """
    edge_cut = np.zeros(2 ** num_nodes, dtype=np.int32)
    ind_u_right = num_nodes - edge[0] - 1
    ind_v_right = num_nodes - edge[1] - 1
    for i in range(len(edge_cut)):
        bit_u = i >> ind_u_right & 1
        bit_v = i >> ind_v_right & 1
        edge_cut[i] = bit_u ^ bit_v
    return edge_cut


def preprocess_subgraphs(graph: Graph, p: int, edge_list: list[tuple[int, int]] = None) -> list[PSubgraph]:
    """
    Calculates p-subgraphs and related structures for each edge in the list.
    :param graph: Considered graph.
    :param p: Number of layers of QAOA.
    :param edge_list: List of considered edges in graph. If None, all edges are considered.
    :return: List of p-subgraphs for all specified edges.
    """
    if edge_list is None:
        edge_list = graph.edges

    nx.set_node_attributes(graph, {node: ind for ind, node in enumerate(graph.nodes)}, name='index')
    nx.set_edge_attributes(graph, {edge: ind for ind, edge in enumerate(graph.edges)}, name='index')
    subgraphs = []
    for edge in edge_list:
        next_subgraph = get_p_subgraph(graph, edge, p)
        subgraph_node_indices = {node: ind for ind, node in enumerate(next_subgraph.nodes)}
        cut_vals = np.array([get_edge_cut((subgraph_node_indices[u], subgraph_node_indices[v]), len(next_subgraph)) for u, v in next_subgraph.edges])
        edge_inds = np.array([find_edge_index(next_subgraph, edge)])

        angle_map = []
        for u, v, ind in next_subgraph.edges.data('index'):
            angle_map.append(ind)
        for u, ind in next_subgraph.nodes.data('index'):
            angle_map.append(ind + graph.number_of_edges())
        angle_map = np.array(angle_map)

        subgraphs.append(PSubgraph(next_subgraph, cut_vals, edge_inds, angle_map))
    return subgraphs


def find_max_cut(graph: Graph) -> tuple[int, int]:
    all_cuv_vals = np.array([get_edge_cut(edge, len(graph)) for edge in graph.edges])
    obj_vals = np.sum(all_cuv_vals, 0)
    max_ind = int(np.argmax(obj_vals))
    return obj_vals[max_ind], max_ind
