"""
Graph utilities.
"""
import networkx as nx
import numpy as np
from networkx import Graph
from numpy import ndarray
from queue import SimpleQueue


def find_edge_index(graph: Graph, edge: tuple[int, int]) -> int:
    """
    Finds the index of specified edge in graph.edges view or -1 if the edge is not found. Does not take into account nodes order of the edge.
    :param graph: Graph for search.
    :param edge: Edge being searched.
    :return: Index of the edge in graph.edges.
    """
    ind = -1
    for ind, e in enumerate(graph.edges):
        if edge == e or edge == e[-1::-1]:
            return ind
    return ind


def get_p_subgraph(graph: Graph, edge: tuple[int, int], p: int) -> Graph:
    """
    Returns subgraph that consists from edges within distance p from the source edge via depth-limited edge BFS.
    Graph elements outside of this subgraph do not matter for calculation of edge p-expectation.
    :param graph: Original graph.
    :param edge: Inducing edge.
    :param p: Number of QAOA layers.
    :return: Induced subgraph.
    """
    edge_queue = SimpleQueue()
    edge_queue.put((edge, 0))
    edges_found = {edge}
    while not edge_queue.empty():
        next_edge, distance = edge_queue.get()
        if distance < p:
            incident_edges = graph.edges(next_edge)
            for incident in incident_edges:
                if incident not in edges_found:
                    edge_queue.put((incident, distance + 1))
                    edges_found.add(incident)
    return graph.edge_subgraph(edges_found)


def ragged_list_to_numpy_matrix(ragged: list[list[int]]) -> ndarray:
    """
    Converts a ragged list representing upper triangular part of a matrix to a square numpy matrix.
    :param ragged: Each element of this list represents a row in upper triangular matrix, therefore the length of each subsequent element must be 1 less
    compared to the previous, ending with an array of 1 element.
    :return: 2D numpy array with the elements of the ragged array written in the upper triangular area of the matrix.
    """
    size = len(ragged[0]) + 1
    matrix = np.zeros((size, size))
    for i, row in enumerate(ragged):
        matrix[i, i + 1:] = row
    return matrix


def read_graphs_adjacency_matrix(path: str) -> tuple[list[Graph], list[float]]:
    """
    Reads all graphs from a given file. Expects the following format: 1st line: empty, 2nd line: arbitrary name.
    This line should end with a number equal to max cut on the graph given below.
    3rd line - n characters that can be 0 or 1, representing the first row of the graph's upper triangular adjacency matrix (except the diagonal element).
    Next n-1 lines: other rows of upper triangular adjacency matrix.
    Same format as 3rd line, but each subsequent line is 1 character shorter than the previous line.
    Then the same format repeats for the next graph in file and so on.
    :param path: Path to the file with the graphs in the specified format.
    :return: List of read graphs and list of maxcut values for each graph.
    """
    header_lines = 2
    graphs = []
    max_objectives = []
    with open(path) as f:
        while True:
            for _ in range(header_lines):
                next_line = f.readline()

            if next_line == "":
                break
            else:
                next_max = float(next_line.split()[-1])
                max_objectives.append(next_max)

            next_line = f.readline().rstrip()
            rows = [[int(char) for char in next_line]]
            for i in range(len(rows[0]) - 1):
                next_line = f.readline().rstrip()
                rows.append([int(char) for char in next_line])
            adjacency_matrix = ragged_list_to_numpy_matrix(rows)
            graph = nx.from_numpy_array(adjacency_matrix)
            graphs.append(graph)

    return graphs, max_objectives
