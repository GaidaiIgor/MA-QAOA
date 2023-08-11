import time

import networkx as nx
import numpy as np
from networkx import Graph
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from scipy import optimize

from src.graph_utils import get_index_edge_list


def get_ma_ansatz(graph: Graph, p: int) -> QuantumCircuit:
    """
    Returns parametrized MA-QAOA ansatz for VQE.
    :param graph: Graph for maxcut.
    :param p: Number of QAOA layers.
    :return: Parametrized MA-QAOA ansatz for VQE.
    """
    edges = get_index_edge_list(graph)
    params = ParameterVector('angles', (len(graph) + len(graph.edges)) * p)

    circuit = QuantumCircuit(len(graph))
    circuit.h(range(len(graph)))
    ind = 0
    for layer in range(p):
        for edge in edges:
            circuit.hamiltonian(Pauli('ZZ'), params[ind], list(edge))
            ind += 1
        for node in range(len(graph)):
            circuit.hamiltonian(Pauli('X'), params[ind], [node])
            ind += 1
    return circuit


def get_observable_maxcut(graph: Graph) -> SparsePauliOp:
    """
    Returns MaxCut Hamiltonian.
    :param graph: Graph for maxcut.
    :return: MaxCut Hamiltonian.
    """
    edges = get_index_edge_list(graph)
    return SparsePauliOp.from_sparse_list([('', [], edges.shape[0] / 2)] + [('ZZ', list(edge), -0.5) for edge in edges], len(graph))


def optimize_angles_ma_qiskit(graph: Graph, p: int) -> float:
    """
    Optimizes MA-QAOA angles and returns optimized cut expectation value.
    :param graph: Graph for maxcut.
    :param p: Number of QAOA layers.
    :return: Optimized cut expectation value.
    """
    estimator = Estimator()
    # estimator = AerEstimator(approximation=True, run_options={'shots': None})
    ansatz = get_ma_ansatz(graph, p)
    optimizer = optimize.minimize
    initial_point = np.ones((ansatz.num_parameters,)) * np.pi / 8
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
    maxcut_hamiltonian = get_observable_maxcut(graph)
    result = vqe.compute_minimum_eigenvalue(-maxcut_hamiltonian)
    return -result.eigenvalue.real


def evaluate_angles_ma_qiskit_fast(angles: ndarray, ansatz: QuantumCircuit, estimator: AerEstimator, hamiltonian: SparsePauliOp) -> float:
    bound_circuit = ansatz.bind_parameters(angles)
    job = estimator.run(bound_circuit, hamiltonian)
    expectation = job.result().values[0]
    return expectation


def evaluate_angles_ma_qiskit(angles: ndarray, graph: Graph, p: int) -> float:
    """
    Evaluates maxcut expectation with MA-QAOA for given angles.
    :param angles: MA-QAOA angles.
    :param graph: Graph for maxcut.
    :param p: Number of QAOA layers.
    :return: Maxcut expectation value.
    """
    maxcut_hamiltonian = get_observable_maxcut(graph)
    # estimator = AerEstimator(approximation=True, run_options={'shots': None})
    estimator = Estimator()
    ansatz = get_ma_ansatz(graph, p).bind_parameters(angles)
    job = estimator.run(ansatz, maxcut_hamiltonian)
    return job.result().values[0]


if __name__ == "__main__":
    start = time.perf_counter()
    # graph = nx.read_gml('../../graphs/nodes_6/0.gml', destringizer=int)
    graph = nx.complete_graph(8)
    p = 1
    # num_angles = (len(graph) + len(graph.edges)) * p
    # angles = np.array([np.pi / 4] * num_angles)
    expectation = optimize_angles_ma_qiskit(graph, p)
    print(expectation)
    end = time.perf_counter()
    print(f'Time elapsed: {end - start}')
