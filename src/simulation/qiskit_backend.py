import time

import networkx as nx
from networkx import Graph
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli, SparsePauliOp
from scipy import optimize
from qiskit_aer.primitives import Estimator as AerEstimator

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


def get_observable_negative_maxcut(graph: Graph) -> SparsePauliOp:
    """
    Returns negative MaxCut Hamiltonian for minimization.
    :param graph: Graph for maxcut.
    :return: MaxCut Hamiltonian.
    """
    edges = get_index_edge_list(graph)
    return SparsePauliOp.from_sparse_list([('', [], -edges.shape[0] / 2)] + [('ZZ', list(edge), 0.5) for edge in edges], len(graph))


def optimize_angles_ma_qiskit(graph: Graph, p: int) -> float:
    """
    Optimizes MA-QAOA angles and returns optimized cut expectation value.
    :param graph: Graph for maxcut.
    :param p: Number of QAOA layers.
    :return: Optimized cut expectation value.
    """
    estimator = Estimator()
    ansatz = get_ma_ansatz(graph, p)
    optimizer = optimize.minimize
    vqe = VQE(estimator, ansatz, optimizer)
    maxcut_hamiltonian = get_observable_negative_maxcut(graph)
    result = vqe.compute_minimum_eigenvalue(maxcut_hamiltonian)
    return -result.eigenvalue.real


def evaluate_angles_ma_qiskit(graph: Graph, p: int, angles: ndarray) -> float:
    maxcut_hamiltonian = get_observable_negative_maxcut(graph)

    # estimator = Estimator()
    # ansatz = get_ma_ansatz(graph, p)
    # job = estimator.run(ansatz, maxcut_hamiltonian, angles)

    estimator = AerEstimator(approximation=False, run_options={'shots': 1024})
    ansatz = get_ma_ansatz(graph, p).bind_parameters(angles)
    job = estimator.run(ansatz, maxcut_hamiltonian)

    return -job.result().values


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
