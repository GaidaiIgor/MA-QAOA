from networkx import Graph
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info import SparsePauliOp

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
            circuit.cx(edge[0], edge[1])
            circuit.rz(2 * params[ind], edge[1])
            circuit.cx(edge[0], edge[1])
            ind += 1

        for node in range(len(graph)):
            circuit.rx(2 * params[ind], node)
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


def evaluate_angles_ma_qiskit(angles: ndarray, ansatz: QuantumCircuit, estimator: BaseEstimator, hamiltonian: SparsePauliOp) -> float:
    """
    Evaluates maxcut expectation with MA-QAOA for given angles, using already constructed qiskit classes.
    :param angles: MA-QAOA angles.
    :param ansatz: Quantum circuit with the same number of parameters as len(angles).
    :param estimator: Estimator that runs the job.
    :param hamiltonian: Target Hamiltonian.
    :return: Maxcut expectation value.
    """
    bound_circuit = ansatz.bind_parameters(angles)
    job = estimator.run(bound_circuit, hamiltonian)
    expectation = job.result().values[0]
    return expectation


# def evaluate_angles_ma_qiskit(angles: ndarray, graph: Graph, p: int) -> float:
#     """
#     Evaluates maxcut expectation with MA-QAOA for given angles.
#     :param angles: MA-QAOA angles.
#     :param graph: Graph for maxcut.
#     :param p: Number of QAOA layers.
#     :return: Maxcut expectation value.
#     """
#     maxcut_hamiltonian = get_observable_maxcut(graph)
#     # estimator = IBMRuntimeEstimator("ibmq_qasm_simulator", options={'shots': 10000, 'resilience_level': 2})
#     estimator = AerEstimator(approximation=False, run_options={'shots': 10000000})
#     # estimator = PrimitiveEstimator()
#     ansatz = get_ma_ansatz(graph, p).bind_parameters(angles)
#     job = estimator.run(ansatz, maxcut_hamiltonian)
#     return job.result().values[0]


# def optimize_angles_ma_qiskit(graph: Graph, p: int) -> float:
#     """
#     Optimizes MA-QAOA angles and returns optimized cut expectation value.
#     :param graph: Graph for maxcut.
#     :param p: Number of QAOA layers.
#     :return: Optimized cut expectation value.
#     """
#     # estimator = PrimitiveEstimator()
#     estimator = AerEstimator(approximation=False, run_options={'shots': 1e4})
#     ansatz = get_ma_ansatz(graph, p)
#     optimizer = partial(optimize.minimize, options={'disp': True})
#     # ansatz.parameter_bounds = [(-np.pi, np.pi)] * ansatz.num_parameters
#     # optimizer = SNOBFIT(verbose=True)
#     initial_point = np.ones((ansatz.num_parameters,)) * np.pi / 8
#     vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
#     maxcut_hamiltonian = get_observable_maxcut(graph)
#     result = vqe.compute_minimum_eigenvalue(-maxcut_hamiltonian)
#     return -result.eigenvalue.real


# def evaluate_angles_ma_qiskit_debug(angles: ndarray, ansatz: QuantumCircuit, simulator: AerSimulator, hamiltonian: SparsePauliOp) -> float:
#     bound_circuit = ansatz.bind_parameters(angles)
#     result = simulator.run(bound_circuit).result()
#     statevector = result.get_statevector()
#     expectation = statevector.expectation_value(hamiltonian)
#     return expectation
