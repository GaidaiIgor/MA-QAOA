from networkx import Graph
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BaseEstimator, BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit_ibm_runtime import Session, IBMRuntimeError

from src.graph_utils import get_index_edge_list


def get_qaoa_ansatz(graph: Graph, p: int, search_space: str = 'ma') -> QuantumCircuit:
    """
    Returns parametrized MA-QAOA ansatz for VQE.
    :param graph: Graph for maxcut.
    :param p: Number of QAOA layers.
    :param search_space: Name of the strategy to choose the number of variable parameters. ma or qaoa.
    :return: Parametrized MA-QAOA ansatz for VQE.
    """
    edges = get_index_edge_list(graph)
    if search_space == 'qaoa':
        num_params = 2 * p
    elif search_space == 'ma':
        num_params = (len(graph) + len(graph.edges)) * p
    else:
        raise Exception('Unknown search space')
    params = ParameterVector('angles', num_params)

    circuit = QuantumCircuit(len(graph))
    circuit.h(range(len(graph)))
    param_ind = 0
    for layer in range(p):
        for edge in edges:
            circuit.cx(edge[0], edge[1])
            circuit.rz(2 * params[param_ind], edge[1])
            circuit.cx(edge[0], edge[1])
            if search_space == 'ma':
                param_ind += 1

        if search_space == 'qaoa':
            param_ind += 1

        for node in range(len(graph)):
            circuit.rx(2 * params[param_ind], node)
            if search_space == 'ma':
                param_ind += 1

        if search_space == 'qaoa':
            param_ind += 1

    return circuit


def get_observable_maxcut(graph: Graph) -> SparsePauliOp:
    """
    Returns MaxCut Hamiltonian.
    :param graph: Graph for maxcut.
    :return: MaxCut Hamiltonian.
    """
    edges = get_index_edge_list(graph)
    return SparsePauliOp.from_sparse_list([('', [], edges.shape[0] / 2)] + [('ZZ', list(edge), -0.5) for edge in edges], len(graph))


def evaluate_angles_ma_qiskit(angles: ndarray, ansatz: QuantumCircuit, estimator: BaseEstimatorV2, hamiltonian: SparsePauliOp) -> float:
    """
    Evaluates maxcut expectation with MA-QAOA for given angles, using already constructed qiskit classes.
    :param angles: MA-QAOA angles.
    :param ansatz: Quantum circuit with the same number of parameters as len(angles).
    :param estimator: Estimator that runs the job.
    :param hamiltonian: Target Hamiltonian.
    :return: Maxcut expectation value.
    """
    bound_circuit = ansatz.assign_parameters(angles)
    try:
        job = estimator.run([(bound_circuit, hamiltonian)])
    except IBMRuntimeError:
        # Re-open timed out session
        backend = estimator.session.backend()
        estimator.session = Session(backend=backend)
        job = estimator.run([(bound_circuit, hamiltonian)])
    expectation = job.result()[0].data.evs
    return expectation
