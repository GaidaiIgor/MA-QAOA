""" Module with extension of Evaluator adding evaluation methods that work through qiskit. """
import qiskit
from networkx import Graph
from qiskit import converters
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_ibm_runtime import Options, Session, QiskitRuntimeService, Estimator as IbmEstimator

from src.optimization.evaluator import Evaluator
from src.simulation.qiskit_backend import get_qaoa_ansatz, get_observable_maxcut, evaluate_angles_ma_qiskit


class EvaluatorQiskit(Evaluator):
    """ Extension of evaluator that adds qiskit-backend evaluation methods. """

    @staticmethod
    def get_evaluator_standard_maxcut_qiskit_simulator(graph: Graph, p: int, search_space: str = 'ma') -> Evaluator:
        """
        Returns evaluator of maxcut expectation evaluated via qiskit's Aer simulator.
        :param graph: Graph for maxcut.
        :param p: Number of QAOA layers.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Evaluator that computes maxcut expectation achieved by MA-QAOA with given angles.
        The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
        """
        ansatz = get_qaoa_ansatz(graph, p)
        observable = get_observable_maxcut(graph)

        estimator = AerEstimator(approximation=True, run_options={'shots': 1024})
        # backend = Aer.get_backend('aer_simulator')
        # estimator = BackendEstimator(backend)

        func = lambda angles: evaluate_angles_ma_qiskit(angles, ansatz, estimator, observable)
        return Evaluator.wrap_parameter_strategy(func, len(graph), len(graph.edges), p, search_space)

    @staticmethod
    def get_evaluator_standard_maxcut_qiskit_hardware(graph: Graph, p: int, search_space: str = 'ma') -> Evaluator:
        """
        Returns qiskit evaluator of maxcut expectation evaluated via IBM's hardware.
        :param graph: Graph for maxcut.
        :param p: Number of QAOA layers.
        :param search_space: Name of the strategy to choose the number of variable parameters.
        :return: Evaluator that computes maxcut expectation achieved by MA-QAOA with given angles.
        The order of input parameters is the same as in `get_evaluator_standard_maxcut`.
        """
        ansatz = get_qaoa_ansatz(graph, p)
        observable = get_observable_maxcut(graph)
        service = QiskitRuntimeService()
        backend = service.get_backend('ibm_osaka')

        ansatz_transpiled = qiskit.transpile(ansatz, backend, optimization_level=3)
        observable = observable.apply_layout(ansatz_transpiled.layout)

        dag = converters.circuit_to_dag(ansatz_transpiled)
        idle_qubits = list(dag.idle_wires())
        qubits_used = dag.num_qubits() - len(idle_qubits)
        print(f'Qubits: {qubits_used}; Depth: {ansatz_transpiled.depth()}')

        session = Session(backend=backend)
        options = Options()
        options.resilience_level = 2
        estimator = IbmEstimator(session=session, options=options)

        func = lambda angles: evaluate_angles_ma_qiskit(angles, ansatz_transpiled, estimator, observable)
        return Evaluator.wrap_parameter_strategy(func, len(graph), len(graph.edges), p, search_space)
