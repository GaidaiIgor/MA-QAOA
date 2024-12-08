import itertools as it
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import numpy.random as random
import pandas as pd
from networkx import Graph
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from src.angle_strategies.direct import interp_qaoa_angles, convert_angles_qaoa_to_fourier, convert_angles_fourier_to_qaoa, convert_angles_ma_to_controlled_ma
from src.angle_strategies.guess_provider import GuessProviderBase
from src.angle_strategies.search_space import SearchSpaceGeneral, SearchSpace
from src.angle_strategies.basis_provider import BasisProviderBase
from src.data_processing import numpy_str_to_array, normalize_qaoa_angles
from src.graph_utils import get_index_edge_list
from src.optimization.optimization import optimize_qaoa_angles
from src.optimization.optimization import Evaluator
from src.preprocessing import evaluate_all_cuts, evaluate_z_term


@dataclass(kw_only=True)
class WorkerMaxCutBase(ABC):
    """
    Base class for a parallel worker solving MaxCut problem on given graphs.
    :var out_col: Name of the output column for the main result.
    """
    out_col: str

    @abstractmethod
    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        """
        Solves MaxCut described by a given dataframe row and returns the row extended with the work results (if applicable).
        :param job_item: 1) Whether current job item has to be processed or not 2) Series of graph properties and calculation results (dataframe row).
        :return: Series extended with work results.
        """
        pass


@dataclass(kw_only=True)
class WorkerExplicit(WorkerMaxCutBase, ABC):
    """
    Base class for workers that perform the work themselves, i.e. do not use other workers.
    :var reader: Function that reads graph from a file.
    """
    reader: callable


@dataclass(kw_only=True)
class WorkerMaxCutBruteForce(WorkerExplicit):
    """ Worker that evaluates maxcut by brute-force and saves it as a graph property (does not modify input series). """

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        execute, series = job_item
        if not execute:
            return series

        graph = self.reader(series['path'])
        cut_vals = evaluate_all_cuts(graph)
        max_cut = int(max(cut_vals))

        graph.graph['max_cut'] = max_cut
        return series


@dataclass(kw_only=True)
class WorkerQAOABase(WorkerExplicit, ABC):
    """
    Base class for workers that solve MaxCut optimization with QAOA-like methods.
    :var p: Number of QAOA layers.
    :var search_space: Name of known search space or instance of SearchSpace class.
    :var guess_provider: Guess provider for the initial guess.
    :var transfer_from: Name of the main column (AR) with the data for a smaller number of layers to compare for transfer.
    :var transfer_p: Number of layers (p) in the transferred data. If the current p is different, the copied angles will be appended with 0 to keep the angle format consistent.
    """
    p: int
    search_space: str | SearchSpace
    guess_provider: GuessProviderBase
    transfer_from: str | None = None
    transfer_p: int | None = None

    def get_evaluator(self, graph: Graph, search_space: str | SearchSpace = None) -> Evaluator:
        """
        Returns evaluator appropriate for the current class.
        :param graph: Graph for MaxCut evaluation.
        :param search_space: Custom search space, or None to use self.search_space.
        :return: Evaluator.
        """
        if search_space is None:
            search_space = self.search_space
        return Evaluator.get_evaluator_standard_maxcut(graph, self.p, search_space=search_space)

    def get_initial_angles(self, evaluator: Evaluator, series: Series, guess_provider: GuessProviderBase = None) -> tuple[ndarray, int]:
        """
        Returns initial angles for the optimization. The number of layers in the guess has to match the current number of layers.
        :param evaluator: Evaluator for which a guess is generated.
        :param series: Series to extract the angles from.
        :param guess_provider: Custom guess provider, or None ot use self.guess_provider.
        :return: 1) Initial angles for optimization. 2) Number of QPU calls to obtain these angles.
        """
        if guess_provider is None:
            guess_provider = self.guess_provider
        initial_angles, nfev = guess_provider.provide_guess(evaluator, series)
        return initial_angles, nfev

    def optimize_angles(self, evaluator: Evaluator, series: Series, starting_angles: ndarray, **kwargs) -> OptimizeResult:
        """
        Optimizes angles.
        :param evaluator: Expectation evaluator.
        :param series: Series describing data item.
        :param starting_angles: Starting angles for optimization.
        :return: Optimization result.
        """
        optimization_result = optimize_qaoa_angles(evaluator, starting_angles=starting_angles, series=series, **kwargs)
        return optimization_result

    def write_standard(self, series: Series, optimization_result: OptimizeResult):
        """
        Writes standard information from the optimization result into series.
        :param series: Series for writing.
        :param optimization_result: Optimization result for writing.
        """
        if optimization_result is not None:
            graph = self.reader(series['path'])
            series[self.out_col] = optimization_result.fun / graph.graph['maxcut']
            series[self.out_col + '_angles'] = optimization_result.x
            series[self.out_col + '_nfev'] = optimization_result.nfev

    def transfer_angles(self, series: Series, angle_suffix: str):
        """
        Transfers angles from a previous layer to the current layer, by augmenting with the appropriate number of zeros at the end.
        :param series: Data series with angles.
        :param angle_suffix: Suffix of angle record.
        :return: Updated series.
        """
        p_diff = self.p - self.transfer_p
        transfer_angles = numpy_str_to_array(series[self.transfer_from + angle_suffix])
        angles_per_layer = len(transfer_angles) // self.transfer_p
        transfer_angles = np.concatenate((transfer_angles, [0] * angles_per_layer * p_diff))
        series[self.out_col + angle_suffix] = transfer_angles

    def transfer_record(self, series: Series):
        """
        Transfers data from the comparison record if its result is better than the current result.
        :param series: Current data series.
        :return: Updated data series.
        """
        if self.transfer_from is None:
            return
        if self.out_col not in series or series[self.out_col] < series[self.transfer_from]:
            series[self.out_col] = series[self.transfer_from]
            self.transfer_angles(series, '_angles')
        if self.out_col + '_nfev' not in series:
            series[self.out_col + '_nfev'] = 0

    def update_series(self, series: Series, optimization_result: OptimizeResult | None) -> Series:
        """
        Updates given series with given optimization results.
        :param series: Series of graph properties and calculation results (dataframe row).
        :param optimization_result: Optimization result.
        :return: Updated series.
        """
        new_series = series.copy()
        self.write_standard(new_series, optimization_result)
        self.transfer_record(new_series)
        return new_series

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        """ Runs standard optimization and writes the results to the given series. """
        optimize, series = job_item
        if optimize:
            path = series['path']
            graph = self.reader(path)
            evaluator = self.get_evaluator(graph)
            starting_angles, nfev = self.get_initial_angles(evaluator, series)
            try:
                optimization_result = self.optimize_angles(evaluator, series, starting_angles)
            except Exception:
                raise Exception(f'Optimization failed at {path}')
            optimization_result.nfev += nfev
        else:
            optimization_result = None
        new_series = self.update_series(series, optimization_result)
        return new_series


@dataclass(kw_only=True)
class WorkerIterative(WorkerQAOABase):
    """ Worker that fixes previous best angles and optimizes only the last layer. """

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        optimize, series = job_item
        if optimize:
            path = series['path']
            graph = self.reader(path)
            evaluator = self.get_evaluator(graph)
            # DEBUG
            previous_angles = convert_angles_ma_to_controlled_ma(numpy_str_to_array(series[f'p_{evaluator.p - 1}_angles']), evaluator.num_phase_terms, evaluator.num_qubits)
            evaluator.fix_params(list(range(len(previous_angles))), previous_angles)
            starting_angles, nfev = self.get_initial_angles(evaluator, series)
            try:
                optimization_result = self.optimize_angles(evaluator, series, starting_angles)
            except Exception:
                raise Exception(f'Optimization failed at {path}')
            optimization_result.nfev += nfev
            optimization_result.x = np.concatenate((previous_angles, optimization_result.x))
        else:
            optimization_result = None
        new_series = self.update_series(series, optimization_result)
        return new_series


@dataclass(kw_only=True)
class WorkerStandard(WorkerQAOABase):
    """ Implements standard processing. """

    def __post_init__(self):
        assert np.any(self.search_space == np.array(['tqa', 'linear', 'qaoa', 'fourier', 'ma'])) or isinstance(self.search_space, SearchSpace), \
            'search_space can be tqa, linear, qaoa, fourier, ma, or an instance of SearchSpace for this worker'


@dataclass(kw_only=True)
class WorkerGeneral(WorkerQAOABase):
    """
    Implements MaxCut optimization with Generalized QAOA.
    :var search_space_type: Describes what terms should be used to construct the driver Hamiltonian.
    1 uses all first-order terms, 12 - all first and second-order terms, 12e - all first and second corresponding to the existing edges in graph only.
    """
    search_space: str = 'general'
    search_space_type: str

    def __post_init__(self):
        assert self.search_space == 'general', 'search space has to be general for this worker'
        assert np.any(self.search_space_type == np.array(['1', '12', '12e'])), 'search space type can be 1 or 12 or 12e for this worker'

    def get_evaluator(self, graph: Graph) -> Evaluator:
        target_vals = evaluate_all_cuts(graph)
        driver_term_vals = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 1)])
        if self.search_space_type != '1':
            if self.search_space_type == '12':
                driver_term_vals_2 = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 2)])
            if self.search_space_type == '12e':
                driver_term_vals_2 = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
            driver_term_vals = np.append(driver_term_vals, driver_term_vals_2, axis=0)

        evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, self.p, self.search_space)
        return evaluator


@dataclass(kw_only=True)
class WorkerGeneralSub(WorkerGeneral):
    """ Implements MaxCut optimization with Generalized QAOA on subgraphs generated by each edge (lightcones). """

    def get_evaluator(self, graph: Graph) -> Evaluator:
        target_terms = [set(edge) for edge in get_index_edge_list(graph)]
        target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]

        driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
        if self.search_space_type == '12':
            driver_terms += [set(term) for term in it.combinations(range(len(graph)), 2)]
        elif self.search_space_type == '12e':
            driver_terms += [set(term) for term in get_index_edge_list(graph)]

        evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, self.p, self.search_space)
        return evaluator


@dataclass(kw_only=True)
class WorkerSubspaceMA(WorkerQAOABase):
    """
    Worker that searches through subspaces of MA-QAOA.
    :var basis_provider: Object that provides search basis.
    """
    search_space: str = 'ma_subspace'
    basis_provider: BasisProviderBase

    def __post_init__(self):
        assert self.search_space == 'ma_subspace', 'Search space can only be ma_subspace for WorkerSubspaceMA'

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        optimize, series = job_item
        if optimize:
            path = series['path']
            graph = self.reader(path)
            evaluator_ma = self.get_evaluator(graph, 'ma')
            basis, nfev_basis = self.basis_provider.provide_basis(evaluator_ma, series)
            shift, nfev_shift = self.get_initial_angles(evaluator_ma, series)
            search_space = SearchSpaceGeneral(basis, shift)
            evaluator = self.get_evaluator(graph, search_space)
            initial_angles = np.array([0] * evaluator.num_angles)
            try:
                optimization_result = self.optimize_angles(evaluator, series, initial_angles)
            except Exception:
                raise Exception(f'Optimization failed at {path}')
            optimization_result.x = search_space.transform_coordinates(optimization_result.x)
            optimization_result.nfev += nfev_basis + nfev_shift
        else:
            optimization_result = None
        new_series = self.update_series(series, optimization_result)
        return new_series


@dataclass(kw_only=True)
class WorkerMultipleRepeats(WorkerQAOABase, ABC):
    """
    Worker that implements multiple repeats with different starting guesses.
    :var num_repeats: Number of repeats.
    """
    num_repeats: int

    @abstractmethod
    def get_initial_angles_all(self, evaluator: Evaluator, series: Series) -> ndarray:
        """
        Returns a 2D array where each row is an initial guess for optimization.
        :param evaluator: Evaluator for which initial angles are generated.
        :param series: Input series with graph properties.
        :return: 2D array of size self.num_repeats x number of angles in the current search space.
        """
        pass

    def update_series(self, series: Series, optimization_results: list[OptimizeResult] | None) -> Series:
        """
        Updates the series based on information from all optimization attempts.
        :param series: Series to update.
        :param optimization_results: List of optimization results from all attempts.
        :return: Updated series.
        """
        best_result = max(optimization_results, key=lambda result: result.fun) if optimization_results is not None else None
        updated_series = WorkerQAOABase.update_series(self, series, best_result)
        if optimization_results is not None:
            total_nfev = sum([result.nfev for result in optimization_results])
            updated_series[self.out_col + '_nfev'] = total_nfev
        return updated_series

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        optimize, series = job_item
        if optimize:
            path = series['path']
            graph = self.reader(path)
            evaluator = self.get_evaluator(graph)

            optimization_results = [None] * self.num_repeats
            initial_angles = self.get_initial_angles_all(evaluator, series)
            for i in range(self.num_repeats):
                try:
                    optimization_results[i] = self.optimize_angles(evaluator, series, initial_angles[i, :])
                except Exception:
                    raise Exception(f'Optimization failed at {path}')
        else:
            optimization_results = None
        new_series = self.update_series(series, optimization_results)
        return new_series


@dataclass(kw_only=True)
class WorkerIterativePerturb(WorkerMultipleRepeats, ABC):
    """
    Worker that implements iterative generation of the initial guess for next layer with perturbations.
    :var alpha: Perturbation multiplier coefficient.
    :var guess_provider_unperturbed:
    """
    alpha: float
    guess_provider_unperturbed: GuessProviderBase

    def __post_init__(self):
        assert self.p > 1, 'p has to be > 1 for this worker'

    @abstractmethod
    def extend_angles(self, angles: ndarray) -> ndarray:
        """
        Returns initial guess for the current layer based on the given angles for the previous layer.
        :param angles: Angles for the previous layer.
        :return: Angles extended for the current layer.
        """
        pass

    def get_initial_angles_all(self, evaluator: Evaluator, series: Series) -> ndarray:
        """ Extends existing angles to find a good guess for the next layer. """
        angles_unperturbed = WorkerQAOABase.get_initial_angles(self, evaluator, series, self.guess_provider_unperturbed)
        angles_best = WorkerQAOABase.get_initial_angles(self, evaluator, series)
        initial_angles_all = [None] * self.num_repeats
        perturb_start = 1 if np.allclose(angles_unperturbed, angles_best) else 2
        for i in range(self.num_repeats):
            initial_angles_all[i] = angles_unperturbed if i == 0 else angles_best
            if i >= perturb_start:
                perturbation = self.alpha * random.normal(scale=abs(initial_angles_all[i]))
                initial_angles_all[i] += perturbation
            initial_angles_all[i] = self.extend_angles(initial_angles_all[i])
        initial_angles_all = np.array(initial_angles_all)
        return initial_angles_all

    def update_series(self, series: Series, optimization_results: list[OptimizeResult] | None) -> Series:
        """
        Updates series with optimization results.
        :param series: Series.
        :param optimization_results: Tuple with best and unperturbed optimization results.
        :return: Updated series.
        """
        new_series = WorkerMultipleRepeats.update_series(self, series, optimization_results)
        new_series[self.out_col + '_angles_unperturbed'] = optimization_results[0].x
        return new_series


@dataclass(kw_only=True)
class WorkerInterp(WorkerIterativePerturb):
    """ Worker for Interp initialization strategy for QAOA. """
    search_space: str = 'qaoa'

    def __post_init__(self):
        WorkerIterativePerturb.__post_init__(self)
        assert self.search_space == 'qaoa', 'search space has to be qaoa for this worker'

    def extend_angles(self, angles: ndarray) -> ndarray:
        return interp_qaoa_angles(angles, self.p - 1)


@dataclass(kw_only=True)
class WorkerFourier(WorkerIterativePerturb):
    """ Worker function for Fourier initialization strategy for QAOA. """
    search_space: str = 'fourier'

    def __post_init__(self):
        WorkerIterativePerturb.__post_init__(self)
        assert self.search_space == 'fourier', 'search space has be to fourier for this worker'

    def get_initial_angles_all(self, evaluator: Evaluator, series: Series):
        initial_angles_all = WorkerIterativePerturb.get_initial_angles_all(self, evaluator, series)
        for i in range(initial_angles_all.shape[0]):
            initial_angles_all[i, :] = convert_angles_qaoa_to_fourier(initial_angles_all[i, :])
        return initial_angles_all

    def extend_angles(self, angles: ndarray) -> ndarray:
        return np.concatenate((angles, [0] * 2))

    def optimize_angles(self, evaluator: Evaluator, starting_angles: ndarray) -> OptimizeResult:
        optimization_result = optimize_qaoa_angles(evaluator, starting_angles=starting_angles, normalize_angles=False)
        return optimization_result

    def update_series(self, series: Series, optimization_results: list[OptimizeResult] | None) -> Series:
        if optimization_results is not None:
            for i in range(len(optimization_results)):
                optimization_results[i].x = normalize_qaoa_angles(convert_angles_fourier_to_qaoa(optimization_results[i].x))
        return WorkerIterativePerturb.update_series(self, series, optimization_results)


@dataclass(kw_only=True)
class WorkerGreedy(WorkerMultipleRepeats):
    """ Worker that implements greedy strategy for QAOA (optimizations from transition states). """
    search_space: str = 'qaoa'

    def __post_init__(self):
        assert self.p > 1, 'p has to be > 1 for WorkerGreedy'
        assert self.search_space == 'qaoa', 'search space has to be qaoa for WorkerGreedy'

    def get_initial_angles_all(self, evaluator: Evaluator, series: Series) -> ndarray:
        angles_per_layer = 2
        initial_angles = WorkerQAOABase.get_initial_angles(self, evaluator, series)
        selected_transitions = random.choice(np.arange(self.p), self.num_repeats, False)
        initial_angles_all = [None] * self.num_repeats
        for i, transition_ind in enumerate(selected_transitions):
            initial_angles_all[i] = np.concatenate((initial_angles[:angles_per_layer * transition_ind], [0] * angles_per_layer, initial_angles[angles_per_layer * transition_ind:]))
        return np.array(initial_angles_all)

    def optimize_angles(self, evaluator: Evaluator, starting_angles: ndarray) -> OptimizeResult:
        optimization_result = optimize_qaoa_angles(evaluator, starting_angles=starting_angles, method='Nelder-Mead')
        return optimization_result


@dataclass(kw_only=True)
class WorkerRepeater(WorkerMaxCutBase):
    """
    Worker that implements processing by calling another worker specified number of times and stores the series corresponding to the largest result.
    :var worker: Worker to repeat.
    :var num_repeats: Number of times to repeat.
    """
    worker: WorkerQAOABase
    num_repeats: int

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        optimize, series = job_item
        if not optimize:
            return self.worker.process_job_item(job_item)

        best_result = -np.inf
        best_series = None
        for _ in range(self.num_repeats):
            next_series = self.worker.process_job_item(job_item)
            if next_series[self.out_col] > best_result:
                best_result = next_series[self.out_col]
                best_series = next_series
        return best_series


@dataclass(kw_only=True)
class WorkerComposite(WorkerMaxCutBase, ABC):
    """
    Worker that implements some composition of other workers.
    :var workers: List of workers in the composition.
    """
    workers: list[WorkerQAOABase]


@dataclass(kw_only=True)
class WorkerSequential(WorkerComposite):
    """
    Worker that implements sequential composition of other workers.
    The inner workers exchange their data by updating the input series and must be properly initialized to enable the desired interaction pattern between them.
    Each worker must implement conversion (if needed) form the search space of the previous worker if their final angles are to be used as initial angles for the next worker.
    """

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        optimize, updated_series = job_item
        if not optimize:
            return self.workers[0].process_job_item(job_item)

        for worker in self.workers:
            updated_series = worker.process_job_item((True, updated_series))
        return updated_series


@dataclass(kw_only=True)
class WorkerSequentialRepeater(WorkerSequential):
    """
    Specialization of WorkerSequential for the cases when the chain is executed multiple times, and we want to ensure that each time each worker in the chain produces different
    angles for the next worker. The optimized angles of each internal worker have to be written to self.out_col + '_angles' column (standard worker behavior).
    If a given worker produces angles that have already been tried, then the angles will be replaced by random angles for the next worker.
    """
    num_repeats: int
    similarity_threshold: float = 1e-3

    def process_job_item(self, job_item: tuple[bool, Series]) -> Series:
        optimize, original_series = job_item
        if not optimize:
            return self.workers[0].process_job_item(job_item)

        best_result = -np.inf
        best_series = None
        tried_angles = [set() for _ in range(len(self.workers) - 1)]
        for i in range(self.num_repeats):
            updated_series = original_series.copy()
            for j, worker in enumerate(self.workers):
                updated_series = worker.process_job_item((True, updated_series))
                if j < len(tried_angles):
                    next_angles = updated_series[worker.out_col + '_angles']
                    similar = [max(abs(next_angles - angles)) < self.similarity_threshold for angles in tried_angles[j]]
                    if any(similar):
                        next_angles = random.uniform(-np.pi / 2, np.pi / 2, len(next_angles))
                        updated_series[worker.out_col + '_angles'] = next_angles
                    else:
                        tried_angles[j].add(next_angles)
            if updated_series[self.out_col] > best_result:
                best_result = updated_series[self.out_col]
                best_series = updated_series
        return best_series


def optimize_expectation_parallel(dataframe_path: str, rows_func: callable, num_workers: int, worker: WorkerMaxCutBase):
    """
    Optimizes cut expectation for a given set of graphs in parallel and writes the output dataframe.
    :param dataframe_path: Path to input dataframe with information about jobs.
    :param rows_func: Function that accepts dataframe and returns boolean array identifying which rows of the dataframe should be considered.
    :param num_workers: Number of parallel workers.
    :param worker: Worker instance.
    """
    df = pd.read_csv(dataframe_path)
    selected_rows = rows_func(df)
    job_items = list(zip(selected_rows, [tuple[1] for tuple in df.iterrows()]))

    results = []
    if num_workers == 1:
        for result in tqdm(map(worker.process_job_item, job_items), total=len(job_items), smoothing=0, ascii=' █'):
            results.append(result)
    else:
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(worker.process_job_item, job_items), total=len(job_items), smoothing=0, ascii=' █'):
                results.append(result)

    # df = DataFrame(results).sort_values('path', key=natsort_keygen())
    df = DataFrame(results).sort_index()
    df.to_csv(dataframe_path, index=False)

    if isinstance(worker, WorkerQAOABase):
        dataset_id = re.search(r'nodes_\d+/depth_\d+', dataframe_path)[0]
        print(f'dataset: {dataset_id}; p: {worker.p}; mean: {np.mean(df[worker.out_col]):.3f}; min: {min(df[worker.out_col]):.3f}; converged: {sum(df[worker.out_col] > 0.9995)}; '
              f'nfev: {np.mean(df[worker.out_col + "_nfev"].where(lambda x: x != 0)):.0f}\n')
