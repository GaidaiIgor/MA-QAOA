import itertools as it
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Pool

import networkx as nx
import numpy as np
import numpy.random as random
import pandas as pd
from natsort import natsort_keygen
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm

from src.angle_strategies import convert_angles_qaoa_to_ma, convert_angles_linear_to_qaoa, convert_angles_tqa_to_qaoa, interp_qaoa_angles, convert_angles_qaoa_to_fourier, \
    convert_angles_fourier_to_qaoa
from src.data_processing import numpy_str_to_array, transfer_expectation_columns, normalize_qaoa_angles
from src.graph_utils import get_index_edge_list
from src.optimization import Evaluator, optimize_qaoa_angles
from src.preprocessing import evaluate_graph_cut, evaluate_z_term


@dataclass(kw_only=True)
class WorkerAbstract(ABC):
    """
    Base abstract worker class defining worker interface.
    :var reader: Function that reads graph from the file.
    """
    reader: callable

    @abstractmethod
    def process_entry(self, entry: tuple[str, Series]) -> Series:
        """
        Performs work on a given dataframe row and returns the row extended with the work results.
        :param entry: 1) Path to graph file; 2) Series of graph attributes (dataframe row).
        :return: Series extended with work results.
        """
        pass


@dataclass(kw_only=True)
class WorkerBaseQAOA(WorkerAbstract, ABC):
    """
    Base abstract QAOA worker class implementing common postprocessing and attributes for QAOA workers.
    :var search_space: Name of angle search space (general, ma, qaoa, fourier, linear, tqa).
    :var p: Number of QAOA layers.
    :var out_col: Name of the output column for AR.
    :var initial_guess_from: Name of the primary column from where the corresponding angles will be taken as initial guess or None for random angles.
    :var transfer_from: Name of the column from where expectation should be copied if it was not calculated in the current round (=nan).
    :var transfer_p: Value of p for the copy column. If current p is different, the copied angles will be appended with 0 to keep the angle format consistent.
    """
    search_space: str
    p: int
    out_col: str
    initial_guess_from: str | None = None
    transfer_from: str | None = None
    transfer_p: int | None = None

    def postprocess_dataframe(self, dataframe: DataFrame) -> DataFrame:
        """
        Transfers missing or better expectations from the previous p. Updates angles to match the format of current p.
        :param dataframe: Dataframe with the current p calculations.
        :return: Updated dataframe with copies expectations and angles.
        """
        if self.transfer_from is not None:
            dataframe = transfer_expectation_columns(dataframe, self.transfer_from, self.out_col, ['_angles'], self.transfer_p, self.p, True)
        return dataframe


@dataclass(kw_only=True)
class WorkerGeneral(WorkerBaseQAOA):
    """
    Implements entry processing with Generalized QAOA.
    :var space_type: Type of search space. 1 - All first order terms; 12 - All 1st and 2nd order terms; 12e - All 1st order and 2nd order only for existing edges.
    """
    search_space: str = field(init=False)
    space_type: str

    def __post_init__(self):
        self.search_space = 'general'
        if self.space_type != '1' and self.search_space != '12' and self.search_space != '12e':
            raise Exception('Space type has to be 1 or 12 or 12e for WorkerGeneral')

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        """
        Worker function for Generalized QAOA.
        :param entry: 1) Path to graph file; 2) Series of graph attributes.
        :return: Updated series with added expectation, angles and number of function evaluations for this p.
        """
        path, series = entry
        starting_point = None if self.initial_guess_from is None else numpy_str_to_array(series[self.initial_guess_from + '_angles'])
        graph = self.reader(path)

        target_vals = evaluate_graph_cut(graph)
        driver_term_vals = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 1)])

        if self.space_type != '1':
            if self.space_type == '12':
                driver_term_vals_2 = np.array([evaluate_z_term(np.array(term), len(graph)) for term in it.combinations(range(len(graph)), 2)])
            if self.space_type == '12e':
                driver_term_vals_2 = np.array([evaluate_z_term(edge, len(graph)) for edge in get_index_edge_list(graph)])
            driver_term_vals = np.append(driver_term_vals, driver_term_vals_2, axis=0)

        evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, self.p)
        result = optimize_qaoa_angles(evaluator, starting_angles=starting_point)

        series[self.out_col] = -result.fun / graph.graph['maxcut']
        series[self.out_col + '_angles'] = result.x
        series[self.out_col + '_nfev'] = result.nfev
        return series


class WorkerGeneralSub(WorkerGeneral):
    """ Implements entry processing with Generalized QAOA on subgraphs. """

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        starting_point = None if self.initial_guess_from is None else numpy_str_to_array(series[self.initial_guess_from + '_angles'])
        graph = self.reader(path)

        target_terms = [set(edge) for edge in get_index_edge_list(graph)]
        target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]

        driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
        if self.space_type == '12':
            driver_terms += [set(term) for term in it.combinations(range(len(graph)), 2)]
        if self.space_type == '12e':
            driver_terms += [set(term) for term in get_index_edge_list(graph)]

        evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, self.p)
        result = optimize_qaoa_angles(evaluator, starting_angles=starting_point)

        series[self.out_col] = -result.fun / graph.graph['maxcut']
        series[self.out_col + '_angles'] = result.x
        series[self.out_col + '_nfev'] = result.nfev
        return series


class WorkerStandard(WorkerBaseQAOA):
    """ Implements standard processing with plain or random starting angles. """

    def process_entry_core(self, path: str, search_space: str = None, **optimize_args) -> tuple:
        """
        Processes entry with plain input and output arguments.
        :param path: Path to graph file.
        :param search_space: Custom search space or None to use self.search_space.
        :param optimize_args: Additional arguments for optimization function.
        :return: 1) Approximation ratio; 2) Corresponding angles; 3) Number of function evaluations.
        """
        if search_space is None:
            search_space = self.search_space

        graph = self.reader(path)
        evaluator = Evaluator.get_evaluator_standard_maxcut(graph, self.p, search_space=search_space)
        method = 'COBYLA'
        options = {'maxiter': np.iinfo(np.int32).max}

        try:
            result = optimize_qaoa_angles(evaluator, **optimize_args, method=method, options=options)
        except Exception:
            raise Exception(f'Optimization failed at {path}')
        return -result.fun / graph.graph['maxcut'], str(result.x), result.nfev

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        starting_angles = None if self.initial_guess_from is None else numpy_str_to_array(series[self.initial_guess_from + '_angles'])
        ar, angles, nfev = self.process_entry_core(path, starting_angles)
        series[self.out_col] = ar
        series[self.out_col + '_angles'] = angles
        series[self.out_col + '_nfev'] = nfev
        return series


class WorkerLinear(WorkerStandard):
    """ Worker that implements linear angle initialization strategies (linear and tqa). """
    def __post_init__(self):
        if self.search_space != 'linear' and self.search_space != 'tqa':
            raise Exception('Search space must be linear or tqa for WorkerLinear')

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        _, linear_angles, total_nfev = WorkerStandard.process_entry_core(self, path, starting_angles=None)

        if self.search_space == 'tqa':
            qaoa_angles = convert_angles_tqa_to_qaoa(linear_angles, self.p)
        elif self.search_space == 'linear':
            qaoa_angles = convert_angles_linear_to_qaoa(linear_angles, self.p)

        ar, angles, nfev = WorkerStandard.process_entry_core(self, path, 'qaoa', starting_angles=qaoa_angles)
        total_nfev += nfev

        series[self.out_col] = ar
        series[self.out_col + '_angles'] = angles
        series[self.out_col + '_nfev'] = nfev
        return series


@dataclass(kw_only=True)
class WorkerIterativePerturb(WorkerStandard):
    """ Worker that implements the common functionality for iterative generation of the initial guess for next p with perturbations.
    The angle extension function has to be implemented by a child. """
    alpha: float

    def __post_init__(self):
        if self.p < 2:
            raise Exception('p has to be > 1 for iterative workers')

    def extend_angles(self, angles: ndarray) -> ndarray:
        """
        Extends given angles to generate the guess for the next layer. Must be defined by a child.
        :param angles: Set of angles.
        :return: Extended set of angles that serves as a guess for the next layer.
        """
        raise Exception('This method is not implemented in the base class. Use the derived classes.')

    def process_entry_core(self, path: str, angles_unperturbed: ndarray, angles_best: ndarray) -> tuple:
        """
        Core functionality of process_entry with plain input and output arguments instead of a series.
        :param path: Path to the graph file.
        :param angles_unperturbed: Best unperturbed angles from the previous layer.
        :param angles_best: Best overall angles from the previous layer.
        :return: 1) Best AR; 2) Best angles; 3) Total number of function evaluations; 4) Optimized unperturbed angles for this layer.
        """
        normalize_angles = self.search_space != 'fourier'
        perturbations_start = 1 if all(angles_unperturbed == angles_best) else 2
        optimization_results = []
        for i in range(self.p):
            starting_angles = angles_unperturbed if i == 0 else angles_best
            if i >= perturbations_start:
                perturbation = self.alpha * random.normal(scale=abs(starting_angles))
                starting_angles += perturbation
            starting_angles = self.extend_angles(starting_angles)
            result = WorkerStandard.process_entry_core(self, path, starting_angles=starting_angles, normalize_angles=normalize_angles)
            optimization_results.append(result)

        df = DataFrame(optimization_results)
        best_ind = np.argmax(df.iloc[:, 0])
        return df.iloc[best_ind, 0], df.iloc[best_ind, 1], sum(df.iloc[:, 2]), df.iloc[0, 1]

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        angles_unperturbed = numpy_str_to_array(series[self.initial_guess_from + '_angles_unperturbed'])
        angles_best = numpy_str_to_array(series[self.initial_guess_from + '_angles_best'])

        if self.search_space == 'fourier':
            angles_unperturbed = convert_angles_qaoa_to_fourier(angles_unperturbed)
            angles_best = convert_angles_qaoa_to_fourier(angles_best)
        ar_best, angles_best, total_nfev, new_angles_unperturbed = self.process_entry_core(path, angles_unperturbed, angles_best)
        if self.search_space == 'fourier':
            new_angles_unperturbed = normalize_qaoa_angles(convert_angles_fourier_to_qaoa(new_angles_unperturbed))
            angles_best = normalize_qaoa_angles(convert_angles_fourier_to_qaoa(angles_best))

        series[self.out_col] = ar_best
        series[self.out_col + '_angles_best'] = angles_best
        series[self.out_col + '_nfev'] = total_nfev
        series[self.out_col + '_angles_unperturbed'] = new_angles_unperturbed
        return series

    def postprocess_dataframe(self, dataframe: DataFrame) -> DataFrame:
        if self.transfer_from is not None:
            angle_suffixes = ['_angles_unperturbed', '_angles_best']
            dataframe = transfer_expectation_columns(dataframe, self.transfer_from, self.out_col, angle_suffixes, self.transfer_p, self.p, True)
        return dataframe


@dataclass(kw_only=True)
class WorkerInterp(WorkerIterativePerturb):
    """ Worker for Interp initialization strategy for QAOA. """
    search_space: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.search_space = 'qaoa'

    def extend_angles(self, angles: ndarray) -> ndarray:
        return interp_qaoa_angles(angles, self.p - 1)


@dataclass(kw_only=True)
class WorkerFourier(WorkerIterativePerturb):
    """ Worker function for Fourier initialization strategy for QAOA. """
    search_space: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.search_space = 'fourier'

    def extend_angles(self, angles: ndarray) -> ndarray:
        return np.concatenate((angles, [0] * 2))


@dataclass(kw_only=True)
class WorkerGreedy(WorkerStandard):
    """ Worker that implements greedy strategy for QAOA (p + 1 optimizations from transition states). """
    search_space: str = field(init=False)

    def __post_init__(self):
        if self.p < 2:
            raise Exception('p has to be > 1 for WorkerGreedy')
        self.search_space = 'qaoa'

    def process_entry_core(self, path: str, starting_angles: ndarray) -> tuple:
        """
        Core functionality of process_entry with plain input and output arguments instead of a series.
        :param path: Path to the graph file.
        :param starting_angles: Starting angles for optimization.
        :return: 1) Best found AR; 2) Corresponding angles; 3) Total number of function evaluations.
        """
        best_ar = 0
        best_angles = None
        total_nfev = 0
        for insert_layer in range(self.p):
            next_transition_state = np.concatenate((starting_angles[:2 * insert_layer], [0] * 2, starting_angles[2 * insert_layer:]))
            next_ar, next_angles, nfev = WorkerStandard.process_entry_core(self, path, starting_angles=next_transition_state)
            total_nfev += nfev
            if best_ar < next_ar:
                best_ar = next_ar
                best_angles = next_angles
        return best_ar, best_angles, total_nfev

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        starting_angles = numpy_str_to_array(series[self.initial_guess_from + '_angles'])
        ar, angles, nfev = self.process_entry_core(path, starting_angles)
        series[self.out_col] = ar
        series[self.out_col + '_angles'] = angles
        series[self.out_col + '_nfev'] = nfev
        return series


@dataclass(kw_only=True)
class WorkerCombined(WorkerInterp, WorkerGreedy):
    """ Worker that tries multiple angle strategies for QAOA. """
    search_space: str = field(init=False)

    def __post_init__(self):
        self.search_space = 'qaoa'

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        angles_unperturbed = numpy_str_to_array(series[self.initial_guess_from + '_angles_unperturbed'])
        angles_best = numpy_str_to_array(series[self.initial_guess_from + '_angles_best'])
        optimization_results = [None] * 2

        optimization_results[0] = WorkerInterp.process_entry_core(self, path, angles_unperturbed, angles_best)
        optimization_results[1] = WorkerGreedy.process_entry_core(self, path, angles_best)

        df = DataFrame(optimization_results)
        best_ind = np.argmax(df.iloc[:, 0])
        series[self.out_col] = df.iloc[best_ind, 0]
        series[self.out_col + '_angles_best'] = df.iloc[best_ind, 1]
        series[self.out_col + '_nfev'] = sum(df.iloc[:, 2])
        series[self.out_col + '_angles_unperturbed'] = df.iloc[0, 3]
        return series


@dataclass(kw_only=True)
class WorkerMA(WorkerStandard):
    """
    Worker that executes MA-QAOA optimization (independent angles on each term).
    :var guess_format: Name of format of starting point (ma or qaoa).
    """
    search_space: str = field(init=False)
    guess_format: str

    def __post_init__(self):
        if self.guess_format != 'ma' and self.guess_format != 'qaoa':
            raise Exception('Guess format can be ma or qaoa')
        self.search_space = 'ma'

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        starting_angles = None if self.initial_guess_from is None else numpy_str_to_array(series[self.initial_guess_from + '_angles'])
        graph = self.reader(path)
        if self.guess_format == 'qaoa':
            if starting_angles is None:
                starting_angles = random.uniform(-np.pi, np.pi, 2 * self.p)
            starting_angles = convert_angles_qaoa_to_ma(starting_angles, len(graph.edges), len(graph))
        ar, angles, nfev = WorkerStandard.process_entry_core(self, path, starting_angles=starting_angles)
        series[self.out_col] = ar
        series[self.out_col + '_angles'] = angles
        series[self.out_col + '_nfev'] = nfev
        return series


class WorkerMaxCut(WorkerAbstract):
    """ Worker that evaluates maxcut by brute-force and writes it to the input file as graph property. """

    def process_entry(self, entry: tuple[str, Series]) -> Series:
        path, series = entry
        graph = self.reader(path)
        cut_vals = evaluate_graph_cut(graph)
        max_cut = int(max(cut_vals))
        graph.graph['maxcut'] = max_cut
        nx.write_gml(graph, path)
        return series


def optimize_expectation_parallel(dataframe_path: str, rows_func: callable, num_workers: int, worker: WorkerBaseQAOA):
    """
    Optimizes cut expectation for a given set of graphs in parallel and writes the output dataframe.
    :param dataframe_path: Path to input dataframe with information about jobs.
    :param rows_func: Function that accepts dataframe and returns boolean array identifying which rows of the dataframe should be considered.
    :param num_workers: Number of parallel workers.
    :param worker: Worker instance.
    """
    df = pd.read_csv(dataframe_path, index_col=0)
    selected_rows = rows_func(df)
    rows_to_process = list(df.loc[selected_rows, :].iterrows())
    remaining_rows = df.loc[~selected_rows, :]

    if len(rows_to_process) == 0:
        return

    results = []
    if num_workers == 1:
        for result in tqdm(map(worker.process_entry, rows_to_process), total=len(rows_to_process), smoothing=0, ascii=' █'):
            results.append(result)
    else:
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(worker.process_entry, rows_to_process), total=len(rows_to_process), smoothing=0, ascii=' █'):
                results.append(result)

    df = pd.concat((DataFrame(results), remaining_rows)).sort_index(key=natsort_keygen())
    df.index.name = 'path'
    if hasattr(worker, 'postprocess_dataframe'):
        df = worker.postprocess_dataframe(df)
    df.to_csv(dataframe_path)

    dataset_id = re.search(r'nodes_\d+/depth_\d+', dataframe_path)[0]
    print(f'dataset: {dataset_id}; p: {worker.p}; mean: {np.mean(df[worker.out_col]):.3f}; min: {min(df[worker.out_col]):.3f}; converged: {sum(df[worker.out_col] > 0.9995)}; '
          f'nfev: {np.mean(df[worker.out_col + "_nfev"]):.0f}\n')
