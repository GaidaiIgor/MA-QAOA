""" Module for initial guess providers. """
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import itertools as it
from numpy import ndarray
from numpy import random
from pandas import Series

from src.angle_strategies.direct import convert_angles_tqa_to_qaoa, convert_angles_linear_to_qaoa, convert_angles_qaoa_to_ma
from src.data_processing import numpy_str_to_array
from src.optimization.optimization import Evaluator


@dataclass(kw_only=True)
class GuessProviderBase(ABC):
    """
    Base class for guess providers.
    :var format: Format of the provided guess.
    """
    format: str

    @abstractmethod
    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> tuple[ndarray, int]:
        """
        Provides initial guess in the specified guess format.
        :param evaluator: Evaluator for which the guess is generated.
        :param series: Series with data about evaluator's job (may be needed for some providers).
        :return: 1) Initial guess for optimization. 2) Number of QPU calls to obtain this guess.
        """
        pass

    def convert_angles_format(self, angles: ndarray, evaluator: Evaluator) -> ndarray:
        """
        Converts angle format to make it compatible with the given evaluator.
        :param angles: Input angles in self.format.
        :param evaluator: Target evaluator for compatibility.
        :return: Evaluator-compatible angles.
        """
        if self.format != evaluator.search_space:
            if evaluator.search_space == 'qaoa':
                if self.format == 'tqa':
                    angles = convert_angles_tqa_to_qaoa(angles, evaluator.p)
                elif self.format == 'linear':
                    angles = convert_angles_linear_to_qaoa(angles, evaluator.p)
                else:
                    raise Exception('Unknown angle conversion')
            elif evaluator.search_space == 'ma':
                if self.format == 'qaoa':
                    angles = convert_angles_qaoa_to_ma(angles, evaluator.num_driver_terms, evaluator.num_qubits)
                else:
                    raise Exception('Unknown angle conversion')
            else:
                raise Exception('Unknown angle conversion')
        return angles

    def provide_guess(self, evaluator: Evaluator, series: Series = None) -> tuple[ndarray, int]:
        """
        Provides guess appropriate for the given evaluator based on the data in series.
        :param evaluator: Evaluator for which guess is generated.
        :param series: Series with data about evaluator's job (may be needed for some providers).
        :return: 1) Initial guess for optimization. 2) Number of QPU calls to obtain this guess.
        """
        initial_guess, nfev = self.provide_initial_guess(evaluator, series)
        initial_guess = self.convert_angles_format(initial_guess, evaluator)
        return initial_guess, nfev


@dataclass(kw_only=True)
class GuessProviderRandom(GuessProviderBase):
    """ Provides random guess. """

    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> tuple[ndarray, int]:
        initial_angles = random.uniform(-np.pi, np.pi, evaluator.num_angles)
        return initial_angles, 0


@dataclass(kw_only=True)
class GuessProviderConstant(GuessProviderBase):
    """
    Provides constant guess independent of input series.
    :var const_val: Value of the constant for the guess.
    """
    format: str = 'qaoa'
    const_val: float = 0.2

    def __post_init__(self):
        assert self.format == 'qaoa', 'format has to be qaoa for GuessProviderConstant'

    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> tuple[ndarray, int]:
        gammas = [self.const_val] * evaluator.p
        betas = [-self.const_val] * evaluator.p
        initial_angles = np.array(list(it.chain(*zip(gammas, betas))))
        return initial_angles, 0


@dataclass(kw_only=True)
class GuessProviderSeries(GuessProviderBase):
    """
    Provides initial angles by reading them from specified record in the series.
    :var guess_from: Name of the column from where the corresponding angles will be taken as initial guess.
    :var cost_from: Name of the column from where the cost of evaluating this guess is taken.
    """
    guess_from: str
    cost_from: str = None

    def provide_initial_guess(self, evaluator: Evaluator, series: Series) -> tuple[ndarray, int]:
        nfev = int(series[self.cost_from]) if self.cost_from is not None else 0
        return numpy_str_to_array(series[self.guess_from]), nfev
