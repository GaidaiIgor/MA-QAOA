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
    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> ndarray:
        """
        Provides initial guess in the specified guess format.
        :param evaluator:
        :param series:
        :return:
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

    def provide_guess(self, evaluator: Evaluator, series: Series = None) -> ndarray:
        """
        Provides guess appropriate for the given evaluator based on the data in series.
        :param evaluator: Evaluator for which guess is generated.
        :param series: Series with data about evaluator's job (may be needed for some providers).
        :return: Initial guess for optimization.
        """
        initial_guess = self.provide_initial_guess(evaluator, series)
        initial_guess = self.convert_angles_format(initial_guess, evaluator)
        return initial_guess


@dataclass(kw_only=True)
class GuessProviderRandom(GuessProviderBase):
    """ Provides random guess. """

    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> ndarray:
        initial_angles = random.uniform(-np.pi, np.pi, evaluator.num_angles)
        return initial_angles


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

    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> ndarray:
        gammas = [self.const_val] * evaluator.p
        betas = [-self.const_val] * evaluator.p
        initial_angles = np.array(list(it.chain(*zip(gammas, betas))))
        return initial_angles


@dataclass(kw_only=True)
class GuessProviderSeries(GuessProviderBase):
    """
    Provides initial angles by reading them from specified record in the series.
    :var guess_from: Name of the column from where the corresponding angles will be taken as initial guess.
    """
    guess_from: str

    def provide_initial_guess(self, evaluator: Evaluator, series: Series = None) -> ndarray:
        return numpy_str_to_array(series[self.guess_from])
