import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.optimization.evaluator import Evaluator


@dataclass(kw_only=True)
class SpaceDimensionProviderBase(ABC):
    """ Base class that determines dimensionality of a search space. """

    @abstractmethod
    def get_number_of_dimensions(self, evaluator: Evaluator) -> int:
        """
        Returns the number of dimensions in the search space based on target evaluator.
        :param evaluator: Target evaluator.
        :return: Number of dimensions.
        """
        pass


@dataclass(kw_only=True)
class SpaceDimensionProviderAbsolute(SpaceDimensionProviderBase):
    """
    Provides absolute number of dimensions given to it.
    :var num_dims: Absolute number of dimensions.
    """
    num_dims: int

    def get_number_of_dimensions(self, evaluator: Evaluator) -> int:
        return self.num_dims


@dataclass(kw_only=True)
class SpaceDimensionProviderRelative(SpaceDimensionProviderBase):
    """
    Provides the number of dimension equal to the desired fraction of the total number of parameters of the given evaluator rounded up.
    :var param_fraction: Desired fraction of parameters.
    """
    param_fraction: float

    def __post_init__(self):
        assert 0 < self.param_fraction <= 1, 'param_fraction has to be between 0 and 1'

    def get_number_of_dimensions(self, evaluator: Evaluator) -> int:
        return math.ceil(self.param_fraction * evaluator.num_angles)
