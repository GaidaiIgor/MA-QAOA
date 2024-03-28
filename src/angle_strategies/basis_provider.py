from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import ndarray, linalg

from src.angle_strategies.guess_provider import GuessProviderBase
from src.angle_strategies.space_dimension_provider import SpaceDimensionProviderBase
from src.optimization.optimization import Evaluator, optimize_qaoa_angles


def gram_schmidt(vectors: ndarray) -> ndarray:
    """
    Orthonormalizes given set of linearly independent vectors.
    :param vectors: Input set of vectors (in rows).
    :return: Orthonormalized set.
    """
    ortho_vectors = np.copy(vectors)
    for i in range(ortho_vectors.shape[0]):
        for j in range(i):
            ortho_vectors[i, :] -= np.dot(ortho_vectors[i, :], ortho_vectors[j, :]) * ortho_vectors[j, :]
        if linalg.norm(ortho_vectors[i, :]) < 1e-10:
            raise Exception(f'Vectors are linearly dependent at iteration {i}')
        ortho_vectors[i, :] /= linalg.norm(ortho_vectors[i, :])
    return ortho_vectors


@dataclass(kw_only=True)
class BasisProviderBase(ABC):
    """
    Base class for search space basis providers.
    :var dimension_provider: Object that determines the dimensionality of the resulting basis.
    """
    dimension_provider: SpaceDimensionProviderBase

    @abstractmethod
    def provide_initial_basis(self, evaluator_ma: Evaluator) -> tuple[ndarray, int]:
        """
        Provides initial (incomplete) basis for the subspace of a given MA evaluator according to the current class's strategy.
        :param evaluator_ma: Full-dimensional MA evaluator for a given graph.
        :return: 1) Basis as a 2D row array; 2) Number of evaluator calls that was made to construct it.
        """
        pass

    def provide_basis(self, evaluator_ma: Evaluator) -> tuple[ndarray, int]:
        """
        Provides basis for the subspace of a given MA evaluator according to the current class's strategy.
        :param evaluator_ma: Full-dimensional MA evaluator for a given graph.
        :return: 1) Basis as a 2D row array; 2) Number of evaluator calls that was made to construct it.
        """
        basis, nfev = self.provide_initial_basis(evaluator_ma)
        full_num_dims = self.dimension_provider.get_number_of_dimensions(evaluator_ma)
        if full_num_dims > basis.shape[0]:
            basis = self.augment_basis_random(basis, full_num_dims - basis.shape[0])
        return basis, nfev

    @staticmethod
    def augment_basis_random(basis: ndarray, num_dims: int) -> ndarray:
        """
        Augments basis by adding num_dim random orthonormal vectors to a given basis. Uses Gram-Schmidt orthogonalization procedure.
        :param basis: Existing basis.
        :param num_dims: Desired number of extra dimensions.
        :return: Basis augmented with random new dimensions.
        """
        max_attempts = 10
        assert basis.shape[1] - basis.shape[0] >= num_dims, 'Desired augmentation exceeds maximum space dimensions'
        for i in range(max_attempts):
            random_vectors = np.random.uniform(-1, 1, (num_dims, basis.shape[1]))
            try:
                new_basis = gram_schmidt(np.concatenate((basis, random_vectors)))
                break
            except Exception:
                pass
        else:
            raise Exception('Could not generate basis within maximum number of attempts.')
        return new_basis


@dataclass(kw_only=True)
class BasisProviderRandom(BasisProviderBase):
    """ Generates random basis for search space. """

    def provide_initial_basis(self, evaluator_ma: Evaluator) -> tuple[ndarray, int]:
        return np.empty((0, evaluator_ma.num_angles)), 0


@dataclass(kw_only=True)
class BasisProviderGradient(BasisProviderBase):
    """
    Generates basis where the first vector is chosen as MA's gradient at the initial point given by the provider. Other vectors are random.
    :var gradient_point_provider: Object that provides the point where gradient is evaluated.
    """
    gradient_point_provider: GuessProviderBase

    def provide_initial_basis(self, evaluator_ma: Evaluator) -> tuple[ndarray, int]:
        gradient_point = self.gradient_point_provider.provide_guess(evaluator_ma)
        result = optimize_qaoa_angles(evaluator_ma, gradient_point, check_success=False, options={'maxiter': 1})
        gradient = result.x - gradient_point
        gradient /= linalg.norm(gradient)
        basis = gradient.reshape((1, -1))
        return basis, result.nfev
