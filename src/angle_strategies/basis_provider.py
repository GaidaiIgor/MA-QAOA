from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import ndarray, linalg

from src.optimization.optimization import Evaluator


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


class BasisProviderBase(ABC):
    """ Base class for search space basis providers. """

    @abstractmethod
    def provide_basis(self, evaluator_ma: Evaluator) -> ndarray:
        """
        Provides basis for the subspace of a given MA evaluator according to the current class's strategy.
        :param evaluator_ma: Full-dimensional MA evaluator for a given graph.
        :return: Basis as a 2D row array.
        """
        pass

    @staticmethod
    def augment_basis(basis: ndarray, num_dims: int) -> ndarray:
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
    """
    Generates random basis for search space.
    :var num_dims: Desired number of basis vectors.
    """
    num_dims: int

    def provide_basis(self, evaluator_ma: Evaluator) -> ndarray:
        basis = np.empty((0, evaluator_ma.num_angles))
        basis = BasisProviderBase.augment_basis(basis, self.num_dims)
        return basis
