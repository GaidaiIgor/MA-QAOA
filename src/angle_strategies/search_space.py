""" Module that implements custom search spaces and related functionality. """
from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass
class SearchSpace:
    """
    Class that represents custom search subspaces within MA-QAOA.
    :var basis: 2D array of custom basis set vectors (in rows) expressed over the original MA-QAOA search space. Must be orthonormal.
    :var shift: 1D array that shifts the origin the new basis.
    """
    basis: ndarray
    shift: ndarray

    def __post_init__(self):
        orthonormality_check = np.matmul(self.basis, self.basis.T)
        identity = np.identity(self.basis.shape[0])
        assert np.allclose(orthonormality_check, identity), 'Basis has to be orthonormal'

    def transform_coordinates(self, coordinates: ndarray) -> ndarray:
        """
        Transforms coordinates in this search space to the coordinates in the original space.
        :param coordinates: 1D array of coordinates in the current search space. Length = self.basis.shape[0].
        :return: 1D array of the corresponding coordinates in the original space. Length = self.basis.shape[1].
        """
        return self.shift + np.matmul(coordinates, self.basis)

    def apply_interface(self, ma_qaoa_func: callable) -> callable:
        """
        Converts interface of a given function that expects MA-QAOA angles to angles in the current search space.
        :param ma_qaoa_func: A function that expects MA-QAOA angles.
        :return: Updated function that accepts angles in the current search space.
        """
        def search_space_wrapped(*args, **kwargs):
            original_coordinates = self.transform_coordinates(args[0])
            return ma_qaoa_func(original_coordinates, *args[1:], **kwargs)
        return search_space_wrapped
