""" Module that implements custom search spaces and related functionality. """
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import itertools as it

from numpy import ndarray


@dataclass
class SearchSpace(ABC):
    """ Base class that represents custom search space. """

    @abstractmethod
    def get_num_angles(self, *args) -> int:
        """
        Returns the number of angles in this search space.
        :param args: Instance-dependent arguments.
        :return: Number of angles in this search space.
        """
        pass

    @abstractmethod
    def apply_interface(self, ma_func: callable, *args) -> callable:
        """
        Adapts full-dimensional MA function to work with the angles describing this search space.
        :param ma_func: Full-dimensional MA function.
        :param args: Instance-dependent arguments.
        :return: Adapted function.
        """
        pass


@dataclass
class SearchSpaceControlled(SearchSpace):
    """ Class that represents search space of controlled (MA-)QAOA. """

    def get_num_angles(self, num_phase_terms: int, num_qubits: int, p: int) -> int:
        """
        Returns the number of adjustable variables for a given QAOA instance.
        :param num_phase_terms: Number of terms in the phase operator.
        :param num_qubits: Number of qubits.
        :param p: Number of QAOA layers.
        :return: Number of adjustable variables.
        """
        num_angles_mixer = num_qubits ** 2
        return (num_phase_terms + num_angles_mixer) * p

    def transform_coordinates(self, coordinates: ndarray, num_phase_terms: int, num_qubits: int, p: int) -> ndarray:
        """
        Transforms coordinates in this search space to the coordinates in the original space.
        :param coordinates: 1D array of coordinates in the current search space.
        :param num_phase_terms: Number of terms in the phase operator.
        :param num_qubits: Number of qubits.
        :param p: Number of QAOA layers.
        :return: 1D array of the corresponding coordinates in the original space.
        """
        return coordinates

    def apply_interface(self, ma_func: callable, num_phase_terms: int, num_qubits: int, p: int) -> callable:
        """
        Converts interface of a given function that expects full controlled MA-QAOA angles to angles restricted to the current search space.
        :param ma_func: Controlled MA-QAOA function.
        :param num_phase_terms: Number of terms in the phase operator.
        :param num_qubits: Number of qubits.
        :param p: Number of QAOA layers.
        :return: Updated function that accepts angles restricted to the current search space.
        """
        def interface_wrapped(*args, **kwargs):
            original_coordinates = self.transform_coordinates(args[0], num_phase_terms, num_qubits, p)
            return ma_func(original_coordinates, *args[1:], **kwargs)
        return interface_wrapped


@dataclass
class SearchSpaceGeneral(SearchSpace):
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

    def get_num_angles(self, *args) -> int:
        return self.basis.shape[0]

    def transform_coordinates(self, coordinates: ndarray) -> ndarray:
        """
        Transforms coordinates in this search space to the coordinates in the original space.
        :param coordinates: 1D array of coordinates in the current search space. Length = self.basis.shape[0].
        :return: 1D array of the corresponding coordinates in the original space. Length = self.basis.shape[1].
        """
        return self.shift + np.matmul(coordinates, self.basis)

    def apply_interface(self, ma_qaoa_func: callable, *args) -> callable:
        def search_space_wrapped(*args, **kwargs):
            original_coordinates = self.transform_coordinates(args[0])
            return ma_qaoa_func(original_coordinates, *args[1:], **kwargs)
        return search_space_wrapped
