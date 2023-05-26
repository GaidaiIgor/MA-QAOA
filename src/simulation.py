"""
Functions that evaluate expectation values in QAOA directly through quantum simulation.
"""
import numpy as np
from numba import njit
from numpy import ndarray, sin, cos

from src.preprocessing import PSubset


@njit
def apply_driver(term_angles: ndarray, term_vals: ndarray, psi: ndarray) -> ndarray:
    """
    Applies exponent of the driver function with given angles to a given state psi.
    :param term_angles: 1D array with the angles for each term.
    :param term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a term in Z-expansion of the driver function for each computational basis.
    :param psi: Current quantum state vector.
    :return: New quantum state vector.
    """
    for i in range(len(term_angles)):
        psi = np.exp(-1j * term_angles[i] * term_vals[i, :]) * psi
    return psi


@njit
def get_exp_x(beta: float) -> ndarray:
    """
    Returns mixer matrix for 1 qubit.
    :param beta: Rotation angle.
    :return: Mixer matrix for 1 qubit, i.e. exp(-i*beta*X).
    """
    return np.array([[cos(beta), -1j * sin(beta)],
                     [-1j * sin(beta), cos(beta)]])


@njit
def apply_unitary_one_qubit(unitary: ndarray, psi: ndarray, bit_ind: int, num_bits: int) -> ndarray:
    """
    Applies a given single qubit unitary matrix (2x2) to a specified qubit (target).
    :param unitary: Unitary matrix to apply.
    :param psi: Current quantum state vector.
    :param bit_ind: Target bit index in big endian notation.
    :param num_bits: Total number of bits.
    :return: New quantum state vector.
    """
    res = np.zeros(np.shape(psi), dtype=np.complex128)
    bit_ind_right = num_bits - bit_ind - 1
    for i in range(len(psi)):
        target_val = i >> bit_ind_right & 1
        neighbour = i + (-1) ** target_val * 2 ** bit_ind_right
        res[i] += psi[i] * unitary[target_val, target_val]  # basis remained the same
        res[neighbour] += psi[i] * unitary[1 - target_val, target_val]  # basis changed in specified bit
    return res


@njit
def apply_mixer_individual(betas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies mixer unitary to a given state psi. Does not explicitly create the mixer matrix. Instead, applies single-qubit unitaries to each qubit independently.
    :param betas: 1D array with rotation angles for each qubit. Size: number of qubits.
    :param psi: Current quantum state vector.
    :return: New quantum state vector.
    """
    for i in range(len(betas)):
        exp_x = get_exp_x(betas[i])
        psi = apply_unitary_one_qubit(exp_x, psi, i, len(betas))
    return psi


@njit
def calc_expectation_diagonal(psi: ndarray, diagonal_vals: ndarray) -> float:
    """
    Calculates expectation value of a given diagonal operator for a given state psi.
    :param psi: Quantum state vector.
    :param diagonal_vals: Values of a diagonal operator.
    :return: Expectation value of a given operator in the given state.
    """
    return np.real(np.vdot(psi, diagonal_vals * psi))


@njit
def calc_expectation_general_qaoa(angles: ndarray, target_vals: ndarray, driver_term_vals: ndarray, p: int) -> float:
    """
    Calculates target function expectation value for given set of driver terms and corresponding weights.
    :param angles: 1D array of angles for all layers. Format: first, term angles for 1st layer in the same order as rows of driver_term_vals,
    then mixer angles for 1st layer in the qubits order, then the same format repeats for other layers.
    :param target_vals: 1D array of target function values for all computational basis states.
    :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
    :param p: Number of QAOA layers.
    :return: Expectation value of the target function in the state corresponding to the given parameters and terms.
    """
    psi = np.ones(driver_term_vals.shape[1], dtype=np.complex128) / np.sqrt(driver_term_vals.shape[1])
    num_params_per_layer = len(angles) // p
    for i in range(p):
        layer_params = angles[i * num_params_per_layer:(i + 1) * num_params_per_layer]
        gammas = layer_params[:driver_term_vals.shape[0]]
        psi = apply_driver(gammas, driver_term_vals, psi)
        betas = layer_params[driver_term_vals.shape[0]:]
        psi = apply_mixer_individual(betas, psi)
    return calc_expectation_diagonal(psi, target_vals)


def calc_expectation_general_qaoa_subsets(angles: ndarray, subsets: list[PSubset], subset_coeffs: list[float], p: int) -> float:
    """
    Calculates objective expectation for given angles with generalized QAOA ansatz by separate simulation of each p-subset.
    :param angles: 1D array of all angles for all layers. Format: Angles are specified in the order of application, i.e.
    all gammas for 1st layer (in the term order), then all betas for 1st layer (in the qubits order), then the same format repeats for all other layers.
    :param subsets: List of p-subsets corresponding to each term of the target function.
    :param subset_coeffs: List of size len(subsets) + 1 with multiplier coefficients for each subset given in the same order. The last extra coefficient is a shift.
    :param p: Number of QAOA layers.
    :return: Expectation value of target function in the state corresponding to the given set of angles and driver terms.
    """
    expectation = subset_coeffs[-1]
    for ind, subset in enumerate(subsets):
        subset_expectation = calc_expectation_general_qaoa(angles[subset.angle_map], subset.target_vals, subset.driver_term_vals, p)
        expectation += subset_coeffs[ind] * subset_expectation
    return expectation
