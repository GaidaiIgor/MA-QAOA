"""
Functions that evaluate expectation values in QAOA directly through quantum simulation
"""
import math

import numpy as np
import scipy.linalg as linalg
from numba import njit
from numpy import ndarray, sin, cos


def apply_uc(all_cuv_vals: ndarray, gammas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies Uc unitary to a given state psi
    :param all_cuv_vals: 2D array where each row is a diagonal of Cuv operator for each edge in the graph
    :param gammas: 1D array with the values of gamma for each edge
    :param psi: Current quantum state vector
    :return: New quantum state vector
    """
    return math.prod([np.exp(-1j * gammas[i] * all_cuv_vals[i, :]) for i in range(len(gammas))]) * psi


def apply_ub_explicit(betas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies Ub unitary to a given state psi. Explicitly creates Ub matrix as a sum of tensor products.
    :param betas: 1D array with rotation angles for each qubit
    :param psi: Current quantum state vector
    :return: New quantum state vector
    """
    pauli_i = np.array([[1, 0], [0, 1]])
    pauli_x = np.array([[0, 1], [1, 0]])
    ub = 1
    for i in range(len(betas)):
        next_tensor = 1
        for j in range(len(betas)):
            if i == j:
                next_tensor = np.kron(pauli_x, next_tensor)
            else:
                next_tensor = np.kron(pauli_i, next_tensor)
        ub *= linalg.expm(-1j * betas[i] * next_tensor)
    return np.matmul(ub, psi)


@njit
def get_exp_x(beta: float) -> ndarray:
    """
    Returns Ub matrix for 1 qubit
    :param beta: Rotation angle
    :return: Ub matrix for 1 qubit, i.e. exp(-i*beta*X)
    """
    return np.array([[cos(beta), -1j * sin(beta)],
                     [-1j * sin(beta), cos(beta)]])


@njit
def apply_unitary_one_qubit(unitary: ndarray, psi: ndarray, target_neighbours: ndarray, target_vals: ndarray) -> ndarray:
    """
    Applies a given single qubit unitary matrix (2x2) to a specified qubit (target).
    :param unitary: Unitary matrix to apply
    :param psi: Current quantum state vector
    :param target_neighbours: 1D array where i-th element is different from i in the target bit (i.e. a column from get_neighbour_labelings)
    :param target_vals: 1D array with the values of the target bit in each basis label. Can be obtained as a column from get_all_binary_labelings.
    :return: New quantum state vector
    """
    res = np.zeros(np.shape(psi), dtype=np.complex128)
    for i in range(len(psi)):
        res[i] += psi[i] * unitary[target_vals[i], target_vals[i]]  # basis remained the same
        res[target_neighbours[i]] += psi[i] * unitary[1 - target_vals[i], target_vals[i]]  # basis changed in specified bit
    return res


@njit
def apply_ub_individual(betas: ndarray, psi: ndarray, neighbours: ndarray, basis_bin: ndarray) -> ndarray:
    """
    Applies Ub unitary to a given state psi. Does not explicitly create Ub matrix. Instead, applies single-qubit unitaries to each qubit independently.
    :param betas: 1D array with rotation angles for each qubit
    :param psi: Current quantum state vector
    :param neighbours: Structure calculated by get_neighbour_labelings
    :param basis_bin: Structure calculated by get_all_binary_labelings
    :return: New quantum state vector
    """
    num_qubits = neighbours.shape[1]
    res = psi
    for i in range(num_qubits):
        exp_x = get_exp_x(betas[i])
        res = apply_unitary_one_qubit(exp_x, res, neighbours[:, i], basis_bin[:, i])
    return res


def calc_expectation_diagonal(psi: ndarray, diagonal_vals: ndarray) -> float:
    """
    Calculates expectation value of a given diagonal operator for a given state psi
    :param psi: Quantum state vector
    :param diagonal_vals: Values of a diagonal operator
    :return: Expectation value of a given operator in the given state
    """
    return np.real(np.vdot(psi, diagonal_vals * psi))


def run_ma_qaoa_simulation(angles: ndarray, p: int, all_cuv_vals: ndarray, neighbours: ndarray, basis_bin: ndarray, edge_inds: list[int] = None) -> float:
    """
    Runs MA-QAOA by direct simulation of quantum evolution. Dumb and slow, but easy to understand and does not require any additional knowledge.
    :param angles: 1D array of all angles for all layers. Format: First, all gammas for 1st layer (in the edge order),
    then all betas for 1st layer (in the nodes order), then the same format repeats for all other layers.
    :param p: Number of QAOA layers
    :param all_cuv_vals: 2D array where each row is a diagonal of Cuv operator for each edge in the graph. Size: num_edges x 2^num_nodes
    :param neighbours: Structure calculated by get_neighbour_labelings
    :param basis_bin: Structure calculated by get_all_binary_labelings
    :param edge_inds: Indices of edges that should be taken into account when calculating expectation value. If None, then all edges are taken into account.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>
    """
    if edge_inds is None:
        edge_inds = list(range(all_cuv_vals.shape[0]))

    psi = np.ones(all_cuv_vals.shape[1], dtype=np.complex128) / np.sqrt(all_cuv_vals.shape[1])
    num_angles_per_layer = int(len(angles) / p)
    for i in range(p):
        layer_angles = angles[i * num_angles_per_layer:(i + 1) * num_angles_per_layer]
        gammas = layer_angles[:all_cuv_vals.shape[0]]
        psi = apply_uc(all_cuv_vals, gammas, psi)
        betas = layer_angles[all_cuv_vals.shape[0]:]
        psi = apply_ub_individual(betas, psi, neighbours, basis_bin)
    return calc_expectation_diagonal(psi, np.sum(all_cuv_vals[edge_inds, :], 0))
