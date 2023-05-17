"""
Functions that evaluate expectation values in QAOA directly through quantum simulation
"""
import numpy as np
import scipy.linalg as linalg
from numba import njit
from numpy import ndarray, sin, cos

from src.preprocessing import PSubgraph


@njit
def apply_uc(all_cut_vals: ndarray, gammas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies Uc unitary to a given state psi.
    :param all_cut_vals: 2D array where each row is a diagonal of Cuv operator for each edge in the graph.
    :param gammas: 1D array with the values of gamma for each edge.
    :param psi: Current quantum state vector.
    :return: New quantum state vector.
    """
    for i in range(len(gammas)):
        psi = np.exp(-1j * gammas[i] * all_cut_vals[i, :]) * psi
    return psi


@njit
def get_exp_x(beta: float) -> ndarray:
    """
    Returns Ub matrix for 1 qubit.
    :param beta: Rotation angle.
    :return: Ub matrix for 1 qubit, i.e. exp(-i*beta*X).
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
def apply_ub_individual(betas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies Ub unitary to a given state psi. Does not explicitly create Ub matrix. Instead, applies single-qubit unitaries to each qubit independently.
    :param betas: 1D array with rotation angles for each qubit. Size: number of qubits.
    :param psi: Current quantum state vector.
    :return: New quantum state vector.
    """
    for i in range(len(betas)):
        exp_x = get_exp_x(betas[i])
        psi = apply_unitary_one_qubit(exp_x, psi, i, len(betas))
    return psi


def calc_expectation_diagonal(psi: ndarray, diagonal_vals: ndarray) -> float:
    """
    Calculates expectation value of a given diagonal operator for a given state psi.
    :param psi: Quantum state vector.
    :param diagonal_vals: Values of a diagonal operator.
    :return: Expectation value of a given operator in the given state.
    """
    return np.real(np.vdot(psi, diagonal_vals * psi))


def calc_expectation_ma_qaoa_simulation_subgraphs(angles: ndarray, p: int, subgraphs: list[PSubgraph]) -> float:
    """
    Calculates objective expectation for given angles with MA-QAOA ansatz by separate simulation of each p-subgraph.
    :param angles: 1D array of all angles for all layers. Format: Angles are specified in the order of application, i.e.
    all gammas for 1st layer (in the edge order), then all betas for 1st layer (in the nodes order), then the same format repeats for all other layers.
    :param p: Number of QAOA layers.
    :param subgraphs: List of p-subgraphs corresponding to each edge in the graph.
    :return: Expectation value of target function in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>.
    """
    num_angles_per_layer = int(len(angles) / p)
    expectation = 0
    for subgraph in subgraphs:
        subgraph_angles = []
        for i in range(p):
            subgraph_angles.extend(angles[subgraph.angle_map + i * num_angles_per_layer])
        subgraph_angles = np.array(subgraph_angles)
        expectation += calc_expectation_ma_qaoa_simulation(subgraph_angles, p, subgraph.cut_vals, subgraph.edge_inds)
    return expectation


def calc_expectation_ma_qaoa_simulation(angles: ndarray, p: int, all_cut_vals: ndarray, edge_inds: ndarray = None) -> float:
    """
    Calculates objective expectation for given angles with MA-QAOA ansatz by simulation of quantum evolution.
    :param angles: 1D array of all angles for all layers. Format: Angles are specified in the order of application, i.e.
    all gammas for 1st layer (in the edge order), then all betas for 1st layer (in the nodes order), then the same format repeats for all other layers.
    :param p: Number of QAOA layers.
    :param all_cut_vals: 2D array where each row is a diagonal of Cuv operator for each edge in the graph. Size: num_edges x 2^num_nodes.
    :param edge_inds: 1D array of indices of edges that should be considered when calculating expectation value. If None, then all edges are considered.
    :return: Expectation value of C (sum of all Cuv) in the state corresponding to the given set of angles, i.e. <beta, gamma|C|beta, gamma>.
    """
    if edge_inds is None:
        edge_inds = list(range(all_cut_vals.shape[0]))

    psi = np.ones(all_cut_vals.shape[1], dtype=np.complex128) / np.sqrt(all_cut_vals.shape[1])
    num_angles_per_layer = int(len(angles) / p)
    for i in range(p):
        layer_angles = angles[i * num_angles_per_layer:(i + 1) * num_angles_per_layer]
        gammas = layer_angles[:all_cut_vals.shape[0]]
        psi = apply_uc(all_cut_vals, gammas, psi)
        betas = layer_angles[all_cut_vals.shape[0]:]
        psi = apply_ub_individual(betas, psi)
    return calc_expectation_diagonal(psi, np.sum(all_cut_vals[edge_inds, :], 0))


def apply_ub_explicit(betas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies Ub unitary to a given state psi. Explicitly creates Ub matrix as a sum of tensor products.
    :param betas: 1D array with rotation angles for each qubit.
    :param psi: Current quantum state vector.
    :return: New quantum state vector.
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
