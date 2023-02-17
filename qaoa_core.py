from typing import Callable, Any

import networkx as nx
import numpy as np
import scipy.linalg as linalg
from numba import njit
from numpy import ndarray, sin, cos
from networkx import Graph
import math


def apply_uc(all_objective_vals: ndarray, gamma: float, psi: ndarray) -> ndarray:
    # Applies Uc unitary to a given state psi
    return np.exp(-1j * gamma * all_objective_vals) * psi


def apply_ub_explicit(beta: float, psi: ndarray) -> ndarray:
    # Applies Ub unitary to a given state psi. Explicitly creates Ub matrix as a sum of tensor products.
    pauli_i = np.array([[1, 0], [0, 1]])
    pauli_x = np.array([[0, 1], [1, 0]])
    num_qubits = int(np.log2(len(psi)))
    ub = 0
    for i in range(num_qubits):
        next_tensor = 1
        for j in range(num_qubits):
            if i == j:
                next_tensor = np.kron(pauli_x, next_tensor)
            else:
                next_tensor = np.kron(pauli_i, next_tensor)
        ub = ub + next_tensor
    ub = linalg.expm(-1j * beta * ub)
    return np.matmul(ub, psi)


@njit
def get_exp_x(beta: float) -> ndarray:
    # Returns Ub matrix for 1 qubit, i.e. exp(-i*beta*X)
    return np.array([[cos(beta), -1j * sin(beta)],
                     [-1j * sin(beta), cos(beta)]])


@njit
def apply_unitary_one_qubit(unitary: ndarray, psi: ndarray, target_neighbours: ndarray, target_vals: ndarray) -> ndarray:
    # Applies a given single qubit unitary matrix (2x2) to a specified qubit (target)
    # target_neighbours is a 1D array where target_neighbours[i] = k s.t. k is different from i in the target bit (i.e. a column from get_neighbour_labelings)
    # target_vals is a 1D array with the values of the target bit in each basis label. Can be obtained as a column from get_all_binary_labelings.
    res = np.zeros(np.shape(psi), dtype=np.complex128)
    for i in range(len(psi)):
        res[i] += psi[i] * unitary[target_vals[i], target_vals[i]]  # basis remained the same
        res[target_neighbours[i]] += psi[i] * unitary[1 - target_vals[i], target_vals[i]]  # basis changed in specified bit
    return res


@njit
def apply_ub_individual(beta: float, psi: ndarray, neighbours: ndarray, basis_bin: ndarray) -> ndarray:
    # Applies Ub unitary to a given state psi. Does not explicitly create Ub matrix. Instead, applies single-qubit unitaries to each qubit independently.
    # Requires neighbours and basis_bin structures (for numba), which can be calculated by get_neighbour_labelings and get_all_binary_labelings functions.
    num_qubits = neighbours.shape[1]
    exp_x = get_exp_x(beta)
    res = psi
    for i in range(num_qubits):
        res = apply_unitary_one_qubit(exp_x, res, neighbours[:, i], basis_bin[:, i])
    return res


def calc_expectation_diagonal(psi: ndarray, diagonal_vals: ndarray) -> float:
    # Calculates expectation value of a given diagonal operator for a given state psi
    return np.real(np.vdot(psi, diagonal_vals * psi))


def run_qaoa_simulation(angles: ndarray, p: int, all_objective_vals: ndarray, all_cuv_vals: ndarray, neighbours: ndarray = None, basis_bin: ndarray = None) \
        -> float:
    # Runs QAOA by direct simulation of quantum evolution. Dumb and slow, but easy to understand and always works.
    # pairs_mat can optionally be supplied for a faster version of ub application function, and is assumed to be calculated via calc_pair_list
    # Returns expectation value for an edge (u, v), defined by all_cuv_vals diagonal operator
    # Initial state is a superposition of basis states
    psi = np.ones(len(all_objective_vals), dtype=np.complex128) / np.sqrt(len(all_objective_vals))
    betas = angles[0:p]
    gammas = angles[p:2 * p]
    for i in range(p):
        psi = apply_uc(all_objective_vals, gammas[i], psi)
        if neighbours is None or basis_bin is None:
            psi = apply_ub_explicit(betas[i], psi)
        else:
            psi = apply_ub_individual(betas[i], psi, neighbours, basis_bin)
    return calc_expectation_diagonal(psi, all_cuv_vals)


def run_qaoa_analytical_p1(angles: ndarray, deg1: int, deg2: int, num_triangles: int) -> float:
    # Runs QAOA by evaluating an analytical formula for <Cuv> when p=1
    # deg1 and deg2 are degrees of vertices u and v, and num_triangles is the number of triangles containing edge (u, v)
    # The formula is taken from Wang, Z., Hadfield, S., Jiang, Z. & Rieffel, E. G.
    # Quantum approximate optimization algorithm for MaxCut: A fermionic view. Phys. Rev. A97(2), 022304 (2018)
    beta = angles[0]
    gamma = angles[1]
    d = deg1 - 1
    e = deg2 - 1
    f = num_triangles
    return 0.5 + 0.25 * sin(4 * beta) * sin(gamma) * (cos(gamma) ** d + cos(gamma) ** e) - \
        0.25 * sin(2 * beta) ** 2 * cos(gamma) ** (d + e - f) * (1 - cos(2 * gamma) ** f)


def run_ma_qaoa_analytical_p1(angles: ndarray, graph: Graph) -> float:
    betas = angles[0:len(graph)]
    gammas = angles[len(graph):]
    nx.set_edge_attributes(graph, {(u, v): gammas[i] * w for i, (u, v, w) in enumerate(graph.edges.data('weight'))}, name='gamma')
    objective = 0
    for u, v, w in graph.edges.data('weight'):
        cuv = w / 2
        d = set(graph[u]) - {v}
        e = set(graph[v]) - {u}
        f = d & e
        cos_prod_d = math.prod([cos(graph.edges[u, m]['gamma']) for m in d - f])
        cos_prod_e = math.prod([cos(graph.edges[v, m]['gamma']) for m in e - f])

        # Triangle terms
        if len(f) != 0:
            cos_prod_f_plus = math.prod([cos(graph.edges[u, m]['gamma'] + graph.edges[v, m]['gamma']) for m in f])
            cos_prod_f_minus = math.prod([cos(graph.edges[u, m]['gamma'] - graph.edges[v, m]['gamma']) for m in f])
            cuv += w / 4 * sin(2 * betas[u]) * sin(2 * betas[v]) * cos_prod_d * cos_prod_e * (cos_prod_f_plus - cos_prod_f_minus)
            cos_prod_d *= math.prod([cos(graph.edges[u, m]['gamma']) for m in f])
            cos_prod_e *= math.prod([cos(graph.edges[v, m]['gamma']) for m in f])

        cuv += w / 2 * sin(graph.edges[u, v]['gamma']) * \
            (sin(2 * betas[u]) * cos(2 * betas[v]) * cos_prod_d + cos(2 * betas[u]) * sin(2 * betas[v]) * cos_prod_e)
        objective += cuv

    return objective


def change_sign(func: Callable[[Any, ...], int | float]) -> Callable[[Any, ...], int | float]:
    # Decorator to change sign of the return value of a given function. Useful to carry out maximization instead of minimization.
    def func_changed_sign(*args, **kwargs):
        return -func(*args, **kwargs)

    return func_changed_sign