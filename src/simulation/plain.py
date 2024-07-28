"""
Functions that evaluate expectation values in QAOA directly through quantum simulation.
"""
import numpy as np
from networkx import Graph
from numba import njit
from numpy import ndarray, sin, cos

from src.graph_utils import get_index_edge_list
from src.preprocessing import PSubset, evaluate_edge_cut


call_counter = 0


@njit
def apply_phase(term_angles: ndarray, term_vals: ndarray, psi: ndarray) -> ndarray:
    """
    Applies the phase operator with given angles to a given state psi.
    :param term_angles: 1D array with the angles for each term.
    :param term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a term of phase operator for each computational basis.
    :param psi: Current quantum state vector.
    :return: New quantum state vector.
    """
    for i in range(len(term_angles)):
        psi = np.exp(-1j * term_angles[i] * term_vals[i, :]) * psi
    return psi


@njit
def get_exp_x(beta: float) -> ndarray:
    """
    :param beta: Rotation angle.
    :return: Matrix representation for exp(-i * beta * X).
    """
    return np.array([[cos(beta), -1j * sin(beta)],
                     [-1j * sin(beta), cos(beta)]])


@njit
def get_exp_y(alpha: float) -> ndarray:
    """
    :param alpha: Rotation angle.
    :return: Matrix representation for exp(-i * beta * Y).
    """
    return np.array([[cos(alpha), -sin(alpha)],
                     [sin(alpha), cos(alpha)]])


@njit
def apply_unitary_one_qubit(bit_ind: int, unitary: ndarray, psi: ndarray, num_bits: int, control_ind: int = None, control_val: int = None) -> ndarray:
    """
    Applies a given single qubit unitary matrix (2x2) to a specified qubit (target).
    :param bit_ind: Target bit index in big endian notation.
    :param unitary: Unitary matrix to apply.
    :param psi: Current quantum state vector.
    :param num_bits: Total number of bits.
    :param control_ind: Index of the control qubit in big endian notation.
    :param control_val: Value of the control necessary to apply the unitary (0 or 1).
    :return: New quantum state vector.
    """
    res = np.zeros(np.shape(psi), dtype=np.complex128)
    bit_ind_right = num_bits - bit_ind - 1
    if control_ind is not None:
        control_ind_right = num_bits - control_ind - 1

    for i in range(len(psi)):
        if control_ind is not None:
            i_control_val = i >> control_ind_right & 1
            if i_control_val != control_val:
                res[i] = psi[i]
                continue

        target_val = i >> bit_ind_right & 1
        neighbor = i + (-1) ** target_val * 2 ** bit_ind_right
        res[i] += psi[i] * unitary[target_val, target_val]  # basis remained the same
        res[neighbor] += psi[i] * unitary[1 - target_val, target_val]  # basis changed in specified bit
    return res


@njit
def apply_mixer_standard(betas: ndarray, psi: ndarray, apply_y: bool = False) -> ndarray:
    """
    Applies mixer unitary to a given state psi, which consists of a single layer of X (and potentially Y) gates applied to each qubit independently.
    :param betas: 1D array with rotation angles for each qubit, specified in the big endian order. Size: number of qubits.
    :param psi: Current quantum state vector.
    :param apply_y: True to apply a layer of Y-mixers with the same angles.
    :return: New quantum state vector.
    """
    for i in range(len(betas)):
        exp_x = get_exp_x(betas[i])
        psi = apply_unitary_one_qubit(i, exp_x, psi, len(betas))
        if apply_y:
            exp_y = get_exp_y(betas[i])
            psi = apply_unitary_one_qubit(i, exp_y, psi, len(betas))
    return psi


@njit
def apply_mixer_controlled(betas: ndarray, psi: ndarray) -> ndarray:
    """
    Applies controlled mixer to a given state psi. The mixer consists of layers of X-gates controlled on 0 and 1 by each qubit sequentially.
    :param betas: 1D array with rotation angles for each gate in the mixer, specified in the column-first order (on a quantum circuit). Size: 2 * #qubits ** 2.
    :param psi: Current quantum state vector. Size: 2 ** #qubits.
    :return: New quantum state vector. Size: 2 ** #qubits.
    """
    num_qubits = round(np.log2(len(psi)))
    gate_ind = 0
    for control_ind in range(num_qubits):
        for control_val in [0, 1]:
            for target_ind in range(num_qubits):
                if control_ind == target_ind:
                    continue
                exp_x = get_exp_x(betas[gate_ind])
                psi = apply_unitary_one_qubit(target_ind, exp_x, psi, num_qubits, control_ind, control_val)
                gate_ind += 1
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


def calc_expectation_per_edge(psi: ndarray, graph: Graph) -> list[float]:
    """
    Calculates cut expectation of each edge in a given state psi.
    :param psi: Quantum state vector.
    :param graph: Graph with edges.
    :return: A list of cut QAOA expectations in the order of graph.edges.
    """
    expectations = []
    for edge in get_index_edge_list(graph):
        edge_vals = evaluate_edge_cut(edge, len(graph))
        edge_expectation = calc_expectation_diagonal(psi, edge_vals)
        expectations.append(edge_expectation)
    return expectations


@njit
def construct_qaoa_state(angles: ndarray, driver_term_vals: ndarray, p: int, mixer_type: str = 'standard') -> ndarray:
    """
    Constructs QAOA state corresponding to the given angles and terms, assuming standard initial state.
    :param angles: 1D array of angles for all layers. Same format as in `calc_expectation_general_qaoa`.
    :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
    :param p: Number of QAOA layers.
    :param mixer_type: Type of mixer to use.
    :return: Resulting quantum state vector.
    """
    psi = np.ones(driver_term_vals.shape[1], dtype=np.complex128) / np.sqrt(driver_term_vals.shape[1])
    num_params_per_layer = len(angles) // p
    for i in range(p):
        layer_params = angles[i * num_params_per_layer:(i + 1) * num_params_per_layer]
        gammas = layer_params[:driver_term_vals.shape[0]]
        psi = apply_phase(gammas, driver_term_vals, psi)
        betas = layer_params[driver_term_vals.shape[0]:]

        if mixer_type == 'standard':
            psi = apply_mixer_standard(betas, psi)
        elif mixer_type == 'x+y':
            psi = apply_mixer_standard(betas, psi, True)
        elif mixer_type == 'controlled':
            psi = apply_mixer_controlled(betas, psi)
    return psi


def calc_expectation_general_qaoa(angles: ndarray, driver_term_vals: ndarray, p: int, target_vals: ndarray, mixer_type: str = 'standard') -> float:
    """
    Calculates target function expectation value for given set of driver terms and corresponding weights.
    :param angles: 1D array of angles for all layers. Format: first, term angles for 1st layer in the same order as rows of driver_term_vals,
    then mixer angles for 1st layer in the qubits order, then the same format repeats for other layers.
    :param driver_term_vals: 2D array of size #terms x 2 ** #qubits. Each row is an array of values of a driver function's term for each computational basis.
    :param p: Number of QAOA layers.
    :param target_vals: 1D array of target function values for all computational basis states.
    :param mixer_type: Type of mixer to use.
    :return: Expectation value of the target function in the state corresponding to the given parameters and terms.
    """
    global call_counter
    call_counter += 1

    psi = construct_qaoa_state(angles, driver_term_vals, p, mixer_type)
    expectation = calc_expectation_diagonal(psi, target_vals)
    return expectation


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
        subset_expectation = calc_expectation_general_qaoa(angles[subset.angle_map], subset.driver_term_vals, p, subset.target_vals)
        expectation += subset_coeffs[ind] * subset_expectation
    return expectation
