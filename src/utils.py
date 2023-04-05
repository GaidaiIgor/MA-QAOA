"""
General miscellaneous functions, implemented for compatibility with numba.
"""
from collections.abc import Iterator

import numpy as np
from numba import njit
from numpy import ndarray


@njit
def get_all_combinations(array: ndarray) -> Iterator[ndarray]:
    """
    Returns all combinations of any length from elements of array.
    :param array: Array with elements.
    :return: Generator of combinations.
    """
    for i in range(len(array) + 1):
        fixed_length_combos = get_fixed_length_combinations(array, i)
        for combo in fixed_length_combos:
            yield combo


@njit
def get_fixed_length_combinations(array: ndarray, combo_length: int) -> Iterator[ndarray]:
    """
    Returns all combinations of fixed length from elements of array.
    :param array: Array with elements.
    :param combo_length: Length of combinations.
    :return: Generator of combinations.
    """
    inds = np.array(list(range(combo_length)))
    reversed_inds = inds.copy()[::-1]

    while True:
        yield array[inds]
        for i in reversed_inds:
            if inds[i] != len(array) + i - combo_length:
                break
        else:
            return

        inds[i] += 1
        for j in range(i + 1, combo_length):
            inds[j] = inds[j - 1] + 1
