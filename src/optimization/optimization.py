""" Functions related to optimization of QAOA angles. """
from __future__ import annotations

import logging
import time

import numpy as np
import numpy.random as random
from numpy import ndarray
from scipy import optimize
from scipy.optimize import OptimizeResult

from src.data_processing import normalize_qaoa_angles
from src.optimization.evaluator import Evaluator, change_sign


def optimize_qaoa_angles(evaluator: Evaluator, starting_angles: ndarray = None, method: str = 'L-BFGS-B', try_nelder_mead: bool = None, options: dict = None, num_restarts: int = 1,
                         objective_max: float = None, objective_tolerance: float = 0.9995, normalize_angles: bool = True, **kwargs) -> OptimizeResult:
    """
    Maximizes evaluator function (MaxCut expectation).
    :param evaluator: Evaluator instance.
    :param starting_angles: Starting point for optimization. Chosen randomly if None.
    :param method: Optimization method.
    :param try_nelder_mead: True to automatically attempt Nelder-Mead optimization method if the main method fails. True by default if the optimization method is not Nelder-Mead.
    :param num_restarts: Number of random starting points to try. Has no effect if specific starting point is provided.
    :param options: Optimization options. Maxint number of iterations will be used by default, if not specified.
    :param objective_max: Maximum achievable objective. Optimization stops if answer sufficiently close to max_objective is achieved.
    :param objective_tolerance: Fraction of 1 that controls how close the result need to be to objective_max before optimization can be stopped.
    :param normalize_angles: True to return optimized angles to the [-pi; pi] range.
    :param kwargs: Keyword arguments for optimizer.
    :return: Maximization result.
    """
    if starting_angles is not None:
        num_restarts = 1
        if len(starting_angles) != evaluator.num_angles:
            raise Exception('Number of starting angles does not match the number expected by evaluator')

    if try_nelder_mead is None:
        try_nelder_mead = method != 'Nelder-Mead'

    if options is None:
        maxint = np.iinfo(np.int32).max
        options = {'maxiter': maxint, 'maxfun': maxint}

    logger = logging.getLogger('QAOA')
    logger.debug('Optimization...')
    time_start = time.perf_counter()

    result_best = None
    for i in range(num_restarts):
        next_angles = random.uniform(-np.pi, np.pi, evaluator.num_angles) if starting_angles is None else starting_angles
        result = optimize.minimize(change_sign(evaluator.func), next_angles, method=method, options=options, **kwargs)
        if not result.success:
            logger.warning(f'Optimization with {method} failed with the following message:')
            logger.warning(result.message)
            if try_nelder_mead:
                logger.warning('Switching to Nelder-Mead')
                result = optimize.minimize(change_sign(evaluator.func), next_angles, method='Nelder-Mead', options=options, **kwargs)
                if not result.success:
                    logger.error('Optimization with Nelder-Mead failed')
                    logger.error(result)
            if not result.success:
                raise Exception('Optimization failed')
        result.fun *= -1

        if normalize_angles:
            result.x = normalize_qaoa_angles(result.x)

        if result_best is None or result.fun > result_best.fun:
            result_best = result

        if objective_max is not None and result_best.fun / objective_max > objective_tolerance:
            break

    time_finish = time.perf_counter()
    logger.debug(f'Optimization done. Time elapsed: {time_finish - time_start}')
    return result_best
