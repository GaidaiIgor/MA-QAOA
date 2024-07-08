""" Functions for Bayesian optimizations. """
from itertools import count

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, UtilityFunction
from numpy import random as random
from scipy.optimize import OptimizeResult


def check_stop(func_vals):
    best_val = np.max(func_vals)
    best_iter = np.where((best_val - func_vals) / best_val < 1e-3)[0][0]
    margin = max(best_iter * 0.2, 10)
    if len(func_vals) >= best_iter + margin + 1:
        return True
    return False


def convert_dict_to_vector(dictionary):
    vector = np.zeros(len(dictionary))
    for key in dictionary:
        vector[int(key[1:])] = dictionary[key]
    return vector


def convert_vector_to_dict(vector):
    dictionary = {f'p{i}': vector[i] for i in range(len(vector))}
    return dictionary


def bayesian_func_wrapper(func):
    def wrapped(**kwargs):
        vector = convert_dict_to_vector(kwargs)
        return -func(vector)
    return wrapped


def bayesian_optimizer(fun, x0, args, **kwargs):
    bounds = {f'p{key}': (-np.pi / 2, np.pi / 2) for key in range(len(x0))}
    fun_wrapped = bayesian_func_wrapper(fun)
    optimizer = BayesianOptimization(None, bounds)
    utility = UtilityFunction()

    num_initial_points = 4 * kwargs['p']
    points = []
    vals = []
    next_point = convert_vector_to_dict(x0)
    for i in count():
        points.append(next_point)
        vals.append(fun_wrapped(**points[-1]))
        if i >= num_initial_points + max(10, np.ceil(0.2 * num_initial_points)):
            if check_stop(vals):
                break
        optimizer.register(points[-1], vals[-1])
        if i < num_initial_points - 1:
            next_point = convert_vector_to_dict(random.uniform(-np.pi / 2, np.pi / 2, 2 * kwargs['p']))
        else:
            next_point = optimizer.suggest(utility)

    best_iter = np.argmax(vals)
    result = OptimizeResult()
    result.fun = -vals[best_iter]
    result.x = convert_dict_to_vector(points[best_iter])
    result.success = True
    result.nfev = len(vals)
    return result


def bayesian_optimizer_2(fun, x0, args, **kwargs):
    bounds = {f'p{key}': (-np.pi / 2, np.pi / 2) for key in range(len(x0))}
    fun_wrapped = bayesian_func_wrapper(fun)
    optimizer = BayesianOptimization(fun_wrapped, bounds, verbose=0)

    slsqp_data = pd.read_csv('graphs/main/nodes_9/depth_3/output/qaoa/random/attempts_1/SLSQP/out.csv')
    path = kwargs['series']['path']
    graph_ind = int(path.split('/')[-1][:-4])
    p = kwargs['p']
    num_iter = slsqp_data.loc[graph_ind, f'p_{p}_nfev']
    optimizer.probe(x0.tolist())
    optimizer.maximize(4 * p, num_iter - 4 * p - 1)
    result = OptimizeResult()
    result.fun = -optimizer.max['target']
    result.x = convert_dict_to_vector(optimizer.max['params'])
    result.success = True
    result.nfev = num_iter
    return result
