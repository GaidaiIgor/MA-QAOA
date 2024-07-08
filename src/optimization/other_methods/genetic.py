""" Functions for genetic optimization. """
import numpy as np
import pandas as pd

from numpy import random as random
from pygad import GA
from scipy.optimize import OptimizeResult


def genetic_func_wrapper(func):
    def wrapped(optimizer, solution, solution_ind):
        wrapped.counter += 1
        return -func(solution)
    wrapped.counter = 0
    return wrapped


def mutation_func(offspring, optimizer):
    for solution_ind in range(offspring.shape[0]):
        for angle_ind in range(offspring.shape[1]):
            mutated = random.uniform() <= optimizer.mutation_probability
            if mutated:
                mutation_amount = random.normal(scale=0.1)
                offspring[solution_ind, angle_ind] += mutation_amount
    return offspring


def on_generation(optimizer):
    if len(optimizer.best_solutions_fitness) >= 10 ** 5:
        raise Exception('Too many generations')
    if optimizer.fitness_func.counter >= optimizer.max_nfev:
        return 'stop'


def genetic_optimizer(fun, x0, args, **kwargs):
    p = kwargs['p']
    slsqp_data = pd.read_csv('graphs/main/nodes_9/depth_3/output/qaoa/constant/0.2/SLSQP/out.csv')
    path = kwargs['series']['path']
    graph_ind = int(path.split('/')[-1][:-4])
    max_nfev = slsqp_data.loc[graph_ind, f'p_{p}_nfev']

    num_generations = 10 ** 6
    initial_population_size = max(10, 4 * p)
    num_parents_mating = max(2, int(np.ceil(0.25 * initial_population_size)))
    fitness_func = genetic_func_wrapper(fun)
    initial_population = [x0] + [random.uniform(-np.pi / 2, np.pi / 2, len(x0)) for _ in range(initial_population_size - 1)]
    crossover_probability = 0.7
    mutation_probability = 0.25
    optimizer = GA(num_generations, num_parents_mating, fitness_func, initial_population=initial_population, mutation_type=mutation_func, mutation_probability=mutation_probability,
                   crossover_type='uniform', crossover_probability=crossover_probability, on_generation=on_generation,
                   save_best_solutions=True, suppress_warnings=True)
    optimizer.max_nfev = max_nfev
    optimizer.run()

    solution, fitness, ind = optimizer.best_solution()
    result = OptimizeResult()
    result.fun = -fitness
    result.x = solution
    result.success = True
    result.nfev = fitness_func.counter
    return result
