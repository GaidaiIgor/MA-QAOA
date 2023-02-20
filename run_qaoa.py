import time

import scipy.optimize as optimize

from preprocessing import *
from qaoa_core import *


def run_qaoa_main():
    # TODO: simulation MA-QAOA
    # TODO: graphs db interface
    # TODO: choice between single edge or whole graph expectation
    multi_angle = False
    p = 1  # Number of QAOA layers
    optimization_attempts = 10
    # Analytical expression parameters
    use_analytical = True

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(0, 2, weight=1)
    graph.add_edge(0, 4, weight=1)
    graph.add_edge(1, 3, weight=1)
    graph.add_edge(1, 5, weight=1)
    graph.add_edge(2, 3, weight=1)
    graph.add_edge(2, 5, weight=1)
    graph.add_edge(3, 4, weight=1)
    graph.add_edge(4, 5, weight=1)

    assert not multi_angle or use_analytical, "Only analytical mode exists for MA-QAOA"
    assert not use_analytical or p == 1, "Cannot use analytical for p != 1"

    if not use_analytical:
        print('Preprocessing...')
        time_start = time.perf_counter()
        neighbours = get_neighbour_labelings(len(graph))
        all_labelings = get_all_binary_labelings(len(graph))
        all_objective_vals = np.array([calc_cut_weight_total(labeling, graph) for labeling in all_labelings])
        all_cuv_vals = np.array([check_edge_cut(labeling, 0, 1) for labeling in all_labelings])
        time_finish = time.perf_counter()
        print(f'Preprocessing done. Time elapsed: {time_finish - time_start}')

    print('Optimization...')
    time_start = time.perf_counter()
    num_angles_per_layer = len(graph) + len(graph.edges) if multi_angle else 2
    angles_best = np.zeros(num_angles_per_layer * p)
    objective_max = sum([w for u, v, w in graph.edges.data('weight')])
    objective_best = 0

    for opt_ind in range(optimization_attempts):
        if objective_max - objective_best < 1e-3:
            break

        next_angles = np.random.uniform(-np.pi, np.pi, len(angles_best))
        if use_analytical:
            if multi_angle:
                result = optimize.minimize(change_sign(run_ma_qaoa_analytical_p1), next_angles, (graph, ))
            else:
                result = optimize.minimize(change_sign(run_qaoa_analytical_p1), next_angles, (graph, ))
        else:
            result = optimize.minimize(change_sign(run_qaoa_simulation), next_angles, (p, all_objective_vals, all_cuv_vals, neighbours, all_labelings))

        if -result.fun > objective_best:
            objective_best = -result.fun
            angles_best = next_angles / np.pi
    time_finish = time.perf_counter()

    print(f'Opt ind: {opt_ind}')
    print(f'Optimization done. Runtime: {time_finish - time_start}')
    print(f'Objective achieved: {objective_best}')


if __name__ == '__main__':
    run_qaoa_main()
    # cProfile.run('main()')
