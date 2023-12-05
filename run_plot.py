"""
Entry points for figure plotting.
"""

import addcopyfighandler
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

from src.data_processing import exponential_form, DataExtractor, linear_function
from src.plot_general import colors, Line, plot_general, save_figure
from itertools import product

assert addcopyfighandler, "Adds an option to copy figures by pressing Ctrl+C"


def generate_paths(nodes: list, depths: list, method: str) -> list:
    all_combos = product(nodes, depths)
    return [f'graphs/new/nodes_{pair[0]}/depth_{pair[1]}/output/{method}/out.csv' for pair in all_combos]


def plot_ar_vs_p_heuristics_core(methods: list[str], labels: list[str], min_y: float, max_p: int):
    lines = []
    for method_ind, method in enumerate(methods):
        extractor = DataExtractor(f'graphs/main/nodes_9/depth_3/output/{method}/out.csv')
        ps = extractor.get_ps()
        ar_mean = extractor.get_ar_aggregated(np.mean)
        ar_min = extractor.get_ar_aggregated(min)
        lines.append(Line(ps[:max_p], ar_mean[:max_p], colors[method_ind], style='-', label=labels[method_ind]))
        lines.append(Line(ps[:max_p], ar_min[:max_p], colors[method_ind], style='--'))

    x_lim = [0.75, max_p + 0.25]
    plot_general(lines, ('p', 'AR'), (1, 0.02), (*x_lim, min_y, 1.005))
    plt.legend(loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [16/17, 16/17], 'r--')


def plot_ar_vs_p_heuristics_qaoa_attempts_1():
    methods = ['constant/0.2', 'tqa/attempts_1', 'interp/attempts_1', 'fourier/attempts_1', 'random/attempts_1']
    methods = ['qaoa/' + method for method in methods]
    labels = ['Constant', 'TQA', 'Interp', 'Fourier', 'Random']
    plot_ar_vs_p_heuristics_core(methods, labels, 0.52, 8)
    save_figure()
    plt.show()


def plot_ar_vs_p_heuristics_qaoa_attempts_p():
    methods = ['greedy', 'tqa', 'interp', 'fourier', 'random']
    methods = ['qaoa/' + method + '/attempts_p' for method in methods]
    labels = ['Greedy', 'TQA', 'Interp', 'Fourier', 'Random']
    plot_ar_vs_p_heuristics_core(methods, labels, 0.52, 8)
    save_figure()
    plt.show()


def plot_ar_vs_p_heuristics_ma_attempts_1():
    methods = ['constant/0.2', 'interp/attempts_1', 'qaoa_relax/constant', 'random_qaoa/attempts_1', 'random/attempts_1']
    methods = ['ma/' + method for method in methods]
    labels = ['Constant', 'Interp', 'QAOA Relax', 'Random QAOA', 'Random']
    plot_ar_vs_p_heuristics_core(methods, labels, 0.72, 3)
    save_figure()
    plt.show()


def plot_ar_vs_p_core(nodes: list[int], depths: list[int], labels: list[str]):
    max_ps = [12, 5]
    methods = ['qaoa/constant/0.2', 'ma/qaoa_relax/constant']
    lines = []
    for method_ind, method in enumerate(methods):
        max_p = max_ps[method_ind]
        df_paths = generate_paths(nodes, depths, method)
        for path_ind, path in enumerate(df_paths):
            extractor = DataExtractor(path)
            xs = extractor.get_ps()
            ys = extractor.get_ar_aggregated(np.mean)[:max_p]
            label = None if method_ind > 0 else labels[path_ind]
            lines.append(Line(xs, ys, colors[path_ind], style='-', marker=method_ind, label=label))
            ys = extractor.get_ar_aggregated(min)[:max_p]
            lines.append(Line(xs, ys, colors[path_ind], style='--', marker=method_ind))

    boundaries = (0.75, max_ps[0] + 0.25, 0.68, 1.0025)
    tick_multiples = (1, 0.02)
    plot_general(lines, ('p', 'AR'), tick_multiples, boundaries)
    plt.legend(loc='lower right', fontsize='small')

    conv_limit = 1  # 0.9995
    np_limit = 16/17
    plt.axhline(conv_limit, c='k', ls='--')
    plt.axhline(np_limit, c='r', ls='--')


def plot_ar_vs_p_nodes():
    nodes = list(range(9, 13))
    depths = [3]
    labels = [f'{i} nodes' for i in nodes]
    plot_ar_vs_p_core(nodes, depths, labels)
    save_figure()
    plt.show()


def plot_ar_vs_p_depths():
    nodes = [12]
    depths = list(range(3, 7))
    labels = [f'Depth = {i}' for i in depths]
    plot_ar_vs_p_core(nodes, depths, labels)
    save_figure()
    plt.show()


def plot_ar_vs_cost_core(nodes: list[int], depths: list[int], labels: list[str]):
    max_ps = [12, 5]
    methods = ['qaoa/constant/0.2', 'ma/qaoa_relax/constant']
    lines = []
    for method_ind, method in enumerate(methods):
        max_p = max_ps[method_ind]
        df_paths = generate_paths(nodes, depths, method)
        for path_ind, path in enumerate(df_paths):
            extractor = DataExtractor(path)
            xs = extractor.get_cost_average()[:max_p]
            ys = extractor.get_ar_aggregated(np.mean)[:max_p]
            label = None if method_ind > 0 else labels[path_ind]
            lines.append(Line(xs, ys, colors[path_ind], style='-', marker=method_ind, label=label))

            xs = extractor.get_cost_worst_case()[:max_p]
            ys = extractor.get_ar_aggregated(min)[:max_p]
            lines.append(Line(xs, ys, colors[path_ind], style='--', marker=method_ind))

    boundaries = (None, None, 0.68, 1.005)
    tick_multiples = (None, 0.02)

    plot_general(lines, ('Cost', 'AR'), tick_multiples, boundaries)
    plt.legend(loc='lower right', fontsize='small')

    conv_limit = 1  # 0.9995
    np_limit = 16 / 17
    # conv_limit = np.log10(1 - conv_limit)
    # np_limit = np.log10(1 - np_limit)

    plt.axhline(conv_limit, c='k', ls='--')
    plt.axhline(np_limit, c='r', ls='--')


def plot_ar_vs_cost_nodes():
    nodes = list(range(9, 13))
    depths = [3]
    labels = [f'{i} nodes' for i in nodes]
    plot_ar_vs_cost_core(nodes, depths, labels)
    save_figure()
    plt.show()


def plot_ar_vs_cost_depths():
    nodes = [12]
    depths = list(range(3, 7))
    labels = [f'Depth = {i}' for i in depths]
    plot_ar_vs_cost_core(nodes, depths, labels)
    save_figure()
    plt.show()


def plot_fit():
    method = 'qaoa/constant/0.2'
    # method = 'ma/qaoa_relax/constant'
    nodes = [9]
    depths = [3]
    fit_func = linear_function
    start_ind = 0
    df_paths = generate_paths(nodes, depths, method)

    lines = []
    for ind, next_path in enumerate(df_paths):
        extractor = DataExtractor(next_path)
        xs = np.array(extractor.get_ps())
        ys = np.log10(1 - extractor.get_ar_aggregated(np.mean))
        params, covariance = optimize.curve_fit(fit_func, xs[start_ind:], ys[start_ind:])
        fitted_data = fit_func(xs, *params)
        lines.append(Line(xs, fitted_data, marker='none', color=ind))
        lines.append(Line(xs, ys, marker='o', style='none', color=ind))

        rmse = np.sqrt(sum((fitted_data[start_ind:] - ys[start_ind:]) ** 2) / len(ys[start_ind:]))
        print(f'Coeffs: {params}')
        print(f'RMSE: {rmse}')

    plot_general(lines)
    plt.show()


# def plot_interp_random_ar_difference_vs_p_nodes():
#     nodes = range(9, 13)
#     lines = []
#     for n_ind, n in enumerate(nodes):
#         ps, ar_qaoa = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/ma/qaoa/out.csv', r'p_\d+$')
#         ps, ar_rand = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/ma/random/out_r1.csv', r'p_\d+$')
#         diff = ar_qaoa - ar_rand
#         lines.append(Line(ps, diff, colors[n_ind]))
#
#     plot_general(lines, ('p', 'Average AR difference'), (1, 0.002), (0.75, 5.25, None, None))
#     save_figure()
#     plt.show()


if __name__ == "__main__":
    # plot_ar_vs_p_heuristics_qaoa_attempts_1()
    # plot_ar_vs_p_heuristics_qaoa_attempts_p()
    # plot_ar_vs_p_heuristics_ma_attempts_1()
    # plot_ar_vs_p_nodes()
    # plot_ar_vs_p_depths()
    # plot_ar_vs_cost_nodes()
    # plot_ar_vs_cost_depths()
    plot_fit()
