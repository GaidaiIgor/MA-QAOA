"""
Entry points for figure plotting.
"""
from typing import Sequence

import addcopyfighandler
import numpy as np
from matplotlib import pyplot as plt

from src.data_processing import exponential_form, polynomial_form, DataExtractor, fit_data, exponential_form_const, polynomial_form_const
from src.plot_general import colors, Line, plot_general, save_figure

assert addcopyfighandler, "Adds an option to copy figures by pressing Ctrl+C"


def generate_paths(ds_param_name: str, ds_param: Sequence, method: str) -> list:
    if ds_param_name == 'nodes':
        return [f'graphs/main/nodes_{node}/depth_3/output/{method}/out.csv' for node in ds_param]
    elif ds_param_name == 'depth':
        return [f'graphs/main/nodes_12/depth_{depth}/output/{method}/out.csv' for depth in ds_param]
    else:
        raise Exception('Unknown data set parameter')


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


def plot_ar_vs_p_core(ds_param_name: str, ds_param: Sequence, labels: list[str]):
    max_ps = [12, 5]
    methods = ['qaoa/constant/0.2', 'ma/qaoa_relax/constant']
    lines = []
    for method_ind, method in enumerate(methods):
        max_p = max_ps[method_ind]
        df_paths = generate_paths(ds_param_name, ds_param, method)
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
    labels = [f'{i} nodes' for i in nodes]
    plot_ar_vs_p_core('nodes', nodes, labels)
    save_figure()
    plt.show()


def plot_ar_vs_p_depths():
    depths = list(range(3, 7))
    labels = [f'Depth = {i}' for i in depths]
    plot_ar_vs_p_core('depth', depths, labels)
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


def plot_fit_general(n_extrap: int, method_name: str, ds_param_name: str, ds_param: Sequence, method: str, aggregator_func: callable, fit_ind_range: Sequence, labels: Sequence):
    ar_fit_funcs = [polynomial_form, exponential_form]
    coeff_fit_funcs = [polynomial_form_const, exponential_form_const]
    fit_labels = ['polynomial', 'exponential']
    ar_extrap = 16 / 17

    df_paths = generate_paths(ds_param_name, ds_param, method)
    fit_lines = []
    coeffs_p_all = np.zeros((len(df_paths), len(ar_fit_funcs), 2))
    rmse_all = np.zeros((len(df_paths), len(ar_fit_funcs)))
    for df_ind, next_path in enumerate(df_paths):
        extractor = DataExtractor(next_path)
        ps = np.array(extractor.get_ps())
        ars = extractor.get_ar_aggregated(aggregator_func)
        fit_lines.append(Line(ps, ars, style='none', color=df_ind, label=labels[df_ind]))

        for ar_fit_ind, ar_fit_func in enumerate(ar_fit_funcs):
            fitted, coeffs_p_all[df_ind, ar_fit_ind, :], rmse_all[df_ind, ar_fit_ind] = fit_data(ps, 1 - ars, ar_fit_func, fit_ind_range, p0=(0.5, 0.5), bounds=(0, np.inf))
            fit_lines.append(Line(ps, 1 - fitted, marker='none', color=df_ind, style=ar_fit_ind))

    plot_general(fit_lines, ('p', 'AR'), (1, 0.02), (0.75, 12.25, 0.68, 1.005))
    plt.axhline(1, c='k', ls='--')
    plt.axhline(16 / 17, c='r', ls='--')
    save_figure(f'ar_vs_p_{ds_param_name}_fit_{method_name}')

    rmse_lines = [Line(ds_param, rmse_all[:, i], color=i, label=fit_labels[i]) for i in range(rmse_all.shape[1])]
    plot_general(rmse_lines, (ds_param_name, 'RMSE'), (1, None), loc='best')
    save_figure(f'rmse_vs_{ds_param_name}_{method_name}')

    for ar_fit_ind in range(coeffs_p_all.shape[1]):
        coeffs_p_extraped = np.zeros((coeffs_p_all.shape[2], len(coeff_fit_funcs)))
        for coeff_ind in range(coeffs_p_all.shape[2]):
            next_coeffs = coeffs_p_all[:, ar_fit_ind, coeff_ind]
            print(f'AR fit: {fit_labels[ar_fit_ind]}; Coeff ind: {coeff_ind}; Coeff vals vs N: {next_coeffs}')
            fit_lines = [Line(ds_param, next_coeffs, style='none')]
            for coeff_fit_ind, coeff_fit_func in enumerate(coeff_fit_funcs):
                fitted, coeffs_n, rmse = fit_data(ds_param, next_coeffs, coeff_fit_func, p0=(0.2, 0.1, 0.1), maxfev=10000, bounds=(0, np.inf))
                fit_lines.append(Line(ds_param, fitted, marker='none', style=coeff_fit_ind))

                print(f'AR fit: {fit_labels[ar_fit_ind]}; Coeff ind: {coeff_ind}; Coeff fit: {fit_labels[coeff_fit_ind]}; Coeffs: {coeffs_n}; RMSE: {rmse}')
                coeffs_p_extraped[coeff_ind, coeff_fit_ind] = coeff_fit_func(n_extrap, *coeffs_n)

            plot_general(fit_lines, (ds_param_name, f'$c_{coeff_ind}$'), (1, None))
            save_figure(f'c{coeff_ind}_vs_{ds_param_name}_fit_{fit_labels[ar_fit_ind]}_{method_name}')
        print(f'RMSE: {rmse_all[:, ar_fit_ind]}')

        for coeff_fit_c0_ind in range(len(coeff_fit_funcs)):
            for coeff_fit_c1_ind in range(len(coeff_fit_funcs)):
                if ar_fit_funcs[ar_fit_ind] == polynomial_form:
                    extraped_p = int(np.ceil((coeffs_p_extraped[0, coeff_fit_c0_ind] / (1 - ar_extrap)) ** (1 / coeffs_p_extraped[1, coeff_fit_c1_ind])))
                elif ar_fit_funcs[ar_fit_ind] == exponential_form:
                    extraped_p = int(np.ceil(-np.log10((1 - ar_extrap) / coeffs_p_extraped[0, coeff_fit_c0_ind]) / coeffs_p_extraped[1, coeff_fit_c1_ind]))
                print(f'AR fit: {fit_labels[ar_fit_ind]}; Coeff fit 0: {fit_labels[coeff_fit_c0_ind]}; Coeff fit 1: {fit_labels[coeff_fit_c1_ind]} Predicted p: {extraped_p}')


def plot_ar_vs_p_fitted_nodes():
    nodes = range(9, 13)
    depths = range(3, 7)
    method = 'qaoa/constant/0.2'
    method = 'ma/qaoa_relax/constant'
    aggregator = np.min
    fit_ind_range = [0, 12]
    n_extrap = 60
    labels = [f'{i} nodes' for i in nodes]
    plot_fit_general(n_extrap, 'qaoa_average', 'nodes', nodes, method, aggregator, fit_ind_range, labels)
    save_figure()
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
    plot_ar_vs_p_fitted_nodes()
