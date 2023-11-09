"""
Contains plot functions.
"""
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator, MultipleLocator
from networkx import Graph

from src.analytical import calc_expectation_ma_qaoa_analytical_p1
from src.angle_strategies import qaoa_decorator
from src.data_processing import get_column_statistic, calculate_min_p, calculate_edge_diameter
from src.plot_general import colors, Line, plot_general, markers, save_figure

import addcopyfighandler
assert addcopyfighandler, "Adds an option to copy figures by pressing Ctrl+C"


def plot_qaoa_expectation_p1(graph: Graph, edge_list: list[tuple[int, int]] = None):
    """ Plots a 2D map of expectation values vs beta and gamma for QAOA and given graph. """
    beta = np.linspace(-np.pi, np.pi, 361)
    gamma = np.linspace(-np.pi, np.pi, 361)
    beta_mesh, gamma_mesh = np.meshgrid(beta, gamma)
    expectation = np.zeros_like(beta_mesh)
    qaoa_evaluator = qaoa_decorator(calc_expectation_ma_qaoa_analytical_p1, len(graph.edges), len(graph))
    for i in range(len(beta)):
        for j in range(len(beta)):
            angles = np.array([beta_mesh[i, j], gamma_mesh[i, j]])
            expectation[i, j] = qaoa_evaluator(angles, graph, edge_list)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(beta_mesh / np.pi, gamma_mesh / np.pi, expectation, cmap=cm.seismic, vmin=0.25, vmax=0.75, rstride=5, cstride=5)
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(elev=90, azim=-90, roll=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return surf


def plot_avg_ar_vs_p_r1_nodes():
    nodes = range(9, 13)
    methods = ['qaoa', 'ma']
    lines = []
    for method_ind, method in enumerate(methods):
        for n_ind, n in enumerate(nodes):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}/random/out_r1.csv', r'p_\d+$')
            lines.append(Line(ps, p_series, colors[n_ind], markers[method_ind]))
    plot_general(lines, ('p', 'Average AR'), (1, 0.02), (0.75, 10.25, None, 1.0025))
    plt.plot([0, 11], [1, 1], 'k--')
    plt.plot([0, 11], [0.99, 0.99], 'r--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_r1_edges():
    depths = range(3, 7)
    methods = ['qaoa', 'ma']
    lines = []
    for method_ind, method in enumerate(methods):
        for depth_ind, depth in enumerate(depths):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_12/depth_{depth}/output/{method}/random/out_r1.csv', r'p_\d+$')
            lines.append(Line(ps, p_series, colors[depth_ind], markers[method_ind]))
    plot_general(lines, ('p', 'Average AR'), (1, 0.02), (0.75, 10.25, None, 1.0025))
    plt.plot([0, 11], [1, 1], 'k--')
    plt.plot([0, 11], [0.99, 0.99], 'r--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_r1_interp_nodes():
    nodes = range(7, 11)
    methods = ['qaoa/interp/out.csv', 'ma/random/out_r1.csv', 'qaoa/random/out_r1.csv']
    lines = []
    markers = 'oXo'
    for method_ind, method in enumerate(methods):
        for color_ind, n in enumerate(nodes):
            extra = 'ed_4' if n == 8 else ''
            ps, p_series = get_column_statistic(f'graphs/nodes_{n}/{extra}/output/{method}', r'p_\d+$')
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(ps, p_series, colors[color_ind], markers[method_ind], line_style))
    plot_general(lines, ('p', 'Average AR'), (1, 0.02), (0.75, 10.25, None, 1.0025))
    plt.plot([0, 11], [1, 1], 'k--')
    plt.plot([0, 11], [0.99, 0.99], 'r--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_r1_interp_edges():
    eds = [3.5, 4, 4.5, 5]
    methods = ['qaoa/interp/out.csv', 'ma/random/out_r1.csv', 'qaoa/random/out_r1.csv']
    lines = []
    markers = 'oXo'
    for method_ind, method in enumerate(methods):
        for color_ind, ed in enumerate(eds):
            ps, p_series = get_column_statistic(f'graphs/nodes_8/ed_{ed:.2g}/output/{method}', r'p_\d+$')
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(ps, p_series, colors[color_ind], markers[method_ind], line_style))
    plot_general(lines, ('p', 'Average AR'), (1, 0.02), (0.75, 10.25, None, 1.0025))
    plt.plot([0, 11], [1, 1], 'k--')
    plt.plot([0, 11], [0.99, 0.99], 'r--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_interp_ma():
    nodes = [8]
    methods = ['ma/random/out_r1.csv', 'ma/interp/out.csv']
    lines = []
    for method_ind, method in enumerate(methods):
        for color_ind, n in enumerate(nodes):
            extra = 'ed_4' if n == 8 else ''
            ps, p_series = get_column_statistic(f'graphs/nodes_{n}/{extra}/output/{method}', r'p_\d+$')
            if method_ind == 1:
                p_series[0] = lines[0].ys[0]
            line_style = '-' if method_ind < 1 else '--'
            lines.append(Line(ps, p_series, colors[color_ind], markers[method_ind], line_style))
    plot_general(lines, ('p', 'Average AR'), (1, 0.02), (0.75, 10.25, None, 1.0025))
    plt.plot([0, 11], [1, 1], 'k--')
    plt.plot([0, 11], [0.99, 0.99], 'r--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_interp_nodes():
    nodes = range(9, 13)
    methods = ['qaoa/interp/out.csv', 'ma/qaoa/out.csv']
    lines = []
    markers = 'oX'
    for method_ind, method in enumerate(methods):
        for n_ind, n in enumerate(nodes):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}', r'p_\d+$')
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(ps, p_series, colors[n_ind], markers[method_ind], line_style))
    x_lim = [0.75, 14.25]
    plot_general(lines, ('p', 'Average AR'), (1, 0.01), (*x_lim, None, 1.0025))
    plt.legend([f'{x} nodes' for x in nodes], loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    plt.text(7.3, 0.96, 'QAOA')
    plt.text(2.2, 0.98, 'MA-QAOA')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_interp_edges():
    depths = range(3, 7)
    methods = ['qaoa/interp/out.csv', 'ma/qaoa/out.csv']
    lines = []
    markers = 'oX'
    for method_ind, method in enumerate(methods):
        for depth_ind, depth in enumerate(depths):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_12/depth_{depth}/output/{method}', r'p_\d+$')
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(ps, p_series, colors[depth_ind], markers[method_ind], line_style))
    x_lim = [0.75, 14.25]
    plot_general(lines, ('p', 'Average AR'), (1, 0.01), (*x_lim, None, 1.0025))
    plt.legend([f'Depth = {x}' for x in depths], loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    plt.text(7.3, 0.96, 'QAOA')
    plt.text(3.7, 0.98, 'MA-QAOA')
    save_figure()
    plt.show()


def plot_min_ar_vs_p_nodes():
    nodes = range(9, 10)
    methods = ['qaoa/greedy/out.csv', 'ma/explicit/out.csv']
    lines = []
    markers = 'oX'
    for method_ind, method in enumerate(methods):
        for n_ind, n in enumerate(nodes):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}', r'p_\d+$', min)
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(ps, p_series, colors[n_ind], markers[method_ind], line_style))
    x_lim = [0.75, 14.25]
    plot_general(lines, ('p', 'Min AR'), (1, 0.02), (*x_lim, None, 1.0025))
    plt.legend([f'{x} nodes' for x in nodes], loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    plt.text(5.5, 0.72, 'QAOA')
    plt.text(3.8, 0.96, 'MA-QAOA')
    save_figure()
    plt.show()


def plot_min_ar_vs_p_interp_edges():
    depths = range(3, 7)
    methods = ['combined/out.csv', 'ma/explicit/out.csv']
    lines = []
    markers = 'oX'
    for method_ind, method in enumerate(methods):
        for depth_ind, depth in enumerate(depths):
            ps, avg_ar = get_column_statistic(f'graphs/new/nodes_12/depth_{depth}/output/{method}', r'p_\d+$', min)
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(ps, avg_ar, colors[depth_ind], markers[method_ind], line_style))
    x_lim = [0.75, 24.25]
    plot_general(lines, ('p', 'Min AR'), (1, 0.02), (*x_lim, None, 1.0025))
    plt.legend([f'Depth = {x}' for x in depths], loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    plt.text(5.5, 0.72, 'QAOA')
    plt.text(5.4, 0.96, 'MA-QAOA')
    save_figure()
    plt.show()


def plot_avg_ar_vs_p_experimental_test():
    nodes = range(9, 10)
    methods = ['ma/explicit/out.csv', 'ma/random/qaoa/out.csv', 'ma/random/ma/out_r1.csv', 'gen1/out.csv', 'gen12e/out.csv', 'xqaoa/random/out.csv', 'xqaoa/explicit/out.csv']
    lines = []
    for method_ind, method in enumerate(methods):
        for n_ind, n in enumerate(nodes):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}', r'p_\d+$')
            lines.append(Line(ps, p_series, colors[method_ind]))
    x_lim = [0.75, 5.25]
    plot_general(lines, ('p', 'Average AR'), (1, 0.01), (*x_lim, None, 1.0025))
    labels = ['Optimal QAOA -> MA', 'Random QAOA -> MA', 'Random MA', 'Generalized 1', 'Generalized 2(e)', 'Random XQAOA', 'Optimal QAOA -> XQAOA']
    plt.legend(labels, loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    save_figure()
    plt.show()


def plot_min_ar_vs_p_experimental_test():
    # methods = ['ma/explicit/out.csv', 'ma/random/qaoa/out.csv', 'ma/random/ma/out_r1.csv', 'gen1/out.csv', 'gen12e/out.csv', 'xqaoa/random/out.csv', 'xqaoa/explicit/out.csv']
    # labels = ['Optimal QAOA -> MA', 'Random QAOA -> MA', 'Random MA', 'Generalized 1', 'Generalized 2(e)', 'Random XQAOA', 'Optimal QAOA -> XQAOA']

    methods = ['qaoa/random/out_r1.csv', 'qaoa/greedy/out.csv', 'qaoa/interp/out.csv', 'qaoa/fourier/out.csv', 'qaoa/tqa/out.csv', 'qaoa/combined/out.csv', 'qaoa/constant/out.csv']
    labels = ['Random', 'Greedy', 'Interp', 'Fourier', 'TQA', 'Combined', 'Constant']

    nodes = range(9, 10)
    lines = []
    for method_ind, method in enumerate(methods):
        for n_ind, n in enumerate(nodes):
            ps, p_series = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}', r'p_\d+$', min)
            lines.append(Line(ps, p_series, colors[method_ind]))

    x_lim = [0.75, 10.25]
    plot_general(lines, ('p', 'Min AR'), (1, 0.02), (*x_lim, None, 1.005))
    plt.legend(labels, loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [16/17, 16/17], 'r--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_cost_interp_nodes():
    nodes = range(9, 13)
    methods = ['qaoa/interp/out.csv', 'ma/qaoa/out.csv']
    lines = []
    markers = 'oX'
    for method_ind, method in enumerate(methods):
        for n_ind, n in enumerate(nodes):
            ps, avg_ar = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}', r'p_\d+$')
            ps, avg_nfev = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/{method}', r'p_\d+_nfev')
            avg_cost = np.array(avg_nfev) * np.array(ps)
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(avg_cost, avg_ar, colors[n_ind], markers[method_ind], line_style))
    x_lim = [-1000, 40000]
    plot_general(lines, ('Cost', 'Avg AR'), (None, 0.01), (*x_lim, None, 1.0025))
    plt.legend([f'{x} nodes' for x in nodes], loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    plt.text(0, 0.9825, 'QAOA')
    plt.text(13000, 0.955, 'MA-QAOA')
    save_figure()
    plt.show()


def plot_avg_ar_vs_cost_interp_edges():
    depths = range(3, 7)
    methods = ['qaoa/interp/out.csv', 'ma/qaoa/out.csv']
    lines = []
    markers = 'oX'
    for method_ind, method in enumerate(methods):
        for depth_ind, depth in enumerate(depths):
            ps, avg_ar = get_column_statistic(f'graphs/new/nodes_12/depth_{depth}/output/{method}', r'p_\d+$')
            ps, avg_nfev = get_column_statistic(f'graphs/new/nodes_12/depth_{depth}/output/{method}', r'p_\d+_nfev')
            avg_cost = np.array(avg_nfev) * np.array(ps)
            line_style = '-' if method_ind < 2 else '--'
            lines.append(Line(avg_cost, avg_ar, colors[depth_ind], markers[method_ind], line_style))
    x_lim = [-1000, 40000]
    plot_general(lines, ('Cost', 'Avg AR'), (None, 0.01), (*x_lim, None, 1.0025))
    plt.legend([f'Depth = {x}' for x in depths], loc='lower right', fontsize='small')
    plt.plot(x_lim, [1, 1], 'k--')
    plt.plot(x_lim, [0.99, 0.99], 'r--')
    plt.text(0, 0.98, 'QAOA')
    plt.text(15000, 0.96, 'MA-QAOA')
    save_figure()
    plt.show()


def plot_interp_random_ar_difference_vs_p_nodes():
    nodes = range(9, 13)
    lines = []
    for n_ind, n in enumerate(nodes):
        ps, ar_qaoa = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/ma/qaoa/out.csv', r'p_\d+$')
        ps, ar_rand = get_column_statistic(f'graphs/new/nodes_{n}/depth_3/output/ma/random/out_r1.csv', r'p_\d+$')
        diff = ar_qaoa - ar_rand
        lines.append(Line(ps, diff, colors[n_ind]))

    plot_general(lines, ('p', 'Average AR difference'), (1, 0.002), (0.75, 5.25, None, None))
    save_figure()
    plt.show()


def plot_interp_random_ar_difference_vs_p_edges():
    eds = [3.5, 4, 4.5, 5]
    lines = []
    for ed_ind, ed in enumerate(eds):
        ps, ar_qaoa = get_column_statistic(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/qaoa/out.csv', r'p_\d+$')
        ps, ar_rand = get_column_statistic(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/random/out_r1.csv', r'p_\d+$')
        diff = ar_qaoa - ar_rand
        lines.append(Line(ps, diff, colors[ed_ind]))

    plot_general(lines, ('p', 'Average AR difference'), (1, 0.002), (0.75, 5.25, None, None))
    save_figure()
    plt.show()


# def plot_min_restarts_vs_p_nodes():
#     nodes = list(range(7, 11))
#     lines = []
#     for n_ind, n in enumerate(nodes):
#         extra = 'ed_4' if n == 8 else ''
#         ps, ar_qaoa = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/qaoa/out.csv', r'p_\d+$')
#         ps, ar_rand = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/random/out_r1.csv', r'p_\d+$')
#         diff = ar_qaoa - ar_rand
#         lines.append(Line(ps, diff, colors[n_ind]))
#
#     plot_general(lines, ('p', 'Average AR difference'), (1, 0.002), (0.75, 5.25, None, None))
#     save_figure()
#     plt.show()


def plot_converged_fraction_vs_min_p_nodes():
    convergence_level = 0.99
    nodes = range(7, 11)
    lines = []
    for n_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        df = pd.read_csv(f'graphs/nodes_{n}/{extra}/output/ma/qaoa/out.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        ys, xs = np.histogram(df['min_p'], range(1, 6))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[n_ind]))
    plot_general(lines, ('min p', 'Fraction of converged graphs'), (1, 0.1), (0.75, 5.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_min_p_edges():
    convergence_level = 0.99
    eds = [3.5, 4, 4.5, 5]
    lines = []
    for ed_ind, ed in enumerate(eds):
        df = pd.read_csv(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/qaoa/out.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        ys, xs = np.histogram(df['min_p'], range(1, 6))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[ed_ind]))
    plot_general(lines, ('min p', 'Fraction of converged graphs'), (1, 0.1), (0.75, 5.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_min_p_r10_nodes():
    convergence_level = 0.99
    nodes = range(7, 11)
    lines = []
    for n_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        df = pd.read_csv(f'graphs/nodes_{n}/{extra}/output/ma/random/out_r10.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        ys, xs = np.histogram(df['min_p'], range(1, 6))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[n_ind]))
    plot_general(lines, ('min p', 'Fraction of converged graphs'), (1, 0.1), (0.75, 5.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_min_p_r10_edges():
    convergence_level = 0.99
    eds = [3.5, 4, 4.5, 5]
    lines = []
    for ed_ind, ed in enumerate(eds):
        df = pd.read_csv(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/random/out_r10.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        ys, xs = np.histogram(df['min_p'], range(1, 6))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[ed_ind]))
    plot_general(lines, ('min p', 'Fraction of converged graphs'), (1, 0.1), (0.75, 5.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_rel_p_nodes():
    convergence_level = 0.99
    nodes = range(7, 11)
    lines = []
    for n_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        df = pd.read_csv(f'graphs/nodes_{n}/{extra}/output/ma/qaoa/out.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        ys, xs = np.histogram(df['min_p'] - df['edge_diameter'], range(-5, 3))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[n_ind]))
    plot_general(lines, ('min p - diameter', 'Fraction of converged graphs'), (1, 0.1), (-5.25, 1.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_rel_p_edges():
    convergence_level = 0.99
    eds = [3.5, 4, 4.5, 5]
    lines = []
    for ed_ind, ed in enumerate(eds):
        df = pd.read_csv(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/qaoa/out.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        ys, xs = np.histogram(df['min_p'] - df['edge_diameter'], range(-5, 3))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[ed_ind]))
    plot_general(lines, ('min p - diameter', 'Fraction of converged graphs'), (1, 0.1), (-5.25, 1.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_rel_p_r10_nodes():
    convergence_level = 0.99
    nodes = range(7, 11)
    lines = []
    for n_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        df = pd.read_csv(f'graphs/nodes_{n}/{extra}/output/ma/random/out_r10.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        df = calculate_edge_diameter(df)
        ys, xs = np.histogram(df['min_p'] - df['edge_diameter'], range(-5, 3))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[n_ind]))
    plot_general(lines, ('min p - diameter', 'Fraction of converged graphs'), (1, 0.1), (-5.25, 1.25, None, None))
    save_figure()
    plt.show()


def plot_converged_fraction_vs_rel_p_r10_edges():
    convergence_level = 0.99
    eds = [3.5, 4, 4.5, 5]
    lines = []
    for ed_ind, ed in enumerate(eds):
        df = pd.read_csv(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/random/out_r10.csv', index_col=0)
        df = calculate_min_p(df, convergence_level)
        df = calculate_edge_diameter(df)
        ys, xs = np.histogram(df['min_p'] - df['edge_diameter'], range(-5, 3))
        lines.append(Line(xs[:-1], ys / df.shape[0], colors[ed_ind]))
    plot_general(lines, ('min p - diameter', 'Fraction of converged graphs'), (1, 0.1), (-5.25, 1.25, None, None))
    save_figure()
    plt.show()


# def plot_avg_ar_vs_restarts():
#     nodes = range(7, 11)
#     ps = range(1, 4)
#     max_restarts = 10
#     lines = []
#     for marker_ind, p in enumerate(ps):
#         for color_ind, n in enumerate(nodes):
#             extra = 'ed_4' if n == 8 else ''
#             xs, ys = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/random/p_{p}/out.csv', r'r_\d+$')
#             if len(xs) > max_restarts:
#                 xs = xs[:max_restarts]
#                 ys = ys[:max_restarts]
#             lines.append(Line(xs, ys, colors[color_ind], markers[marker_ind]))
#     plot_general(lines, 'Restarts', 'Average AR')
#     plt.gca().xaxis.set_major_locator(MultipleLocator(1))
#     plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
#     plt.xlim(left=0.75, right=10.25)
#     plt.ylim(top=1.0025)
#     plt.plot([0, 11], [1, 1], 'k--')
#     plt.savefig('temp/figures/avg_ar_vs_restarts.jpg', dpi=300, bbox_inches='tight')
#     plt.show()


if __name__ == "__main__":
    # plot_avg_ar_vs_p_r1_nodes()
    # plot_avg_ar_vs_p_r1_edges()
    # plot_avg_ar_vs_p_r1_interp_nodes()
    # plot_avg_ar_vs_p_r1_interp_edges()
    # plot_avg_ar_vs_p_interp_ma()

    # plot_avg_ar_vs_p_interp_nodes()
    # plot_avg_ar_vs_p_interp_edges()
    # plot_min_ar_vs_p_nodes()
    # plot_min_ar_vs_p_interp_edges()
    # plot_avg_ar_vs_p_experimental_test()
    plot_min_ar_vs_p_experimental_test()
    # plot_avg_ar_vs_cost_interp_nodes()
    # plot_avg_ar_vs_cost_interp_edges()
    # plot_interp_random_ar_difference_vs_p_nodes()
    # plot_interp_random_ar_difference_vs_p_edges()
    # plot_converged_fraction_vs_min_p_nodes()
    # plot_converged_fraction_vs_min_p_edges()
    # plot_converged_fraction_vs_min_p_r10_nodes()
    # plot_converged_fraction_vs_min_p_r10_edges()
    # plot_converged_fraction_vs_rel_p_nodes()
    # plot_converged_fraction_vs_rel_p_edges()
    # plot_converged_fraction_vs_rel_p_r10_nodes()
    # plot_converged_fraction_vs_rel_p_r10_edges()

    # plot_conv_avg_ar_vs_restarts_ma()
    # plot_num_converged_vs_rel_p()
    # plot_converged_fraction_vs_min_p_nodes()
    # plot_converged_fraction_vs_min_p_edges()
    # plot_interp_random_ar_difference_vs_nodes()
