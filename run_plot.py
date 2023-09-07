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
from src.data_processing import get_column_average
from src.plot_general import colors, Line, assign_distinct_colors, plot_general, markers, save_figure

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


def plot_avg_ar_vs_p_r1():
    nodes = range(7, 11)
    methods = ['qaoa', 'ma']
    p_ranges = [list(range(1, 11)), list(range(1, 6))]
    lines = []
    for marker_ind, method in enumerate(methods):
        for color_ind, n in enumerate(nodes):
            extra = 'ed_4' if n == 8 else ''
            p_series = []
            for p in p_ranges[marker_ind]:
                next_ar = get_column_average(f'graphs/nodes_{n}/{extra}/output/{method}/random/p_{p}/out.csv', r'r_1$')[1][0]
                p_series.append(next_ar)
            lines.append(Line(p_ranges[marker_ind], p_series, colors[color_ind], markers[marker_ind]))
    plot_general(lines, ('p', 'Average AR'), (1, 0.01), (0.75, 10.25, None, 1.0025))
    plt.plot([0, 11], [1, 1], 'k--')
    save_figure()
    plt.show()


def plot_avg_ar_vs_restarts():
    nodes = range(7, 11)
    ps = range(1, 4)
    max_restarts = 10
    lines = []
    for marker_ind, p in enumerate(ps):
        for color_ind, n in enumerate(nodes):
            extra = 'ed_4' if n == 8 else ''
            xs, ys = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/random/p_{p}/out.csv', r'r_\d+$')
            if len(xs) > max_restarts:
                xs = xs[:max_restarts]
                ys = ys[:max_restarts]
            lines.append(Line(xs, ys, colors[color_ind], markers[marker_ind]))
    plot_general(lines, 'Restarts', 'Average AR')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.xlim(left=0.75, right=10.25)
    plt.ylim(top=1.0025)
    plt.plot([0, 11], [1, 1], 'k--')
    plt.savefig('temp/figures/avg_ar_vs_restarts.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_avg_ar_vs_p_nodes():
    methods = ['qaoa/interp', 'ma/qaoa']
    nodes = range(7, 11)
    lines = []
    for marker_ind, method in enumerate(methods):
        for color_ind, n in enumerate(nodes):
            extra = 'ed_4' if n == 8 else ''
            xs, ys = get_column_average(f'graphs/nodes_{n}/{extra}/output/{method}/out.csv', r'p_\d+$')
            lines.append(Line(xs, ys, colors[color_ind], markers[marker_ind]))
    plot_general(lines, 'p', 'Average AR')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.xlim(left=0.75, right=10.25)
    plt.ylim(top=1.0025)
    plt.plot([0, 11], [1, 1], 'k--')
    plt.savefig('temp/figures/avg_ar_vs_p_nodes.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_avg_ar_vs_p_edges():
    methods = ['qaoa/interp', 'ma/qaoa']
    edge_diameters = np.linspace(3.5, 5, 4)
    lines = []
    for marker_ind, method in enumerate(methods):
        for color_ind, ed in enumerate(edge_diameters):
            xs, ys = get_column_average(f'graphs/nodes_8/ed_{ed:.2g}/output/{method}/out.csv', r'p_\d+$')
            lines.append(Line(xs, ys, colors[color_ind], markers[marker_ind]))
    plot_general(lines, 'p', 'Average AR')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.xlim(left=0.75, right=10.25)
    plt.ylim(top=1.0025)
    plt.plot([0, 11], [1, 1], 'k--')
    plt.savefig('temp/figures/avg_ar_vs_p_edges.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_num_converged_vs_rel_p():
    nodes = range(7, 11)
    lines = []
    for color_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        df = pd.read_csv(f'graphs/nodes_{n}/{extra}/output/ma/qaoa/out.csv', index_col=0)
        ys, xs = np.histogram(df['p_rel_ed'], 8, (-5, 3))
        lines.append(Line(xs[:-1], ys / 10000, colors[color_ind]))
    plot_general(lines, 'min p – ED', 'Fraction of converged graphs')
    plt.gca().set_box_aspect(0.5)
    # plt.ylim(bottom=-100)
    plt.savefig(f'temp/figures/num_converged_vs_rel_p.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_converged_fraction_vs_min_p_nodes():
    nodes = range(7, 11)
    lines = []
    for color_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        df = pd.read_csv(f'graphs/nodes_{n}/{extra}/output/ma/random/merged_r10.csv', index_col=0)
        ys, xs = np.histogram(df['min_p'], 5, (1, 6))
        lines.append(Line(xs[:-1], ys / 10000, colors[color_ind]))
    plot_general(lines, 'min p', 'Fraction of converged graphs')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().set_box_aspect(0.5)
    plt.savefig(f'temp/figures/converged_fraction_vs_min_p_nodes_r10.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_converged_fraction_vs_min_p_edges():
    edge_diameters = np.linspace(3.5, 5, 4)
    lines = []
    for color_ind, ed in enumerate(edge_diameters):
        df = pd.read_csv(f'graphs/nodes_8/ed_{ed:.2g}/output/ma/random/merged_r10.csv', index_col=0)
        ys, xs = np.histogram(df['min_p'], 5, (1, 6))
        lines.append(Line(xs[:-1], ys / 10000, colors[color_ind]))
    plot_general(lines, 'min p', 'Fraction of converged graphs')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().set_box_aspect(0.5)
    plt.savefig(f'temp/figures/converged_fraction_vs_min_p_edges.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_interp_random_ar_difference_vs_p():
    nodes = range(7, 11)
    lines = []
    for color_ind, n in enumerate(nodes):
        extra = 'ed_4' if n == 8 else ''
        ps, qaoa_ar = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/qaoa/out.csv', r'p_\d+$')
        random_ar = []
        for p in ps:
            next_ar = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/random/p_{p}/out.csv', r'r_1$')[1][0]
            random_ar.append(next_ar)
        lines.append(Line(ps, qaoa_ar - np.array(random_ar), colors[color_ind]))
    plot_general(lines, 'p', 'Average AR difference')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.001))
    plt.xlim(left=0.75, right=5.25)
    plt.savefig('temp/figures/interp_random_ar_difference_vs_p.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_interp_random_ar_difference_vs_nodes():
    nodes = list(range(7, 11))
    ps = list(range(1, 6))
    ar_qaoa = []
    ar_random = []
    for n in nodes:
        extra = 'ed_4' if n == 8 else ''
        calc_ps, next_ar_qaoa = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/qaoa/out.csv', r'p_\d+$')
        next_ar_qaoa = next_ar_qaoa.to_list()
        if ps[-1] > calc_ps[-1]:
            next_ar_qaoa += [1] * (ps[-1] - calc_ps[-1])
        ar_qaoa.append(next_ar_qaoa)
        ar_random_p_series = []
        for p in ps:
            next_ar_random = get_column_average(f'graphs/nodes_{n}/{extra}/output/ma/random/p_{p}/out.csv', r'r_1$')[1][0]
            ar_random_p_series.append(next_ar_random)
        ar_random.append(ar_random_p_series)
    ar_qaoa = np.array(ar_qaoa)
    ar_random = np.array(ar_random)

    ar_qaoa = ar_qaoa.transpose()
    ar_random = ar_random.transpose()

    lines = []
    for row_ind in range(ar_qaoa.shape[0]):
        lines.append(Line(nodes, ar_qaoa[row_ind, :] - ar_random[row_ind, :], colors[row_ind]))

    plot_general(lines, 'Nodes', 'Average AR difference')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.001))
    # plt.xlim(left=0.75, right=5.25)
    plt.savefig('temp/figures/interp_random_ar_difference_vs_nodes.jpg', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_avg_ar_vs_p_r1()
    # plot_conv_avg_ar_vs_restarts_ma()
    # plot_avg_ar_vs_p_nodes()
    # plot_avg_ar_vs_p_edges()
    # plot_num_converged_vs_rel_p()
    # plot_converged_fraction_vs_min_p_nodes()
    # plot_converged_fraction_vs_min_p_edges()
    # plot_interp_random_ar_difference_vs_p()
    # plot_interp_random_ar_difference_vs_nodes()
