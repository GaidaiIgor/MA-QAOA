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
from src.plot_general import colors, Line, assign_distinct_colors, plot_general, markers

import addcopyfighandler
assert addcopyfighandler, "Adds an option to copy figures by pressing Ctrl+C"


def extract_numbers(str_arr: list[str]) -> list[int]:
    """ Extracts numbers after _ from column names. """
    return [int(name.split('_')[1]) for name in str_arr]


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


def get_ma_conv_exp_vs_restarts_line(df_path: str):
    df = pd.read_csv(df_path, index_col=0).iloc[:, ::2]
    restarts = extract_numbers(df.columns)
    expectations = np.mean(df, axis=0)
    return restarts, expectations


def plot_ma_conv_exp_vs_restarts():
    nodes = range(7, 11)
    ps = range(1, 4)
    lines = []
    for marker_ind, p in enumerate(ps):
        for color_ind, n in enumerate(nodes):
            extra = 'ed_4' if n == 8 else ''
            xs, ys = get_ma_conv_exp_vs_restarts_line(f'graphs/nodes_{n}/{extra}/output/ma/random/p_{p}/out.csv')
            lines.append(Line(xs, ys, colors[color_ind], markers[marker_ind]))
    plot_general(lines, 'Restarts', 'Expectation')
    plt.get_current_fig_manager().window.state('zoomed')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.xlim(left=0.5, right=10)
    plt.gca().set_box_aspect(1)
    plt.gcf().set_size_inches(8, 8)
    plt.tight_layout(pad=0.05)
    plt.savefig('temp/figures/ma_convergence.jpg', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_ma_conv_exp_vs_restarts()

