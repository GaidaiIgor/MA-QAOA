"""
Contains plot functions.
"""
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator
from networkx import Graph

from src.analytical import calc_expectation_ma_qaoa_analytical_p1
from src.angle_strategies import qaoa_decorator


def plot_qaoa_expectation_p1(graph: Graph, edge_list: list[tuple[int, int]] = None):
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


def plot_edge_diameter_vs_min_p():
    df = pd.read_csv('graphs/nodes_7/output/ma/random/out.csv', index_col=0)
    plt.plot(df['min_p'], df['edge_diameter'])
    plt.xlabel('Minimum p')
    plt.ylabel('Edge diameter')
    plt.show()


def plot_edge_diameter_vs_min_p_2():
    df = pd.read_csv('graphs/nodes_7/output/ma/random/out.csv', index_col=0)
    plt.hist2d(df['min_p'], df['edge_diameter'])
    plt.xlabel('Minimum p')
    plt.ylabel('Edge diameter')
    plt.show()


def plot_edge_diameter_vs_min_p_3():
    p = 3
    df = pd.read_csv('graphs/nodes_7/output/ma/random/out.csv', index_col=0)
    eds = np.unique(df['edge_diameter'])
    counts = [sum((df['edge_diameter'] == ed) & (df['min_p'] == p)) for ed in eds]
    plt.plot(eds, counts)
    plt.xlabel('Edge diameter')
    plt.ylabel('# of converged graphs')
    plt.show()


if __name__ == "__main__":
    plot_edge_diameter_vs_min_p_3()
