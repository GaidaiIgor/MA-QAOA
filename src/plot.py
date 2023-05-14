import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from networkx import Graph

from src.original_qaoa import run_qaoa_analytical_p1


def plot_qaoa_expectation_p1(graph: Graph, edge_list: list[tuple[int, int]] = None):
    beta = np.linspace(-np.pi, np.pi, 361)
    gamma = np.linspace(-np.pi, np.pi, 361)
    beta_mesh, gamma_mesh = np.meshgrid(beta, gamma)
    expectation = np.zeros_like(beta_mesh)
    for i in range(len(beta)):
        for j in range(len(beta)):
            angles = np.array([beta_mesh[i, j], gamma_mesh[i, j]])
            expectation[i, j] = run_qaoa_analytical_p1(angles, graph, edge_list)

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


def main():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    main()
