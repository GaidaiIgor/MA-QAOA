""" Plots for MA-subspace performance characterization. """
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing import DataExtractor
from src.plots.plots_qaoa import plot_methods_9_nodes_general


def plot_ar_vs_cost_subspace():
    x_func = DataExtractor.get_cost_average
    y_func = lambda ext: DataExtractor.get_ar_aggregated(ext, np.mean)
    axis_labels = ('Cost', 'AR')
    boundaries = (-500, 14500, 0.8, 1.00625)
    line_labels = ['QAOA']
    plot_methods_9_nodes_general(['qaoa/constant/0.2'], x_func, y_func, axis_labels=axis_labels, boundaries=boundaries, line_labels=line_labels, figure_id=0)

    transpose = False
    max_p = 5
    if transpose:
        param_vals = [0.1 * i for i in range(11)]
        line_labels = [f'Subspace p = {i}' for i in range(1, max_p + 1)]
    else:
        param_vals = [0.2 * i for i in range(6)]
        line_labels = [f'Subspace {param:.1g}' for param in param_vals]
    methods = [f'ma_subspace/random/frac_{param:.1g}' for param in param_vals]
    plot_methods_9_nodes_general(methods, x_func, y_func, max_p=max_p, transpose=transpose, line_labels=line_labels, colors=range(1, len(methods) + 1), figure_id=0)


if __name__ == '__main__':
    plot_ar_vs_cost_subspace()

    plt.show()
