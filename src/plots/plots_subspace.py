""" Plots for MA-subspace performance characterization. """
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing import DataExtractor
from src.plots.plots_qaoa import plot_methods_ar_9_nodes_general


def plot_ar_vs_cost_subspace():
    # param_vals = np.linspace(0.2, 0.8, 4)
    # subspace_methods = [f'ma_subspace/random/frac_{frac:.1f}' for frac in param_vals]
    # subspace_labels = [f'Random {frac:.1f}' for frac in param_vals]

    param_vals = [1]
    subspace_methods = [f'ma_subspace/gradient/ppl_{ppl}' for ppl in param_vals]
    subspace_labels = [f'Random {ppl} PPL' for ppl in param_vals]

    methods = ['qaoa/constant/0.2', 'ma/constant/0.2'] + subspace_methods
    labels = ['QAOA', 'MA'] + subspace_labels
    x_mean_func = DataExtractor.get_cost_average
    x_min_func = DataExtractor.get_cost_worst_case
    axis_labels = ('Cost', 'AR')
    boundaries = (-500, 14500, None, None)
    plot_methods_ar_9_nodes_general(methods, labels, x_mean_func, x_min_func, axis_labels=axis_labels, boundaries=boundaries)


if __name__ == '__main__':
    plot_ar_vs_cost_subspace()

    plt.show()
