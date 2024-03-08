""" Plots for MA-subspace performance characterization. """
import matplotlib.pyplot as plt

from src.data_processing import DataExtractor
from src.plots.plots_qaoa import plot_methods_ar_9_nodes_general


def plot_ar_vs_cost_subspace():
    ppl_vals = [1, 2, 4, 8, 16]
    ppl_methods = [f'ma_subspace/random/ppl_{ppl}' for ppl in ppl_vals]
    ppl_labels = [f'Random {ppl} PPL' for ppl in ppl_vals]

    methods = ['qaoa/constant/0.2'] + ppl_methods
    labels = ['QAOA'] + ppl_labels
    arg_x_func = DataExtractor.get_cost_average
    axis_labels = ('Cost', 'AR')
    boundaries = (-500, 8000, None, None)
    plot_methods_ar_9_nodes_general(methods, labels, arg_x_func, axis_labels=axis_labels, boundaries=boundaries)


if __name__ == '__main__':
    plot_ar_vs_cost_subspace()

    plt.show()
