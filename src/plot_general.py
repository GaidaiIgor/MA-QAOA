"""
General plotting functions.
"""
from dataclasses import dataclass

import distinctipy
from matplotlib import pyplot as plt

colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (0, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75), (0.75, 0.75, 0)]
colors += distinctipy.get_colors(10, colors + [(1, 1, 1)])
markers = 'oX*'


@dataclass
class Line:
    """
    Class that represents a line in a 2D plot.
    """
    xs: list[float]
    ys: list[float]
    color: tuple = colors[0]
    marker: str = 'o'
    style: str = 'solid'


def assign_distinct_colors(lines: list[Line]):
    """
    Assigns distinct colors to a collection of lines from `colors` variable.
    :param lines: List of lines.
    :return: None.
    """
    for i in range(len(lines)):
        lines[i].color = colors[i]


def plot_general(lines: list[Line], x_label, y_label):
    """
    Plots specified list of lines.
    :param lines: List of lines.
    :param x_label: x label.
    :param y_label: y label.
    :return: None.
    """
    for line in lines:
        plt.plot(line.xs, line.ys, color=line.color, marker=line.marker, linestyle=line.style, markersize=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
