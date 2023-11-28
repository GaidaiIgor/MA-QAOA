"""
General plotting functions.
"""
import inspect
from dataclasses import dataclass
from typing import Sequence

import distinctipy
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (0, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75), (0.75, 0.75, 0)]
colors += distinctipy.get_colors(10, colors + [(1, 1, 1)])
markers = 'o*Xvs'
marker_sizes = {'o': 5, '*': 8, 'X': 5, 'v': 5, 's': 5, 'none': 0}
styles = ['-', '--']


@dataclass
class Line:
    """ Class that represents a line in a 2D plot. """
    xs: Sequence
    ys: Sequence
    color: tuple | int = colors[0]
    marker: str = 'o'
    style: str = '-'
    label: str = '_nolabel_'

    def __post_init__(self):
        if isinstance(self.color, int):
            self.color = colors[self.color]
        if isinstance(self.marker, int):
            self.marker = markers[self.marker]
        if isinstance(self.style, int):
            self.style = styles[self.style]


def assign_distinct_colors(lines: list[Line]):
    """
    Assigns distinct colors to a collection of lines from `colors` variable.
    :param lines: List of lines.
    :return: None.
    """
    for i in range(len(lines)):
        lines[i].color = colors[i]


def plot_general(lines: list[Line], labels: tuple[str | None, str | None] = None, tick_multiples: tuple[float | None, float | None] = None,
                 boundaries: tuple[float | None, float | None, float | None, float | None] = None, font_size: int = 20):
    """
    Plots specified list of lines.
    :param lines: List of lines.
    :param labels: Labels for x and y axes.
    :param tick_multiples: Base multiples for ticks along x and y axes.
    :param boundaries: x min, x max, y min, y max floats defining plot boundaries.
    :param font_size: Font size.
    :return: None.
    """
    plt.rcParams.update({'font.size': font_size})

    for line in lines:
        plt.plot(line.xs, line.ys, color=line.color, marker=line.marker, linestyle=line.style, markersize=marker_sizes[line.marker], label=line.label)

    if labels is not None:
        if labels[0] is not None:
            plt.xlabel(labels[0])
        if labels[1] is not None:
            plt.ylabel(labels[1])

    if tick_multiples is not None:
        if tick_multiples[0] is not None:
            plt.gca().xaxis.set_major_locator(MultipleLocator(tick_multiples[0]))
        if tick_multiples[1] is not None:
            plt.gca().yaxis.set_major_locator(MultipleLocator(tick_multiples[1]))

    if boundaries is not None:
        if boundaries[0] is not None:
            plt.xlim(left=boundaries[0])
        if boundaries[1] is not None:
            plt.xlim(right=boundaries[1])
        if boundaries[2] is not None:
            plt.ylim(bottom=boundaries[2])
        if boundaries[3] is not None:
            plt.ylim(top=boundaries[3])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.gca().set_box_aspect(1)
    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout(pad=0.5)


def save_figure():
    """ Saves figure. """
    file_name = inspect.currentframe().f_back.f_code.co_name[5:]
    plt.savefig(f'temp/figures/{file_name}.jpg', dpi=300, bbox_inches='tight')
