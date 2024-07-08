"""
General plotting functions.
"""
import inspect
from dataclasses import dataclass
from typing import Sequence

import distinctipy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator
from numpy import ndarray

colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (0, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75), (0.75, 0.75, 0), (1, 0.5, 0), (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76),
          (0.5, 0.5, 0.5)]
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
    marker: str | int = 'o'
    style: str | int = '-'
    label: str = '_nolabel_'

    def set_color(self, color: tuple | int):
        self.color = color
        if isinstance(color, int):
            self.color = colors[color]

    def __post_init__(self):
        if isinstance(self.color, int):
            self.color = colors[self.color]
        if isinstance(self.marker, int):
            self.marker = markers[self.marker]
        if isinstance(self.style, int):
            self.style = styles[self.style]


def data_matrix_to_lines(data: ndarray, line_labels: list[str] = None, colors: list[int] = None, **kwargs) -> list[Line]:
    """
    Converts a given data matrix to a set of lines (each line is a row).
    :param data: 3D data matrix of size 2 x num_lines x num_points. 1st dim - (x, y); 2nd - lines; 3rd - data points.
    Trailing zeros in each row are ignored and have to be consistent for x and y dimensions.
    :param line_labels: Line labels.
    :param colors: Line colors.
    :return: List of lines.
    """
    lines = []
    for i in range(data.shape[1]):
        xs = np.trim_zeros(data[0, i, :], trim='b')
        ys = np.trim_zeros(data[1, i, :], trim='b')
        lines.append(Line(xs, ys))
        if colors is not None:
            lines[-1].set_color(colors[i])
        if line_labels is not None:
            lines[-1].label = line_labels[i]
    return lines


def assign_distinct_colors(lines: list[Line]):
    """
    Assigns distinct colors to a collection of lines from `colors` variable.
    :param lines: List of lines.
    :return: None.
    """
    for i in range(len(lines)):
        lines[i].color = colors[i]


def pick_event_handler(event):
    if isinstance(event.artist, Line2D):
        x = event.artist.get_xdata()[event.ind][0]
        y = event.artist.get_ydata()[event.ind][0]
        bbox_settings = dict(boxstyle='round', fc=(1.0, 0.7, 0.7), ec='none')
        arrow_settings = dict(arrowstyle='wedge, tail_width=1', fc=(1.0, 0.7, 0.7), ec='none', patchA=None, patchB=Ellipse((2, -1), 0.5, 0.5), relpos=(0.2, 0.5))
        annotation = event.artist.axes.annotate(f'({x:.3g}, {y:.3g})', xy=(x, y), xytext=(20, 20), textcoords='offset points', size=10, bbox=bbox_settings,
                                                arrowprops=arrow_settings)
        annotation.draggable()
        event.canvas.draw()
        print(x, y)


def plot_general(lines: list[Line], axis_labels: tuple[str | None, str | None] = None, tick_multiples: tuple[float | None, float | None] = None,
                 boundaries: tuple[float | None, float | None, float | None, float | None] = None, font_size: int = 20, legend_loc: str = 'best', figure_id: int = None, **kwargs):
    """
    Plots specified list of lines.
    :param lines: List of lines.
    :param axis_labels: Labels for x and y axes.
    :param tick_multiples: Base multiples for ticks along x and y axes.
    :param boundaries: x min, x max, y min, y max floats defining plot boundaries.
    :param font_size: Font size.
    :param legend_loc: Location of legend.
    :param figure_id: ID of the figure where the results should be plotted or None to create new figure.
    :return: None.
    """
    if figure_id is None:
        new_figure = True
        fig = plt.figure()
    else:
        new_figure = plt.fignum_exists(figure_id)
        fig = plt.figure(figure_id)
    fig.canvas.mpl_connect('pick_event', pick_event_handler)

    plt.rcParams.update({'font.size': font_size})
    for line in lines:
        plt.plot(line.xs, line.ys, color=line.color, marker=line.marker, linestyle=line.style, markersize=marker_sizes[line.marker], label=line.label, picker=5)
        if line.label != '_nolabel_':
            plt.legend(loc=legend_loc, draggable=True)

    if axis_labels is not None:
        if axis_labels[0] is not None:
            plt.xlabel(axis_labels[0])
        if axis_labels[1] is not None:
            plt.ylabel(axis_labels[1])

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

    if new_figure:
        plt.get_current_fig_manager().window.state('zoomed')
        plt.gca().set_box_aspect(1)
        plt.gcf().set_size_inches(10, 10)
        plt.tight_layout(pad=0.5)


def save_figure(file_name: str = None):
    """
    Saves figure to a file in temp/figures.
    :param file_name: Name of the file or None to use caller's name (without plot_).
    :return: None.
    """
    file_name = inspect.currentframe().f_back.f_code.co_name[5:] if file_name is None else file_name
    plt.savefig(f'temp/figures/{file_name}.jpg', dpi=300, bbox_inches='tight')
