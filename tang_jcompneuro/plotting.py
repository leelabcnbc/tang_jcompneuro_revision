"""this handles some basic plotting functions"""

from typing import List, Tuple
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# first, a function to show scatter plot of two models,
# without all cell classification stuff.
# this will be used in supplementary material to show
# things like different nonlinearity of GLM
# or different CNN architectures.

image_subset_and_neuron_subset_list = (
    ('OT', 'OT'),
    ('all', 'OT'),
    ('all', 'HO')
)


def show_one_basic(x: np.ndarray, y: np.ndarray, *,
                   xlabel: str = None, ylabel: str = None, title: str = None, ax: Axes = None,
                   show_pearson: bool = True, show_mean: bool = True, mean_title=None,
                   x_mean_loc=(1, 0),  # show x bottom right
                   y_mean_loc=(0, 1),  # show y top left
                   pearson_loc=(0, 0.75),  # a little below xmean
                   show_ref_line=True, fontsize='small',
                   alpha=0.25, s=6,
                   xticks=(0, 0.5, 1),
                   yticks=(0, 0.5, 1),
                   xticklabels=('0', '0.5', '1'),
                   yticklabels=('0', '0.5', '1'),
                   ):
    if ax is None:
        ax = plt.gca()

    assert x.shape == y.shape == (x.size,)
    assert np.all(x >= 0) and np.all(x <= 1)
    assert np.all(y >= 0) and np.all(y <= 1)

    ax.scatter(x, y, alpha=alpha, s=s)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    if show_ref_line:
        ax.plot([0, 1], [0, 1], linestyle='--')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show_mean:

        if mean_title is not None:
            prefix = mean_title + '\n'
        else:
            prefix = ''

        ax.text(*x_mean_loc, '{}{:.3f}'.format(prefix, x.mean()),
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=fontsize)
        ax.text(*y_mean_loc, '{}{:.3f}'.format(prefix, y.mean()),
                horizontalalignment='left',
                verticalalignment='top', fontsize=fontsize)

    if show_pearson:
        pearson_this = pearsonr(x, y)[0]
        assert np.isfinite(pearson_this)

        ax.text(*pearson_loc, 'n={}\nr={:.4f}'.format(x.size, pearson_this), fontsize=fontsize,
                horizontalalignment='left',
                verticalalignment='top')


def show_perf_vs_size(x_list: List[np.ndarray],
                      y_list: List[np.ndarray],
                      label_list: List[str], *,
                      xlabel: str = None, ylabel: str = None, title: str = None, ax: Axes = None,
                      xticks=(0, 25, 50, 75, 100),
                      yticks=(0, 0.5, 1),
                      xlim=(0, 100),
                      ylim=(0, 1),

                      xticklabels=('0', '25', '50', '75', '100'),
                      yticklabels=('0', '0.5', '1'),
                      style_list=None,
                      linewidth=1,
                      show_legend=True,
                      vline=None
                      ):
    """x being model size, number of parameter, dataset size, etc.
    y being performance.
    """

    if style_list is None:
        # should give a default set
        raise NotImplementedError

    assert len(x_list) == len(y_list) == len(label_list)
    for idx, (x_this, y_this, label_this) in enumerate(zip(x_list, y_list, label_list)):
        linestyle, color, marker = style_list[idx]
        ax.plot(x_this, y_this,
                linestyle=linestyle, color=color, marker=marker, label=label_this,
                linewidth=linewidth)

    if vline is not None:
        # color maybe adjusted later
        ax.axvline(vline, color='r', linewidth=linewidth)

    # ax.set_xlim(0, 1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show_legend:
        ax.legend()