"""this handles some basic plotting functions"""

from typing import List
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


def show_one_main(stat_sub_array: list, stat_all_array: list, stat_name_array: list, *,
                  ax: Axes = None, title: str = None, ylabel: str = None,
                  yticks=(0, 0.2, 0.4, 0.6, 0.8, 1),
                  yticklabels=('0', '0.2', '0.4', '0.6', '0.8', '1'),
                  color_list=None, stat_ref_name='cnn',
                  ):
    # based on https://github.com/leelabcnbc/tang_jcompneuro/blob/master/thesis_plots/v1_fitting/results_basic.ipynb
    assert len(stat_sub_array) == len(stat_all_array) == len(stat_name_array)
    if ax is None:
        ax = plt.gca()

    if color_list is None:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        color_list = plt.get_cmap('Set1').colors

    stat_all_ref = stat_all_array[stat_name_array.index(stat_ref_name)]
    counter_now = 0
    label_grp = []
    rect_grp = []
    for model_class_idx, (stat_sub, stat_all, stat_name) in enumerate(
            zip(stat_sub_array, stat_all_array, stat_name_array)):
        num_model_this = len(stat_sub)
        model_names, model_perfs = zip(*stat_sub)
        rects_this = ax.bar(counter_now + np.arange(num_model_this) + 1,
                            model_perfs,
                            0.95, color=color_list[model_class_idx])

        label_grp.append(stat_name)
        rect_grp.append(rects_this[0])

        for text_idx, text in enumerate(model_names):
            ax.text(text_idx + 1 + counter_now,
                    model_perfs[text_idx],
                    s=text, rotation='vertical', horizontalalignment='center',
                    verticalalignment='top', color='white', fontsize='small')

        assert stat_all is not None
        rc, = ax.plot([counter_now + 0.5, counter_now + num_model_this + 0.5], [stat_all, stat_all],
                      color=color_list[model_class_idx], linestyle='--')
        rect_grp.append(rc)
        label_grp.append(f'{stat_name}_all')
        ax.text(counter_now + num_model_this / 2 + 0.5, stat_all, s='{:.3f}'.format(stat_all),
                horizontalalignment='center',
                verticalalignment='bottom', color='black', fontsize='small')
        if stat_name != stat_ref_name:
            # this works because CNN model is put first.
            ax.text(counter_now + num_model_this / 2 + 0.5,
                    stat_all + 0.1, s='{:.1f}%'.format(((stat_all_ref / stat_all) - 1) * 100),
                    horizontalalignment='center',
                    verticalalignment='bottom', color='black', fontsize='x-small', fontweight='bold')
        counter_now += num_model_this + 1

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, counter_now)
    ax.set_ylim(0, 1)
    ax.set_xticks([])

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.legend(rect_grp, label_grp, loc='best', fontsize='small', ncol=2, columnspacing=0)


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
    # sometimes can go over 1, for ccnorm_5.
    assert np.all(x >= 0) and np.all(x <= 1.5)
    assert np.all(y >= 0) and np.all(y <= 1.5)

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


def show_one_decomposed_bar(stat_chunks_array, stat_name_array, *,
                            ax: Axes = None, xlabel=None,
                            title=None,
                            color_bias: int = None, set_ylabel=False):
    # https://github.com/leelabcnbc/tang_jcompneuro/blob/master/thesis_plots/v1_fitting/comparison_among_all_non_vgg_models_decomposed_by_fine_subsets.ipynb
    color_list = plt.get_cmap('Set2').colors

    assert isinstance(stat_chunks_array, np.ndarray) and stat_chunks_array.ndim == 2
    assert stat_chunks_array.shape[1] == len(stat_name_array)

    assert color_bias is not None

    if ax is None:
        ax = plt.gca()

    n_model = len(stat_name_array)

    data_mean_bottom = np.zeros((n_model,), dtype=np.float64)
    for chunk_idx, chunk_data in enumerate(stat_chunks_array):
        ax.barh(np.arange(n_model) + 1,
                chunk_data, height=0.95,
                left=data_mean_bottom,
                color=color_list[color_bias + chunk_idx])
        assert data_mean_bottom.shape == chunk_data.shape
        data_mean_bottom += chunk_data
    ax.set_xlim(0, data_mean_bottom.max() * 1.1)
    ax.set_ylim(0, n_model + 2)

    if set_ylabel:
        ax.set_yticks(np.arange(n_model)+1)
        ax.set_yticklabels(stat_name_array, fontdict={'fontsize': 'medium'})

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def show_one_decomposed_scatter():
    pass
