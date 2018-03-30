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
                      legend_param=None,
                      vline=None,
                      hline=None,
                      xlabel_param=None,
                      # letter=None,
                      ):
    """x being model size, number of parameter, dataset size, etc.
    y being performance.
    """

    if style_list is None:
        # should give a default set
        raise NotImplementedError

    if xlabel_param is None:
        xlabel_param = dict()

    # if letter is not None:
    #     ax.text(0, 1, letter, horizontalalignment='left', verticalalignment='top',
    #             transform=ax.get_figure().transFigure, fontweight='bold')

    assert len(x_list) == len(y_list) == len(label_list)
    for idx, (x_this, y_this, label_this) in enumerate(zip(x_list, y_list, label_list)):
        linestyle, color, marker = style_list[idx]
        ax.plot(x_this, y_this,
                linestyle=linestyle, color=color, marker=marker, label=label_this,
                linewidth=linewidth)

    if vline is not None:
        # color maybe adjusted later
        ax.axvline(vline, color='black', linewidth=linewidth, linestyle='--')

    if hline is not None:
        # color maybe adjusted later
        ax.axhline(hline, color='black', linewidth=linewidth, linestyle='--')

    # ax.set_xlim(0, 1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, **xlabel_param)
    ax.set_yticklabels(yticklabels)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show_legend:
        if legend_param is None:
            ax.legend()
        else:
            ax.legend(**legend_param)


def show_one_decomposed_bar(stat_chunks_array, stat_name_array, *,
                            ax: Axes = None, xlabel=None,
                            title=None,
                            color_bias: int = None, set_ylabel=False,
                            ylabel_styles=None, letter_map=None):
    # https://github.com/leelabcnbc/tang_jcompneuro/blob/master/thesis_plots/v1_fitting/comparison_among_all_non_vgg_models_decomposed_by_fine_subsets.ipynb
    color_list = plt.get_cmap('Set2').colors

    assert isinstance(stat_chunks_array, np.ndarray) and stat_chunks_array.ndim == 2
    assert stat_chunks_array.shape[1] == len(stat_name_array)

    assert color_bias is not None

    if ax is None:
        ax = plt.gca()
    if letter_map is not None:
        ax.text(-0.02, 1, chr(letter_map + ord('a')), horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, fontweight='bold')

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
        ax.set_yticks(np.arange(n_model) + 1)
        ax.set_yticklabels(stat_name_array, fontdict={'fontsize': 'medium'})

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if ylabel_styles is not None:
        assert len(ylabel_styles) == len(stat_name_array)
        # https://stackoverflow.com/questions/24617429/matplotlib-different-colors-for-each-axis-label
        for ytick, style in zip(ax.get_yticklabels(), ylabel_styles):
            if style is not None:
                if style in {'bold', 'semibold'}:
                    ytick.set_weight(style)
                elif style in {'italic'}:
                    ytick.set_style(style)
                else:
                    raise NotImplementedError


def show_one_decomposed_scatter(raw_chunks_x, raw_chunks_y, *,
                                ax: Axes = None, xlabel=None, ylabel=None,
                                title=None,
                                color_bias: int = None, letter_map=None):
    if letter_map is not None:
        ax.text(0, 1, chr(letter_map + ord('a')),
                horizontalalignment='left',
                verticalalignment='top', fontweight='bold',
                transform=ax.transAxes)

    # this is for spotlight
    assert len(raw_chunks_x) == len(raw_chunks_y)
    color_list = plt.get_cmap('Set2').colors
    chunk_x_all, chunk_y_all = [], []
    for idx, (raw_chunk_this_x, raw_chunk_this_y) in enumerate(zip(raw_chunks_x, raw_chunks_y)):
        assert raw_chunk_this_x.shape == raw_chunk_this_y.shape == (raw_chunk_this_y.size,)

        # only sample up to 50 points.
        if raw_chunk_this_x.size > 50:
            rng_state_this = np.random.RandomState(seed=0)
            index_rand = rng_state_this.choice(raw_chunk_this_x.size, 50, replace=False)
        else:
            index_rand = slice(None)

        ax.scatter(raw_chunk_this_x[index_rand], raw_chunk_this_y[index_rand],
                   color=color_list[color_bias + idx], alpha=0.5,
                   s=12)
        # show linear regression line.
        fit_this = np.polyfit(raw_chunk_this_x, raw_chunk_this_y, deg=1)

        #         start_end_vector_this = np.array([raw_chunk_this_x.min(),raw_chunk_this_x.max()])
        start_end_vector_this = np.array([0.5, 1])
        ax.plot(start_end_vector_this, (fit_this[0] * (start_end_vector_this - 0.5) * 2 + fit_this[1]) / 2,
                color=color_list[color_bias + idx], linewidth=1)

        #         ax.text(0.95,0.6-idx*0.1, '{:.3f}'.format(fit_this[0]),
        #                 horizontalalignment='right',
        #                 verticalalignment='top', fontsize='medium',
        #                 color=color_list[color_bias+idx])

        chunk_x_all.append(raw_chunk_this_x.copy())
        chunk_y_all.append(raw_chunk_this_y.copy())

        # add text.

    chunk_x_all = np.concatenate(chunk_x_all)
    chunk_y_all = np.concatenate(chunk_y_all)
    fit_this = np.polyfit(chunk_x_all, chunk_y_all, deg=1)
    #     start_end_vector_this = np.array([chunk_x_all.min(),chunk_x_all.max()])
    start_end_vector_this = np.array([0.5, 1])
    # linear transform things values in [0,1] x [0,1] to [0.5,1] x [0,0.5]
    ax.plot(start_end_vector_this, (fit_this[0] * (start_end_vector_this - 0.5) * 2 + fit_this[1]) / 2, color='black',
            linewidth=1, linestyle='--')

    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim(-0.15, 1.1)
    ax.set_ylim(-0.15, 1.1)
    ax.axis('off')

    if title is not None:
        ax.text(0.5, 0.975, title,
                horizontalalignment='center',
                verticalalignment='top', fontsize='medium',
                transform=ax.transAxes)
    if xlabel is not None:
        ax.text(0.5, 0, xlabel,
                horizontalalignment='center',
                verticalalignment='top', fontsize='medium',
                )
    if ylabel is not None:
        ax.text(0, 0.5, ylabel, rotation='vertical',
                horizontalalignment='right',
                verticalalignment='center', fontsize='medium',
                )

    # add pearson stuff
    ax.text(0, 1, '{:.4f}'.format(chunk_y_all.mean()),
            horizontalalignment='left',
            verticalalignment='top', fontsize='medium')
    ax.text(1, 0, '{:.4f}'.format(chunk_x_all.mean()),
            horizontalalignment='right',
            verticalalignment='bottom', fontsize='medium')
    corr_this = pearsonr(chunk_x_all, chunk_y_all)[0]
    r_text = 'n={}\nr={:.2f}'.format(chunk_x_all.size, corr_this)
    ax.text(0, 0.7, r_text, fontsize='medium', horizontalalignment='left')
