"""this file saves some initial efforts for exploration of single neuron CNN hyper parameters."""

from collections import OrderedDict
import numpy as np
from .cell_classification import compute_cell_classification
from .cell_stats import compute_ccmax


def neuron_to_explore_idx(each_class_count=2):
    # I will get the highest-ccmax neurons for each of 7 subclasses.
    # this is also an after-thought, given that
    # those with high ccmax turns out to be explained the best.
    # see end of
    # https://github.com/leelabcnbc/tang_jcompneuro/blob/master/results_ipynb/model_comparison_overall.ipynb
    #
    # there, we see that ccnorm^2 \propto ccmax^2.
    # or cc_raw^2/cc_max^2 = a^2 cc_max^2.
    # or cc_raw^2 = a^2 cc_max^4
    # or cc_raw = a cc_max^2.
    #
    #
    # so, typically, higher cc_max would mean better cc_raw, which are good
    # for exploration.
    #
    #

    # will return a dict, indexed by [subset][cell_class]

    # first, load all cell class dict.

    neural_dataset_key = 'MkA_Shape'

    cell_class_dict = compute_cell_classification(neural_dataset_key, 3)
    for y in cell_class_dict.values():
        assert y.shape == (1142,)

    #
    #     'CV': 'curvature',
    #     'CN': 'corner',
    #     'CRS': 'cross',
    #     'Other': 'composition',
    #     'Multi': 'mixed',
    #     'SS': 'classical',
    #     'EB': 'end-stopping',

    overall_dict = OrderedDict()

    for subset in ('OT', 'all'):
        ccmax_this = compute_ccmax(neural_dataset_key, subset=subset)
        assert ccmax_this.shape == (1142,)
        this_dict = OrderedDict()
        for class_to_check_this in ('CV', 'CN', 'CRS', 'Other', 'Multi', 'SS', 'EB'):
            # get index of this class
            ccmax_this_class = ccmax_this[cell_class_dict[class_to_check_this]]
            assert ccmax_this_class.size > each_class_count
            index_this_class = np.flatnonzero(cell_class_dict[class_to_check_this])
            best_one_index = index_this_class[np.argsort(ccmax_this_class)[::-1][:each_class_count]]
            this_dict[class_to_check_this] = OrderedDict(
                [(k, v) for k, v in zip(best_one_index, ccmax_this[best_one_index])])
        overall_dict[subset] = this_dict

    return overall_dict


def one_layer_models_to_explore():
    # return a list of different arch configs
    # to try.
    #
    # num channel 3, 6, 9, 12, 15
    #
    # batchnorm or not
    #
    # vanilla fc (with pooling),
    # vanilla fc + dropout (with pooling),
    # or factored layer.
    #
    # 9x9 kernel.
    # I remembered that stride != 1 is always bad. same for dilation.
    # so maybe just 9x9 kernel and 13x13 kernel.
    # 9x9 gives 12x12 output.
    #    then try output
    #    2x2 pooling (size 8, stride 4) w or w/o dropout
    #    2x2 pooling (size 6, stride 6) w or w/o dropout
    #
    #    4 x 4 pooling (size 6, stride 2) w or w/o dropout
    #    4 x 4 pooling (size 3, stride 3) w or w/o dropout
    #
    #    no pooling. readout. hopefully, things will be similar.
    #
    #
    # 13x13 gives 8x8 output.
    #    then try
    #    2x2 pooling (size 4, stride 4) w or w/o dropout
    #    2x2 pooling (size 6, stride 2) w or w/o dropout
    #
    #    4x4 pooling (size 2, stride 2) w or w/o dropout
    #
    #    no pooling.
    #
    pass
    # so in total 5 x 2 x (9 + 7) = 160 structures.


def two_layer_models_to_explore():
    # do this after finishing one layer.
    pass


def opt_configs_to_explore():
    # simple ones.
    #    L2 all the way, and same
    # complex ones.
    #    L2 (no bias) for conv. copy what I got in simple ones.
    #    L1 (no bias) for fc. try some values and get some reasonable range.
    #
    # also, poisson or mse.
    #    hopefully I don't need to deal with it,
    #    as I want everything to be as simple as possible.
    #
    pass
