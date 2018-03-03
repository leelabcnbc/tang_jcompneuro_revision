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
