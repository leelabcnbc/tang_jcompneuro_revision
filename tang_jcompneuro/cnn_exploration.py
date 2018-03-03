"""this file saves some initial efforts for exploration of single neuron CNN hyper parameters."""

from collections import OrderedDict
from itertools import product
from copy import deepcopy
import numpy as np
from .cell_classification import compute_cell_classification
from .cell_stats import compute_ccmax
from .configs import cnn_arch, cnn_init


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


def generate_all_fc_config():
    fc_config_dict = OrderedDict()
    fc_config_dict['vanilla'] = cnn_arch.generate_one_fc_config(False, None)
    # standard dropout
    fc_config_dict['dropout'] = cnn_arch.generate_one_fc_config(False, 0.5)
    fc_config_dict['factored'] = cnn_arch.generate_one_fc_config(True)
    return fc_config_dict


def _generate_all_conv_config_generic(num_channel_list, pooling_dict, kernel_size):
    prefix = f'k{kernel_size}'
    conv_dict = OrderedDict()
    for num_channel in num_channel_list:
        for pool_name, pool_config in pooling_dict.items():
            for bn in (
                    True,  # BN doesn't work. probably because too many same-colored patch,
                             # creating zero var batches.
                             # this can lead to numerical instability,
                             # check
                             # increasing bn_eps may help. however, that defeats the purpose of BN.
                           # for details. see
                           #
                    False,
            ):
                bn_prefix = 'bn' if bn else 'nobn'
                conv_dict[f'{prefix}c{num_channel}_{bn_prefix}_{pool_name}'] = cnn_arch.generate_one_conv_config(
                    kernel_size, num_channel, bn=bn, pool=pool_config
                )

    return conv_dict


def _generate_all_conv_config_9x9(num_channel_list):
    # 9x9 gives 12x12 output.
    #    then try output
    #    2x2 pooling (size 8, stride 4) w or w/o dropout
    #    2x2 pooling (size 6, stride 6) w or w/o dropout
    #
    #    4 x 4 pooling (size 6, stride 2) w or w/o dropout
    #    4 x 4 pooling (size 3, stride 3) w or w/o dropout
    #
    #    no pooling. readout. hopefully, things will be similar.
    pooling_dict = OrderedDict()
    # use JCNS paper notation.
    pooling_dict['k8s4max'] = cnn_arch.generate_one_pool_config(8, 4)
    pooling_dict['k6s6max'] = cnn_arch.generate_one_pool_config(6, 6)
    pooling_dict['k6s2max'] = cnn_arch.generate_one_pool_config(6, 2)
    pooling_dict['k3s3max'] = cnn_arch.generate_one_pool_config(3, 3)
    pooling_dict['nopool'] = None

    conv_dict = _generate_all_conv_config_generic(num_channel_list, pooling_dict, 9)

    return conv_dict


def _generate_all_conv_config_13x13(num_channel_list):
    # 13x13 gives 8x8 output.
    #    then try
    #    2x2 pooling (size 4, stride 4) w or w/o dropout
    #    2x2 pooling (size 6, stride 2) w or w/o dropout
    #
    #    4x4 pooling (size 2, stride 2) w or w/o dropout
    #
    #    no pooling.
    pooling_dict = OrderedDict()
    # use JCNS paper notation.
    pooling_dict['k4s4max'] = cnn_arch.generate_one_pool_config(4, 4)
    pooling_dict['k6s2max'] = cnn_arch.generate_one_pool_config(6, 2)
    pooling_dict['k2s2max'] = cnn_arch.generate_one_pool_config(2, 2)
    pooling_dict['nopool'] = None

    conv_dict = _generate_all_conv_config_generic(num_channel_list, pooling_dict, 13)

    return conv_dict


def generate_all_conv_config(num_channel_list_list):
    conv_dict_9 = _generate_all_conv_config_9x9(num_channel_list_list[0])
    conv_dict_13 = _generate_all_conv_config_13x13(num_channel_list_list[1])
    assert conv_dict_9.keys() & conv_dict_13.keys() == set()
    conv_dict_9.update(conv_dict_13)
    return conv_dict_9


def one_layer_models_to_explore():
    num_channel_list = (3, 6, 9, 12, 15)
    num_channel_list_2 = (3, 6, 9)
    fc_dict = generate_all_fc_config()
    conv_dict = generate_all_conv_config([num_channel_list, num_channel_list_2])

    # then let's generate.
    all_config_dict = OrderedDict()

    for (conv_name, conv_config), (fc_name, fc_config) in product(
            conv_dict.items(), fc_dict.items()
    ):
        if (conv_name.endswith('nopool') and fc_name == 'factored') or \
                (not conv_name.endswith('nopool') and fc_name != 'factored'):
            all_config_dict[conv_name + '_' + fc_name] = cnn_arch.generate_one_config(
                [deepcopy(conv_config)], deepcopy(fc_config),
                # 'softplus',  # too slow convergence. maybe using BN without can increase convergence much faster.
                               # maybe that's what NIPS2017 paper does.
                'relu',      # this has much better convergence.
                True
            )

    # useless now. as we use relu instead.
    # if add_old_ones:
    #     # add the reference old one.
    #     assert 'legacy_b12' not in all_config_dict
    #     all_config_dict['legacy_b12'] = cnn_arch.legacy_one_layer_generator(12)

    return all_config_dict

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
    # only try softplus.
    #
    # so in total 5 x 2 x 9 +  3 x 2 x 7 = 132 structures.


def two_layer_models_to_explore():
    # do this after finishing one layer.
    pass


def init_config_to_use_fn():
    return cnn_init.legacy_generator()


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
