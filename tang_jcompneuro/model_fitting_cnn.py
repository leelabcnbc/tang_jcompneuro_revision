"""interface functions for model_fitting of CNNs"""

from itertools import product
from collections import OrderedDict
from copy import deepcopy
import time

import numpy as np
from .configs import cnn_opt, cnn_arch, cnn_init
from .cnn import CNN
from .training_aux import train_one_case


# sets of opt configs to use.
# based on
# https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/results_ipynb/single_neuron_exploration/cnn_initial_exploration.ipynb

def _opt_configs_to_explore_1layer():
    """set of opt configs to use.
    based on
    https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/results_ipynb/single_neuron_exploration/cnn_initial_exploration.ipynb
    """

    def layer_gen(x):
        return cnn_opt.generate_one_layer_opt_config(l1=0.0, l2=x, l1_bias=0.0, l2_bias=0.0)

    # generate all conv stuff.
    conv_dict = OrderedDict()
    conv_dict['1e-3L2'] = [layer_gen(0.001)]
    conv_dict['1e-4L2'] = [layer_gen(0.0001)]

    fc_dict = OrderedDict()
    fc_dict['1e-3L2'] = layer_gen(0.001)

    opt_dict = OrderedDict()
    opt_dict['sgd'] = cnn_opt.generate_one_optimizer_config('sgd')
    opt_dict['adam002'] = cnn_opt.generate_one_optimizer_config('adam', lr=0.002)

    # maybe I should also check out loss
    loss_dict = OrderedDict()
    loss_dict['mse'] = 'mse'

    # not doable, because I don't have a final nonlinearity.
    # loss_dict['poisson'] = 'poisson'

    result_dict = OrderedDict()

    for (conv_name, conv_val), (fc_name, fc_val), (opt_name, opt_val), (loss_name, loss_val) in product(
            conv_dict.items(), fc_dict.items(), opt_dict.items(), loss_dict.items()
    ):
        result_dict[f'{conv_name}_{fc_name}_{opt_name}_{loss_name}'] = cnn_opt.generate_one_opt_config(
            deepcopy(conv_val), deepcopy(fc_val), loss_val, deepcopy(opt_val)
        )

    return result_dict


def gen_on_conv_config_k9(num_channel, pool_config):
    return cnn_arch.generate_one_conv_config(
        9, num_channel, bn=False, pool=pool_config
    )


def _model_configs_to_explore_1layer():
    """set of archs to use.
    based on
    # https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/results_ipynb/single_neuron_exploration/cnn_initial_exploration.ipynb
    """
    num_channel_list = (
        1, 2, 3, 4, 5,
        # 6, 9, 12, 15
    )
    fc_config = cnn_arch.generate_one_fc_config(False, None)
    pool_config = cnn_arch.generate_one_pool_config(6, 2)

    result_dict = OrderedDict()
    for num_channel in num_channel_list:
        # b means baseline
        result_dict[f'b.{num_channel}'] = cnn_arch.generate_one_config(
            [gen_on_conv_config_k9(num_channel, deepcopy(pool_config)),
             ], deepcopy(fc_config), 'relu', True
        )
    return result_dict


def init_config_to_use_fn():
    return cnn_init.legacy_generator()


def get_trainer(model_subtype):
    arch_config = models_to_train[model_subtype]

    def trainer(datasets):
        # best performance in my experiments.
        from torch.backends import cudnn
        cudnn.enabled = True
        cudnn.benchmark = False
        # print(cudnn.enabled, cudnn.benchmark)
        assert cudnn.enabled and not cudnn.benchmark

        best_val = -np.inf
        best_config = None
        best_y_test_hat = None
        best_corr = None
        for opt_config_name, opt_config in opt_configs_to_explore.items():
            # train this config
            model = CNN(arch_config, init_config_to_use_fn(), mean_response=datasets[1].mean(axis=0),
                        seed=0)
            model.cuda()
            t1 = time.time()
            y_val_cc, y_test_hat, new_cc = train_one_case(model, datasets, opt_config, seed=0, show_every=10000000,
                                                          return_val_perf=True)
            t2 = time.time()
            print(opt_config_name, y_val_cc, f'{t2-t1} sec')
            if y_val_cc > best_val:
                best_config = opt_config_name
                best_val = y_val_cc
                best_y_test_hat = y_test_hat
                best_corr = new_cc
        assert best_config is not None and best_y_test_hat is not None and best_corr is not None
        print('best config {} with val {:.6f} and test {:.6f}'.format(best_config, best_val, best_corr))
        return {
            'y_test_hat': best_y_test_hat,
            'corr': best_corr,
            'attrs': {
                'best_val': best_val,
                'best_config': best_config
            },
        }

    return trainer


models_to_train = _model_configs_to_explore_1layer()
opt_configs_to_explore = _opt_configs_to_explore_1layer()
