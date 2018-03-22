"""interface functions for model_fitting of CNNs"""

from itertools import product
from collections import OrderedDict
from copy import deepcopy
import time
import json

import numpy as np
from .configs import cnn_opt, cnn_arch, cnn_init
from .cnn import CNN
from .training_aux import train_one_case, count_params


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
        1, 2,
        3,
        4, 5,
        7, 8,
        10, 11,
        6,
        9,
        12,
        15,
        18,
    )
    fc_config = cnn_arch.generate_one_fc_config(False, None)
    pool_config_max = cnn_arch.generate_one_pool_config(6, 2)
    pool_config_avg = cnn_arch.generate_one_pool_config(6, 2, pool_type='avg')

    pool_dict = [(None, pool_config_max),
                 ('avg', pool_config_avg)]

    channel_detail = 9

    result_dict = OrderedDict()
    for num_channel, (pool_name, pool_config), act_fn in product(num_channel_list,
                                                                 pool_dict, ('relu', None, 'halfsq',
                                                                             'sq', 'abs')):

        if pool_name is None:
            name_this = f'b.{num_channel}'
        else:
            assert isinstance(pool_name, str)
            name_this = f'b.{num_channel}_{pool_name}'

        if act_fn is None:
            name_this = f'{name_this}_linear'
        elif act_fn != 'relu':
            name_this = f'{name_this}_{act_fn}'

        # b means baseline
        if num_channel != channel_detail and name_this != f'b.{num_channel}':
            # I won't check it.
            continue
        result_dict[name_this] = cnn_arch.generate_one_config(
            [gen_on_conv_config_k9(num_channel, deepcopy(pool_config)),
             ], deepcopy(fc_config), act_fn, True
        )
    # finally, add MLP stuff

    for k in (4,  # so that we have roughly 144 units.
              20, 40, 60, 80, 100, 120,
              145,  # > 95% variance preserved.
            # check
    # https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/results_ipynb/debug/cnn/cnn_wrapper.ipynb
              ):
        name_this = f'mlp.{k}'  # k is dim to keep.
        # this is because baseline model has 883 parameters.
        # (k+1)*mlp + mlp + 1 = 883
        # (k + 2) * mlp = 882
        mlp_this = 882 // (k + 2)
        result_dict[name_this] = cnn_arch.generate_one_config(
            [], cnn_arch.generate_one_fc_config(False, None, mlp_this), 'relu', True
        )

    return result_dict


def init_config_to_use_fn():
    return cnn_init.legacy_generator()


def get_trainer(model_subtype, cudnn_enabled=True, cudnn_benchmark=False,
                show_every=10000000, show_arch_config=False,
                max_epoch=20000):
    if '@' in model_subtype:
        model_subtype_real, scale_hack = model_subtype.split('@')
        scale_hack = float(scale_hack)
        assert scale_hack in {0.05, 0.005}
    else:
        model_subtype_real = model_subtype
        scale_hack = None

    arch_config = models_to_train[model_subtype_real]

    if show_arch_config:
        print(model_subtype_real, 'scale hack', scale_hack)
        print(json.dumps(arch_config, indent=2))

    def trainer(datasets):
        # best performance in my experiments.
        from torch.backends import cudnn
        cudnn.enabled = cudnn_enabled
        cudnn.benchmark = cudnn_benchmark
        # print(cudnn.enabled, cudnn.benchmark)
        assert cudnn.enabled == cudnn_enabled and cudnn.benchmark == cudnn_benchmark

        best_val = -np.inf
        best_config = None
        best_y_test_hat = None
        best_corr = None
        inf_counter = 0

        # check input size
        if 'mlp' not in model_subtype_real:
            assert datasets[0].ndim == 4 and datasets[0].shape[1:] == (1, 20, 20)
            input_size = 20
        else:
            assert datasets[0].ndim == 2
            input_size = (datasets[0].shape[1], 1)

        if show_arch_config:
            # show dataset detail
            assert len(datasets) == 6
            print([x.shape for x in datasets])

        for opt_config_name, opt_config in opt_configs_to_explore.items():
            # train this config
            # print('seed changed')
            # print('scale hacked')
            model = CNN(arch_config, init_config_to_use_fn(), mean_response=datasets[1].mean(axis=0),
                        # change seed if you get unlucky for unstable input...
                        # this is the case especially for MkE2_Shape.
                        # i think this was an issue before as well.
                        # except that pytorch 0.2.0 doesn't report such errors.
                        # check /inf_debug_script.py
                        # seed=42,
                        seed=0,
                        # last ditch
                        # for some avg_sq
                        # scale_hack=0.9,
                        # for other avg_sq
                        # as well as other models.
                        scale_hack=scale_hack,
                        # for MLP model, use PCAed data.
                        input_size=input_size,
                        # scale_hack = 0.0
                        )
            if show_arch_config:
                print(model)
                print('# of params', count_params(model))
            # print('change trainer seed')
            model.cuda()
            t1 = time.time()
            try:
                y_val_cc, y_test_hat, new_cc = train_one_case(model, datasets, opt_config,
                                                              seed=0, show_every=show_every,
                                                              return_val_perf=True,
                                                              max_epoch=max_epoch)
            except RuntimeError as e:
                # just zero.
                if e.args == ('value cannot be converted to type double without overflow: inf',):
                    y_val_cc = 0.0
                    new_cc = 0.0
                    y_test_hat = np.zeros_like(datasets[3], dtype=np.float32)
                    inf_counter += 1
                else:
                    # print('we will not handle it')
                    raise
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
                'best_config': best_config,
                # use this to check how many such tragedies happen.
                'inf_counter': inf_counter,
            },
        }

    return trainer


models_to_train = _model_configs_to_explore_1layer()
models_to_train_detailed_keys = [x for x in models_to_train if x.startswith('b.9')]
models_to_train_mlp = [x for x in models_to_train if x.startswith('mlp.')]
opt_configs_to_explore = _opt_configs_to_explore_1layer()
