import torch
import torch.cuda
import os.path
import time
from collections import OrderedDict

import h5py
import numpy as np

from . import dir_dictionary
from .cnn import CNN
from .training_aux import train_one_case, eval_fn, count_params
from .cnn_exploration import (init_config_to_use_fn,
                              one_layer_models_to_explore,
                              opt_configs_to_explore,
                              load_dataset,
                              two_layer_models_to_explore)


def do_inner(arch_config, datasets_fn, f_out: h5py.File, opt_configs_to_test, key_prefix, note):
    for opt_name, opt_config in opt_configs_to_test.items():

        key_this = '/'.join(key_prefix + (opt_name,))

        if key_this not in f_out:

            model = CNN(arch_config, init_config_to_use_fn(), mean_response=datasets_fn()[1].mean(axis=0),
                        seed=0)
            model.cuda()
            t1 = time.time()
            y_test_hat, new_cc = train_one_case(model, datasets_fn(), opt_config, seed=0, show_every=10000000)
            t2 = time.time()
            # check that this is really the case.
            new_cc_debug = eval_fn(y_test_hat, datasets_fn()[3].astype(np.float32))['corr']
            assert new_cc == new_cc_debug
            # train new.
            del model
            f_out.create_dataset(key_this, data=y_test_hat)
            f_out[key_this].attrs['corr'] = new_cc
            f_out[key_this].attrs['note'] = np.string_(note)
            f_out[key_this].attrs['time'] = t2 - t1
            f_out.flush()
        else:
            print(f'{key_this} done before')

        print(key_this, f_out[key_this].attrs['corr'], f"{f_out[key_this].attrs['time']} @ {note}")


def explore_one_neuron_1L(arch_name, neuron_idx, subset, dataset_key='MkA_Shape'):
    # https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/76cb8d906ee3625eed9894d2b3317678291a4470/results_ipynb/single_neuron_exploration/debug/neuron_553.ipynb
    #
    #
    # stored in /results/models/cnn_exploration/arch_name/dataset_key/neuron_idx.hdf5
    # will try out all subsets and opt configs.
    #
    # fetch card info.
    # https://stackoverflow.com/a/48152675/3692822
    # https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu/48152675
    #

    note = torch.cuda.get_device_name(torch.cuda.current_device())

    arch_config_list = one_layer_models_to_explore()
    assert arch_name in arch_config_list
    opt_configs_to_test = opt_configs_to_explore()

    dir_to_save = os.path.join(dir_dictionary['models'], 'cnn_exploration', arch_name, dataset_key, subset)
    os.makedirs(dir_to_save, exist_ok=True)
    file_to_save = os.path.join(dir_to_save, str(neuron_idx) + '.hdf5')
    print(file_to_save)
    with h5py.File(file_to_save) as f_out:
        cache_dict = {'cache': None}

        def datasets_fn():
            if cache_dict['cache'] is None:
                cache_dict['cache'] = load_dataset(dataset_key, neuron_idx, subset)['new']
            return cache_dict['cache']

        # each loss config.
        key_prefix = (arch_name, dataset_key, subset, str(neuron_idx))
        do_inner(arch_config_list[arch_name], datasets_fn,
                 f_out, opt_configs_to_test, key_prefix, note)


def explore_one_neuron_2L(arch_name, neuron_idx, subset, dataset_key='MkA_Shape'):
    # https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/76cb8d906ee3625eed9894d2b3317678291a4470/results_ipynb/single_neuron_exploration/debug/neuron_553.ipynb
    #
    #
    # stored in /results/models/cnn_exploration/arch_name/dataset_key/neuron_idx.hdf5
    # will try out all subsets and opt configs.
    #
    # fetch card info.
    # https://stackoverflow.com/a/48152675/3692822
    # https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu/48152675
    #

    note = torch.cuda.get_device_name(torch.cuda.current_device())

    arch_config_list = two_layer_models_to_explore()
    assert arch_name in arch_config_list
    opt_configs_to_test = opt_configs_to_explore(2)

    dir_to_save = os.path.join(dir_dictionary['models'], 'cnn_exploration_2L', arch_name, dataset_key, subset)
    os.makedirs(dir_to_save, exist_ok=True)
    file_to_save = os.path.join(dir_to_save, str(neuron_idx) + '.hdf5')
    print(file_to_save)
    with h5py.File(file_to_save) as f_out:
        cache_dict = {'cache': None}

        def datasets_fn():
            if cache_dict['cache'] is None:
                cache_dict['cache'] = load_dataset(dataset_key, neuron_idx, subset)['new']
            return cache_dict['cache']

        # each loss config.
        key_prefix = (arch_name, dataset_key, subset, str(neuron_idx))
        do_inner(arch_config_list[arch_name], datasets_fn,
                 f_out, opt_configs_to_test, key_prefix, note)


def get_num_params():
    count_dict = OrderedDict()
    init_config = init_config_to_use_fn()
    arch_config_list = one_layer_models_to_explore()
    assert len(arch_config_list) == 132 // 2  # since BN is removed.
    for config_name, config in arch_config_list.items():
        model_this = CNN(config, init_config)
        count_dict[config_name] = count_params(model_this)
    return count_dict
