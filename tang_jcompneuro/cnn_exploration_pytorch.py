import torch
import torch.cuda
from collections import OrderedDict
import os.path
import time

import h5py
import numpy as np

from . import dir_dictionary
from .cnn import CNN
from .training_aux import train_one_case, eval_fn
from .cnn_exploration import init_config_to_use_fn, one_layer_models_to_explore, opt_configs_to_explore


def _load_dataset(dataset_key, neuron_idx, subset):
    group_to_use = OrderedDict()
    group_to_use['new'] = f'/{dataset_key}/{subset}/with_val/100/0'

    result = OrderedDict()

    datafile = os.path.join(dir_dictionary['datasets'], 'split_datasets.hdf5')
    with h5py.File(datafile, 'r') as f:
        for k, g in group_to_use.items():
            g_this = f[g]
            # load X_train/test/val
            # load y_train/test/val
            X_train = g_this['train/X'][...]
            y_train = g_this['train/y'][:, neuron_idx:neuron_idx + 1]
            X_test = g_this['test/X'][...]
            y_test = g_this['test/y'][:, neuron_idx:neuron_idx + 1]

            X_val = g_this['val/X'][...] if 'val' in g_this else None
            y_val = g_this['val/y'][:, neuron_idx:neuron_idx + 1] if 'val' in g_this else None
            result[k] = (X_train, y_train, X_test, y_test, X_val, y_val)

    return result


def do_inner(arch_config, datasets, f_out: h5py.File, opt_configs_to_test, key_prefix, note):
    for opt_name, opt_config in opt_configs_to_test.items():

        key_this = '/'.join(key_prefix + (opt_name,))

        if key_this not in f_out:

            model = CNN(arch_config, init_config_to_use_fn(), mean_response=datasets[1].mean(axis=0))
            model.cuda()
            t1 = time.time()
            y_test_hat, new_cc = train_one_case(model, datasets, opt_config, seed=0, show_every=10000000)
            t2 = time.time()
            # check that this is really the case.
            new_cc_debug = eval_fn(y_test_hat, datasets[3].astype(np.float32))['corr']
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
        datasets = _load_dataset(dataset_key, neuron_idx, subset)['new']
        # each loss config.
        key_prefix = (arch_name, dataset_key, subset, str(neuron_idx))
        do_inner(arch_config_list[arch_name], datasets,
                 f_out, opt_configs_to_test, key_prefix, note)
