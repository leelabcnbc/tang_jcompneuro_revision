"""
function version of
https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/53ac1188610c5d73a003b68d6f0ec48fd1f42029/results_ipynb/debug/population_cnn/tang_neuron_cnn.ipynb
"""

from itertools import product
from collections import OrderedDict
import numpy as np
# load cell classification dict
from tang_jcompneuro.cell_classification import get_ready_to_use_classification
from tang_jcompneuro.training_aux import count_params
# remember to scale y to make them have mean around 0.5
from tang_jcompneuro.io import load_split_dataset
from tang_jcompneuro.cnn import CNN

from . import cnn_aux, training_aux
import os.path
from tang_jcompneuro import dir_dictionary
import h5py

# if not, concurrent read/write mode can make h5py fail, sometimes.
cell_class_dict = get_ready_to_use_classification(readonly=True)


def load_data(dataset, image_subset, neuron_subset, seed):
    datasets_all = load_split_dataset(dataset, image_subset, True, slice(None), seed=seed)
    # trim + scale.
    datasets_all_new = []
    for idx, x in enumerate(datasets_all):
        print(x.shape)
        if idx % 2:
            y = x * 2
            y = y[:, cell_class_dict[dataset][neuron_subset]]
            # print(y.shape, y.mean(axis=0))
            # print(np.median(y.mean(axis=0)), y.mean(axis=0).min(), y.mean(axis=0).max())
            # looks good to me.
        else:
            y = x
        datasets_all_new.append(y)
    return tuple(datasets_all_new)


def gen_all_opt_config():
    opt_dict = OrderedDict()
    for scale, loss_type in product((0.01, 0.001, 0.0001, 0.00001, 0.000001), ('poisson', 'mse')):
        opt_config = cnn_aux.get_maskcnn_v1_opt_config(scale=scale,
                                                       bn_scale_nolearning=False,
                                                       smoothness=0.0, group=0.0,
                                                       loss_type=loss_type)
        name_this = loss_type + '_' + str(round(scale * 1000000))
        opt_dict[name_this] = opt_config
    return opt_dict


all_opt_configs = gen_all_opt_config()


def gen_all_arch_config(dataset, image_subset, neuron_subset, debug=False):
    # according to the old ipynb,
    # 118 channels, k1 = 9, k23 = 3 gives roughly same number of parameters
    num_channel_baseline = {
        ('MkA_Shape', 'all', 'OT'): 118,  # this number is the one that gives around same number of params
        # as for k1=9, k23=3.
        ('MkA_Shape', 'all', 'HO'): 106,
    }[dataset, image_subset, neuron_subset]

    num_channel_ratio_dict = OrderedDict(
        [('100', 1), ('75', 0.75), ('50', 0.5), ('25', 0.25)]
    )

    k1_list = (5, 7, 9, 11, 13)
    k2_list = (3,)  # no much time for playing with it. also, this affects number of parameters tremendously.

    arch_config_dict = OrderedDict()

    for (ratio_name, ratio), k1, k2 in product(num_channel_ratio_dict.items(), k1_list, k2_list):
        name_this = f'{k1}_{k2}_{ratio_name}'
        num_channel_this = round(num_channel_baseline * ratio)
        arch_config_this = cnn_aux.get_maskcnn_v1_arch_config(out_channel=num_channel_this, kernel_size_l1=k1,
                                                              kernel_size_l23=k2,
                                                              act_fn='softplus'
                                                              )
        if debug:
            print(name_this, num_channel_this)
        arch_config_dict[name_this] = arch_config_this
    return arch_config_dict


def file_and_key_to_save(model_type: str, model_subtype: str,
                         neural_dataset_key, subset, neuron_subset: str, seed: int):
    assert model_type == 'cnn_population'

    dir_to_save = os.path.join(
        dir_dictionary['models'], model_type, model_subtype,
        neural_dataset_key, subset, str(seed)
    )

    file_name_base = f'{neuron_subset}.hdf5'

    key_to_save = '/'.join([neural_dataset_key, subset, neuron_subset, str(seed), model_type, model_subtype,
                            neuron_subset])
    return dir_to_save, file_name_base, key_to_save


def _train_one(dataset, image_subset, neuron_subset, seed,
               arch_name, opt_name):
    datasets = load_data(dataset, image_subset, neuron_subset, seed)
    results = train_one(gen_all_arch_config(dataset, image_subset, neuron_subset)[arch_name],
                        all_opt_configs[opt_name], datasets)
    # save.
    y_val_cc, y_test_hat, new_cc = results
    # scale back.
    y_test_hat = y_test_hat / 2.0
    assert y_val_cc.ndim == 1 and np.all(np.isfinite(y_val_cc))
    assert new_cc.ndim == 1 and np.all(np.isfinite(new_cc))
    assert y_test_hat.ndim == 2 and np.all(np.isfinite(y_test_hat))
    return y_val_cc, y_test_hat, new_cc


def train_one_wrapper(dataset, image_subset, neuron_subset, seed,
                      arch_name, opt_name, save=True):
    from torch.backends import cudnn
    cudnn.benchmark = True
    cudnn.enabled = True
    if save:
        model_subtype = arch_name + '+' + opt_name
        # then check whether exists or not.
        dir_to_save, file_name_base, key_to_save = file_and_key_to_save('cnn_population', model_subtype,
                                                                        dataset, image_subset, neuron_subset, seed)
        os.makedirs(dir_to_save, exist_ok=True)
        with h5py.File(os.path.join(dir_to_save, file_name_base)) as f_out:
            if key_to_save in f_out:
                print(f'{key_to_save} done before')
            else:
                print(f'{key_to_save} start')
                # train
                y_val_cc, y_test_hat, new_cc = _train_one(dataset, image_subset, neuron_subset, seed,
                                                          arch_name, opt_name)

                # save
                grp = f_out.create_group(key_to_save)
                grp.create_dataset('y_test_hat', data=y_test_hat)
                grp.create_dataset('corr', data=new_cc)
                grp.create_dataset('corr_val', data=y_val_cc)
                print(f'{key_to_save} done')

    else:
        return _train_one(dataset, image_subset, neuron_subset, seed,
                          arch_name, opt_name)


def train_one(arch_config, opt_config, datasets):
    model = CNN(arch_config,
                cnn_aux.v1_maskcnn_generator(),
                input_size=datasets[0].shape[2], n=datasets[1].shape[1],
                seed=0, mean_response=datasets[1].mean(axis=0))
    print('num_param', count_params(model))
    model = model.cuda()
    model = model.train()
    train_results = training_aux.train_one_case(model, datasets, opt_config,
                                                seed=0, eval_loss_type=opt_config['loss'],
                                                show_every=500, return_val_perf=True)
    return train_results
