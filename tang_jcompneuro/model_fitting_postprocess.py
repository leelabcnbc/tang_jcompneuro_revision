"""collect model fitting results

this would essentially give those `_metric` files in previous cases.

However, I will not average over seeds.

that part will be done online, and it should not be slow.

for each

(neural_dataset_key, subset, str(percentage), str(seed), model_type, model_subtype)

I will save three fields.

1. y_test_hat (num_test_im x num_neuron)
2. corr (num_neuron,)

done.

# to speed things up, I should process on a per folder basis.

focus on one particular combination of
(neural_dataset_key, subset, str(percentage), str(seed), model_type, model_subtype)

each time.
this assumes that files for each combination lives in the same folder.
this is always true in my experiments.

"""

import os.path
from functools import partial
import h5py
from . import dir_dictionary
from .model_fitting import get_num_test_im, get_num_neuron
import numpy as np


def get_model_performance_filename(model_type):
    return os.path.join(dir_dictionary['analyses'], model_type + '.hdf5')


def load_model_performance(neural_dataset_key, subset, percentage, seed, model_type, model_subtype,
                           load_corr=True, load_y_test_hat=False):
    result = {}
    with h5py.File(get_model_performance_filename(model_type), 'r') as f:
        grp = '/'.join(str(x) for x in (neural_dataset_key, subset, percentage, seed, model_type, model_subtype))
        grp_this = f[grp]
        if load_corr:
            result['corr'] = grp_this['corr'][...]
        if load_y_test_hat:
            result['y_test_hat'] = grp_this['y_test_hat'][...]

    return result


def _generic_callback(name, obj, env: dict, only_check_key=False):
    if env['stop']:
        return

    if isinstance(obj, h5py.Group) and 'y_test_hat' in obj and 'corr' in obj:
        # not sure if this is relative or absolute
        # I believe it's relative,
        # as .visititems() works on every group. not only h5py.File.
        assert not name.startswith('/')
        (neural_dataset_key, subset, percentage,
         seed, model_type, model_subtype, neuron_idx) = name.split('/')
        seed = int(seed)
        percentage = int(percentage)
        neuron_idx = int(neuron_idx)

        assert env['model_type'] == model_type

        if only_check_key:
            env['key'] = (neural_dataset_key, subset, percentage,
                          seed, model_type, model_subtype)
            env['stop'] = True
            # collect a sample
            env['dtype_y_test_hat'] = obj['y_test_hat'].dtype
            env['dtype_corr'] = obj['corr'].dtype
        else:
            assert env['key'] == (neural_dataset_key, subset, percentage,
                                  seed, model_type, model_subtype)
            assert 0 <= neuron_idx < env['num_neuron']
            y_test_hat_this = obj['y_test_hat'][...]
            assert y_test_hat_this.dtype == env['y_test_hat'].dtype
            assert y_test_hat_this.shape == (env['num_test_im'], 1)
            assert np.all(np.isnan(env['y_test_hat'][:, neuron_idx])), f'neuron {neuron_idx} has duplicate'
            env['y_test_hat'][:, neuron_idx: neuron_idx + 1] = y_test_hat_this
            corr_this = obj['corr'][()]
            assert np.isfinite(corr_this) and np.isscalar(corr_this)
            assert corr_this.dtype == env['corr'].dtype
            assert np.isnan(env['corr'][neuron_idx])
            env['corr'][neuron_idx] = corr_this


def _get_env(file, model_type):
    env = {
        'stop': False,
        'model_type': model_type,
    }
    with h5py.File(file, 'r') as f:
        f.visititems(partial(_generic_callback, env=env, only_check_key=True))

    return env


def handle_one_folder(model_type, root, files):
    # get the key that I should work on.
    # I won't rely on folder structure.
    # instead, I will check one of hdf5 files.
    print(f'handle {root}')
    key = None
    env = None
    for _ in files:
        if _.lower().endswith('.hdf5'):
            env = _get_env(os.path.join(root, _), model_type)
            key = env['key']
            break
    assert key is not None and env is not None

    print(key)
    key_str = '/'.join(str(x) for x in key)

    out_file = get_model_performance_filename(model_type)
    with h5py.File(out_file) as f_out:
        # check if this key exists already.
        if key_str in f_out:
            print('done before!')
            return
        # let's work on every file in this folder,
        # and collect things.
        # since we know the key, we have an idea about shape of everything.
        #
        neural_dataset_key, subset = key[:2]
        num_neuron = get_num_neuron(neural_dataset_key)
        num_test_im = get_num_test_im(neural_dataset_key, subset)

        env = {
            'y_test_hat': np.full((num_test_im, num_neuron), fill_value=np.nan,
                                  dtype=env['dtype_y_test_hat']),
            'corr': np.full((num_neuron,), fill_value=np.nan,
                            dtype=env['dtype_corr']),
            'stop': False,
            'key': key,
            'num_neuron': num_neuron,
            'num_test_im': num_test_im,
            'model_type': model_type,
        }

        for file_to_process in files:
            if file_to_process.lower().endswith('.hdf5'):
                with h5py.File(os.path.join(root, file_to_process), 'r') as f_in:
                    f_in.visititems(partial(_generic_callback, env=env))

        if not np.all(np.isfinite(env['corr'])):
            print('some neurons are missing')
            print(np.flatnonzero(np.isnan(env['corr'])))
        assert np.all(np.isfinite(env['y_test_hat']))
        assert np.all(np.isfinite(env['corr']))

        grp = f_out.create_group(key_str)
        grp.create_dataset('y_test_hat', data=env['y_test_hat'])
        grp.create_dataset('corr', data=env['corr'])
        f_out.flush()
        print('done!')


def handle_one_model_type(model_type):
    model_type_folder = os.path.join(dir_dictionary['models'], model_type)
    for root, dirs, files in os.walk(model_type_folder):
        ok = any([x.lower().endswith('.hdf5') for x in files])
        if ok:
            handle_one_folder(model_type, root, files)
            # input('hi')
