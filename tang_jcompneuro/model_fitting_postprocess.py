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
from collections import OrderedDict
import h5py
from . import dir_dictionary
from .model_fitting import get_num_test_im, get_num_neuron
import numpy as np
from .cell_stats import compute_ccmax
from .cell_classification import get_ready_to_use_classification
from itertools import product
import pandas as pd


def load_data_generic(models_to_examine, *,
                      datasets_to_check=('MkA_Shape', 'MkE2_Shape'),
                      subsets_to_check=('all', 'OT'),
                      load_naive=False,
                      load_coarse=True,
                      avoid_dict=None,
                      percentages_to_load=(100,),
                      seeds=range(2),
                      metric='ccnorm_5',
                      squared=False,
                      score_col_name='score',
                      ):
    """adapted from
    https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/84b04ec342ff58099c1528d80d8f5775a56c9846/results_ipynb/step_1_rough_exploration/model_performance_comparison.ipynb
    """
    if avoid_dict is None:
        avoid_dict = {
            'MkE2_Shape': {('glm', 'linear_softplus')}
        }
    score_all = []
    for dataset, subset in product(datasets_to_check, subsets_to_check):
        print(dataset, subset)
        avoidance = avoid_dict.get(dataset, set())
        for xxx in models_to_examine:
            # print(xxx)
            use_sub = False
            use_sub_test = False
            if len(xxx) == 2:
                model_type, model_subtype = xxx
            elif len(xxx) == 3:
                model_type, model_subtype, use_sub = xxx
                use_sub_test = False
            else:
                assert len(xxx) == 4
                model_type, model_subtype, use_sub, use_sub_test = xxx
            if (model_type, model_subtype) in avoidance:
                continue
            for percentage in percentages_to_load:
                score_new_cc = np.asarray(
                    [load_model_performance(dataset, subset, percentage, s, model_type, model_subtype,
                                            use_sub=use_sub, use_sub_test=use_sub_test, metric=metric,
                                            squared=squared)['corr'] for s in seeds]).mean(axis=0)

                score_all.append({
                    'dataset': dataset,
                    'subset': subset,
                    # I use $, which I never use in my naming, simply to make parsing easier.
                    'model': model_type + '_' + model_subtype + '$' f'{use_sub}/{use_sub_test}',
                    'percentage': percentage,
                    score_col_name: score_new_cc.mean() if load_naive else chunk_neurons(dataset, score_new_cc,
                                                                                  coarse=load_coarse),

                    # later on, I can add neuron subset, etc.
                })
    score_all = pd.DataFrame(score_all,
                             columns=['dataset', 'subset', 'model', 'percentage', score_col_name]).set_index(
        ['dataset', 'subset', 'model', 'percentage'], verify_integrity=True).sort_index()
    return score_all


def get_model_performance_filename(model_type):
    return os.path.join(dir_dictionary['analyses'], model_type + '.hdf5')


_get_ready_to_use_classification_lazy_cache = dict()


def get_ready_to_use_classification_lazy(coarse):
    if coarse not in _get_ready_to_use_classification_lazy_cache:
        _get_ready_to_use_classification_lazy_cache[coarse] = get_ready_to_use_classification(coarse)
    return _get_ready_to_use_classification_lazy_cache[coarse]


def _chunk_neurons_one_level(score, class_dict):
    result = OrderedDict()
    for class_label, class_idx in class_dict.items():
        assert class_idx.shape == score.shape and class_idx.dtype == np.bool_
        data_this_class = score[class_idx]
        result[class_label] = {'raw': data_this_class, 'mean': data_this_class}
    return result


def chunk_neurons(neural_dataset_key, score, coarse=True):
    # basically, load v3 cell classification for this neuron,
    # and return per class score and chunk.
    # remember to deal with duplicate.
    # check
    # https://github.com/leelabcnbc/tang_jcompneuro/blob/d8e3bb719df89765723a25607baf0d4189162cd6/thesis_plots/v1_fitting/comparison_among_cnn_glm_vgg_decomposed_by_fine_subsets.ipynb
    # (maybe handle it somewhere else).
    class_dict = get_ready_to_use_classification_lazy(coarse)[neural_dataset_key]
    if coarse:
        result = _chunk_neurons_one_level(class_dict, score)
    else:
        result = OrderedDict()
        for class_label, class_dict_inner in class_dict.items():
            result[class_label] = _chunk_neurons_one_level(score, class_dict_inner)
    return result


def _load_model_performance_one(neural_dataset_key, subset, percentage, seed, model_type, model_subtype,
                                load_corr,
                                load_y_test_hat,
                                load_corr_val, metric, squared):
    result = {}

    if metric == 'raw':
        divider = 1.0
    elif metric == 'ccnorm_5':
        divider = compute_ccmax(neural_dataset_key, subset, 5)
    else:
        raise NotImplementedError

    with h5py.File(get_model_performance_filename(model_type), 'r') as f:
        grp = '/'.join(str(x) for x in (neural_dataset_key, subset, percentage, seed, model_type, model_subtype))
        grp_this = f[grp]
        if load_corr:
            result['corr'] = grp_this['corr'][...]
            assert np.isscalar(divider) or divider.shape == result['corr'].shape
            result['corr'] /= divider
            if squared:
                result['corr'] **= 2
        if load_y_test_hat:
            result['y_test_hat'] = grp_this['y_test_hat'][...]
        if load_corr_val:
            result['corr_val'] = grp_this['corr_val'][...]
            assert np.isscalar(divider) or divider.shape == result['corr_val'].shape
            result['corr_val'] /= divider
            if squared:
                result['corr_val'] **= 2
    return result


def load_model_performance(neural_dataset_key, subset, percentage, seed, model_type, model_subtype,
                           load_corr=True,
                           load_y_test_hat=False,
                           load_corr_val=True,
                           use_sub=False, use_sub_test=False,
                           metric: str = 'raw', squared=False):
    if not use_sub:
        return _load_model_performance_one(neural_dataset_key, subset, percentage, seed, model_type, model_subtype,
                                           load_corr, load_y_test_hat, load_corr_val, metric, squared)
    else:
        assert '@' not in model_subtype
        result_list = [_load_model_performance_one(neural_dataset_key,
                                                   subset, percentage, seed, model_type, model_subtype + suffix,
                                                   load_corr, load_y_test_hat, load_corr_val, metric, squared
                                                   ) for suffix in ('', '@0.05', '@0.005')]

        # get argmax via corr_val
        corr_val_all = np.array(
            [result_this['corr' if use_sub_test else 'corr_val'] for result_this in result_list])
        assert corr_val_all.ndim == 2 and corr_val_all.shape[0] == 3
        max_model_each = np.argmax(corr_val_all, axis=0)
        assert max_model_each.shape == (corr_val_all.shape[1],)
        # then take max of each one.
        result = {}

        if load_corr:
            result['corr'] = np.array(
                [result_list[model_idx]['corr'][neuron_idx] for neuron_idx, model_idx in enumerate(max_model_each)]
            )
        if load_y_test_hat:
            result['y_test_hat'] = np.concatenate(
                [result_list[model_idx]['y_test_hat'][:, neuron_idx:neuron_idx + 1] for neuron_idx, model_idx in
                 enumerate(max_model_each)], axis=1
            )
        if load_corr_val:
            result['corr_val'] = np.array(
                [result_list[model_idx]['corr_val'][neuron_idx] for neuron_idx, model_idx in enumerate(max_model_each)]
            )

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
            # env['dtype_corr'] = obj['corr'].dtype
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
            # not true.
            # my code can return either float32 or float64
            # when loss=0.0 (float64), regardless whether it's CNN or GLM.
            # assert corr_this.dtype == env['corr'].dtype
            assert np.isnan(env['corr'][neuron_idx])
            env['corr'][neuron_idx] = corr_this

            # extract corr_val
            if model_type == 'cnn':
                corr_val_this = obj.attrs['best_val']
            elif model_type == 'glm':
                corr_val_this = obj['model/corr_val'][()]
            else:
                raise NotImplementedError

            assert np.isfinite(corr_val_this) and np.isscalar(corr_val_this)
            # not true.
            # my code can return either float32 or float64
            # when loss=0.0 (float64), regardless whether it's CNN or GLM.
            # assert corr_this.dtype == env['corr'].dtype
            assert np.isnan(env['corr_val'][neuron_idx])
            env['corr_val'][neuron_idx] = corr_val_this


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
                            dtype=np.float64),
            'corr_val': np.full((num_neuron,), fill_value=np.nan,
                                dtype=np.float64),
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
        assert np.all(np.isfinite(env['corr_val']))

        grp = f_out.create_group(key_str)
        grp.create_dataset('y_test_hat', data=env['y_test_hat'])
        grp.create_dataset('corr', data=env['corr'])
        grp.create_dataset('corr_val', data=env['corr_val'])
        f_out.flush()
        print('done!')


def handle_one_model_type(model_type, filter=None):
    model_type_folder = os.path.join(dir_dictionary['models'], model_type)
    for root, dirs, files in os.walk(model_type_folder):
        ok = any([x.lower().endswith('.hdf5') for x in files])
        if ok:
            root_rel = os.path.relpath(root, os.path.commonpath([root, model_type_folder]))
            if filter is not None and not filter(root_rel):
                continue
            handle_one_folder(model_type, root, files)
            # input('hi')
