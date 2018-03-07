"""generic training helpers.

training.py and training_aux.py are mostly for CNN.
"""
import os.path
import stat
import os
from shlex import quote
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import h5py
import numpy as np
import os.path
import time
from . import dir_dictionary
from .io import get_num_neuron_all_datasets
from . import data_preprocessing
from .model_fitting_glm import suffix_fn as suffix_fn_glm, get_trainer as get_trainer_glm
from .io import load_split_dataset
from subprocess import run
from itertools import product

validation_dict = {
    'cnn': True,
    'glm': True,
    # 'gabor': False,
}

switch_val_test_dict = {
    'cnn': False,
    'glm': True,
}

suffix_fn_dict = {
    'cnn': lambda x: None,
    'glm': lambda x: suffix_fn_glm(x),
    # 'gabor': lambda x: None
}

split_steps_fn_dict = {
    'cnn': lambda x: 50,
    'glm': lambda x: 100,
    # 'gabor': lambda x: 25,
}

# what portions of datasets to train.
training_portions_fn_dict = {
    # only train one seed first.
    'cnn': lambda x: {'seed_list': range(1)},
    'glm': lambda x: {},
}

chunk_dict = {
    'cnn': 5,
    'glm': None,
}

assert (validation_dict.keys() == suffix_fn_dict.keys() ==
        split_steps_fn_dict.keys() == training_portions_fn_dict.keys() ==
        switch_val_test_dict.keys() == chunk_dict.keys())

_cache_vars = {'num_neuron_dict': None}


def get_num_neuron_dict():
    if _cache_vars['num_neuron_dict'] is None:
        _cache_vars['num_neuron_dict'] = get_num_neuron_all_datasets()
    return _cache_vars['num_neuron_dict']


def get_trainer(model_type, model_subtype):
    # fetch the training funcs, which takes in a dataset and returns three things.
    # 1. y_test_hat
    # 2. corr
    # 3. (optional) attrs. a dict. to store some attributes.
    # 4. (optional) model. a dict. to create a subgroup called 'model' to store additional things.
    if model_type == 'glm':
        trainer = get_trainer_glm(model_subtype)
    else:
        raise NotImplementedError

    return trainer


def dataset_spec_encode(neural_dataset_key, subset, percentage: int, seed: int):
    encoded = '@'.join([neural_dataset_key, subset, str(percentage), str(seed)])
    assert encoded == quote(encoded)

    # check decode
    assert dataset_spec_decode(encoded) == (neural_dataset_key, subset, percentage, seed)

    return encoded


def dataset_spec_decode(encoded):
    neural_dataset_key, subset, percentage, seed = encoded.split('@')
    percentage = int(percentage)
    seed = int(seed)
    return neural_dataset_key, subset, percentage, seed


def neuron_spec_encode(neuron_start, neuron_end):
    encoded = '-'.join([str(neuron_start), str(neuron_end)])
    assert encoded == quote(encoded)
    assert neuron_spec_decode(encoded) == (neuron_start, neuron_end)
    # check decode
    return encoded


def neuron_spec_decode(encoded):
    neuron_start, neuron_end = encoded.split('-')
    return int(neuron_start), int(neuron_end)


def get_data_one_slice(datasets_all, idx_relative):
    new_datasets = []
    for idx, x_or_y in enumerate(datasets_all):
        if idx % 2 == 0:
            new_datasets.append(x_or_y)
        else:
            if x_or_y is None:
                new_datasets.append(x_or_y)
            else:
                new_datasets.append(x_or_y[:, idx_relative:idx_relative + 1])

    return tuple(new_datasets)


def train_one_case_generic_save_data(train_result: dict, key_this: str, f_out: h5py.File):
    assert {'y_test_hat', 'corr'} <= train_result.keys() <= {'y_test_hat', 'corr', 'attrs', 'model'}
    # save
    y_test_hat = train_result['y_test_hat']
    assert y_test_hat.ndim == 2 and y_test_hat.shape[1] == 1
    grp_this = f_out.create_group(key_this)
    grp_this.create_dataset('y_test_hat', data=y_test_hat)
    assert np.isscalar(train_result['corr']) and np.isfinite(train_result['corr'])
    grp_this.create_dataset('corr', data=train_result['corr'])
    if 'attrs' in train_result:
        # save attrs
        for k, v in train_result['attrs'].items():
            grp_this.attrs['k'] = v
    if 'model' in train_result:
        grp_this_model = grp_this.create_group('model')
        for k_model, v_model in train_result['model'].items():
            grp_this_model.create_dataset(k_model, data=v_model)

    f_out.flush()


def train_one_case_generic(model_type, model_subtype, dataset_spec, neuron_spec):
    # this will be the function that slurm scripts will call.
    #
    # dataset_spec will be a string that is bash safe.
    # so is neuron_spec.

    neural_dataset_key, subset, percentage, seed = dataset_spec_decode(dataset_spec)
    neuron_start, neuron_end = neuron_spec_decode(neuron_spec)

    dir_to_save, file_name_base, key_to_save = file_and_key_to_save(model_type, model_subtype,
                                                                    neural_dataset_key, subset, percentage, seed,
                                                                    neuron_start, neuron_end)
    neuron_range = slice(neuron_start, neuron_end)
    os.makedirs(dir_to_save, exist_ok=True)
    with h5py.File(os.path.join(dir_to_save, file_name_base)) as f_out:
        trainer = get_trainer(model_type, model_subtype)
        # get dataset
        datasets_all = load_split_dataset(neural_dataset_key, subset, validation_dict[model_type],
                                          neuron_range, percentage=percentage,
                                          seed=seed, last_val=not switch_val_test_dict[model_type],
                                          suffix=suffix_fn_dict[model_type](model_subtype))
        # then training one by one.
        for neuron_idx_relative, neuron_idx_real in enumerate(range(neuron_start, neuron_end)):
            key_this = key_to_save + '/' + str(neuron_idx_real)
            if key_this in f_out:
                print(f'{key_this} done before')
            else:
                print(f'{key_this} start')
                t1 = time.time()
                datasets_this = get_data_one_slice(datasets_all, neuron_idx_relative)
                train_result = trainer(datasets_this)
                train_one_case_generic_save_data(train_result, key_this, f_out)
                t2 = time.time()
                print(f'{key_this} @ {t2-t1}sec')


def generate_one_script(header, model_type, model_subtype, dataset_spec, neuron_spec_or_spec_list):
    template_function_middle = f"""
. activate tf15
# wait for a while. otherwise, it may not work... maybe some bug of conda.
sleep 2
cd {dir_dictionary['root']}
. ./setup_env_variables.sh
""".strip()

    # https://stackoverflow.com/questions/8577027/how-to-declare-a-long-string-in-python
    # https://stackoverflow.com/questions/363223/how-do-i-get-both-stdout-and-stderr-to-go-to-the-terminal-and-a-log-file
    template_function_inner = ("PYTHONUNBUFFERED=1 python scripts/model_fitting/fitting_master.py "
                               f"{{model_type}} {{model_subtype}} {{dataset_spec}} {{neuron_spec}} "
                               f"2>&1 | tee {dir_dictionary['root']}/trash/"
                               f"model_fitting_{{model_type}}_{{model_subtype}}_{{dataset_spec}}_{{neuron_spec}}"
                               ).strip()

    assert isinstance(model_type, str) and quote(model_type) == model_type
    assert isinstance(model_subtype, str) and quote(model_subtype) == model_subtype

    script_to_run = header + '\n' + template_function_middle + '\n\n\n'
    if isinstance(neuron_spec_or_spec_list, str):
        # then simple.
        script_to_run += template_function_inner.format(
            model_type=model_type, model_subtype=model_subtype,
            dataset_spec=dataset_spec, neuron_spec=neuron_spec_or_spec_list
        ) + '\n'
    else:
        for neuron_spec in neuron_spec_or_spec_list:
            script_to_run += template_function_inner.format(
                model_type=model_type, model_subtype=model_subtype,
                dataset_spec=dataset_spec, neuron_spec=neuron_spec
            ) + ' &\n'
        script_to_run += 'wait\n'

    return script_to_run


# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def generate_all_scripts(header, model_type, model_subtype_list):
    """this is what those _sub files call. they provide header and subtypes.

    it will return a list of scripts.

    and then, the script should either run these scripts one by one,
    or sbatch them
    """
    chunk_option = chunk_dict[model_type]
    script_dict = OrderedDict()
    for model_subtype in model_subtype_list:
        # generate all datasets
        seed_list = training_portions_fn_dict[model_type](model_subtype).get('seed_list',
                                                                             data_preprocessing.seed_list)
        subset_list = training_portions_fn_dict[model_type](model_subtype).get('subset_list',
                                                                               data_preprocessing.subset_list)
        neural_dataset_to_process = training_portions_fn_dict[model_type](model_subtype).get(
            'neural_dataset_to_process',
            data_preprocessing.neural_dataset_to_process)
        train_percentage_list = training_portions_fn_dict[model_type](model_subtype).get(
            'train_percentage_list', data_preprocessing.train_percentage_list
        )

        for neural_dataset_key, subset, percentage, seed in product(
                neural_dataset_to_process, subset_list, train_percentage_list, seed_list
        ):
            dataset_spec = dataset_spec_encode(neural_dataset_key, subset, percentage, seed)
            # generate chunks to process.
            neuron_fitting_pairs = get_neuron_fitting_pairs(get_num_neuron_all_datasets()[neural_dataset_key],
                                                            split_steps_fn_dict[model_type](model_subtype))

            # convert every one into specs.
            neuron_fitting_pairs = [neuron_spec_encode(x, y) for (x, y) in neuron_fitting_pairs]

            if chunk_option is not None:
                specs_to_iter = chunker(neuron_fitting_pairs, chunk_option)
            else:
                specs_to_iter = neuron_fitting_pairs

            for neuron_spec in specs_to_iter:
                if isinstance(neuron_spec, str):
                    script_name = (model_subtype, dataset_spec, neuron_spec)
                else:
                    script_name = (model_subtype, dataset_spec, '&'.join(neuron_spec))
                script_dict[script_name] = generate_one_script(header,
                                                               model_type,
                                                               model_subtype,
                                                               dataset_spec,
                                                               neuron_spec)

    return script_dict


def run_all_scripts(script_dict, slurm=True):
    """this is another function that those _sub files should call. this actually execute files"""
    if slurm:
        trash_global = os.path.join(dir_dictionary['root'], 'trash')
        os.chdir(trash_global)

    for script_name, script_content in script_dict.items():
        # make sure it will run.
        assert script_content.startswith('#!/usr/bin/env bash\n')
        file_temp = NamedTemporaryFile(delete=False)
        file_temp.write(script_content.encode('utf-8'))
        file_temp.close()
        print(script_name, 'start')
        if not slurm:
            os.chmod(file_temp.name, stat.S_IEXEC)
            # then run it.
            run(file_temp.name)
        else:
            run(['sbatch', file_temp.name], check=True)
        os.remove(file_temp.name)
        print(script_name, 'done')


def get_neuron_fitting_pairs(n_neuron, step):
    result = []
    assert n_neuron > 0
    for x in range(0, n_neuron, step):
        y = min(n_neuron, x + step)
        result.append((x, y))

    return result


def file_and_key_to_save(model_type: str, model_subtype: str,
                         neural_dataset_key, subset, percentage: int, seed: int,
                         start_neuron, end_neuron):
    assert model_type in validation_dict

    dir_to_save = os.path.join(
        dir_dictionary['models'], model_type, model_subtype,
        neural_dataset_key, subset, str(percentage), str(seed)
    )

    file_name_base = f'{start_neuron}_{end_neuron}.hdf5'

    key_to_save = '/'.join([neural_dataset_key, subset, str(percentage), str(seed), model_type, model_subtype])
    return dir_to_save, file_name_base, key_to_save
