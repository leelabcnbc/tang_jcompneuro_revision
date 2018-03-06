"""generate GLM-ready datasets, from split_datasets"""

from itertools import product

import h5py

from tang_jcompneuro.data_preprocessing import (train_percentage_list, subset_list, seed_list,
                                                split_data_file, neural_dataset_to_process,
                                                split_dataset_name_gen, split_file_name_gen)
from tang_jcompneuro.glm_data_preprocessing import generate_ready_transformer_dict, GLMDataPreprocesser


def handle_one_case_outer(transformer_config, f_in: h5py.File, f_out: h5py.File):
    for neural_dataset_key, subset, seed in product(neural_dataset_to_process, subset_list, seed_list):
        handle_one_case(neural_dataset_key, subset, seed, transformer_config, f_in, f_out)


def handle_one_case(neural_dataset_key, subset, seed, transformer_config,
                    f_in: h5py.File, f_out: h5py.File):
    print(f'handle {neural_dataset_key}/{subset}/{seed}')
    dataset_main_name = split_dataset_name_gen(neural_dataset_key, subset, False, 100, seed)

    # then get transformer.
    # check
    # https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/b48bcb7d18bddba032163bba8d4f63f08b054844/results_ipynb/debug/glm_preprocessing.ipynb
    train_set_this = f_in[dataset_main_name]['train/X'][...]

    class_this, kwargs = transformer_config
    transformer_this: GLMDataPreprocesser = class_this(**kwargs)
    transformer_this.get_transformer(train_set_this)

    # ok. let's apply to every one.
    for has_val, train_percentage in product((True, False), train_percentage_list):
        handle_one_case_inner(neural_dataset_key, subset, has_val, train_percentage,
                              seed, f_in, f_out, transformer_this)

    # finally, for dataset_main_name,
    # save per_dim_var in it.
    f_out[dataset_main_name].attrs['per_dim_var'] = transformer_this.per_dim_var


def handle_one_case_inner(neural_dataset_key, subset, has_val, train_percentage, seed,
                          f_in: h5py.File, f_out: h5py.File, transformer: GLMDataPreprocesser):
    dataset_main_name = split_dataset_name_gen(neural_dataset_key, subset, has_val, train_percentage, seed)

    print(f'handle {dataset_main_name}')

    sets_to_handle = ('train', 'val', 'test') if has_val else ('train', 'test')

    for set_to_handle_this in sets_to_handle:
        data_to_save = dataset_main_name + f'/{set_to_handle_this}/X'
        if data_to_save not in f_out:
            set_original = f_in[dataset_main_name][f'{set_to_handle_this}/X'][...]
            set_transformed = transformer.transform(set_original)
            # then save
            f_out.create_dataset(data_to_save, data=set_transformed)
            f_out.flush()
            print(f'{set_to_handle_this} done')
        else:
            print(f'{set_to_handle_this} done before')


def main():
    transformer_dict = generate_ready_transformer_dict()
    with h5py.File(split_data_file, 'r') as f_in:
        for name, config in transformer_dict.items():
            print(f'========== handle {name} ==========')
            with h5py.File(split_file_name_gen(name)) as f_out:
                handle_one_case_outer(config, f_in, f_out)


if __name__ == '__main__':
    main()
