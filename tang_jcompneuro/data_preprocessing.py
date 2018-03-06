"""some config vars for data preprocessing."""

import os.path

from . import dir_dictionary

train_percentage_list = (100, 75, 50, 25)
subset_list = ('all', 'OT')
seed_list = range(5)


def split_file_name_gen(suffix=None):
    if suffix is None:
        return os.path.join(dir_dictionary['datasets'], 'split_datasets.hdf5')
    else:
        assert isinstance(suffix, str)
        return os.path.join(dir_dictionary['datasets'], f'split_datasets_{suffix}.hdf5')


split_data_file = split_file_name_gen()
neural_dataset_to_process = ('MkA_Shape',
                             'MkE2_Shape')


def split_dataset_name_gen(neural_dataset_key, subset, has_val, train_percentage, seed):
    dataset_name = '/'.join((neural_dataset_key, subset, 'with_val' if has_val else 'without_val',
                             str(train_percentage), str(seed)))
    return dataset_name
