"""prepare datasets in train, val, and test
or train, test.

here, I will use stratified split. with middle level labels,
which have probably the correct level of complexity (not too broad nor too fine-grained,
which could cause problem, IMO).


run the following code to get a sense of
"""
import os.path
from itertools import product
from collections import Counter
import numpy as np
import h5py
from sklearn.model_selection import StratifiedShuffleSplit
from tang_jcompneuro.io import neural_dataset_dict, load_image_dataset, load_neural_dataset
from tang_jcompneuro.stimulus_classification import stimulus_label_dict_tang, get_subset_slice
from tang_jcompneuro import dir_dictionary


def main():
    neural_dataset_to_process = ('MkA_Shape',
                                 'MkE2_Shape')

    train_percentage_list = (100, 75, 50, 25)
    has_val_list = (
        True,
        False,
    )
    subset_list = ('all', 'OT')
    seed_list = range(5)
    split_data_file = os.path.join(dir_dictionary['datasets'], 'split_datasets.hdf5')

    # create the file.
    if not os.path.exists(split_data_file):
        with h5py.File(split_data_file):
            pass

    count = 0
    for (neural_dataset_key, subset, has_val, train_percentage, seed) in product(
            neural_dataset_to_process, subset_list, has_val_list, train_percentage_list, seed_list
    ):
        save_one(neural_dataset_key, subset, has_val, train_percentage, split_data_file, seed)
        count += 1
    print(count)


def one_shuffle(labels, test_size, seed):
    shuffler = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    counter = 0
    for train_val_idx, test_idx in shuffler.split(np.empty((labels.size, 1)), labels):
        counter += 1
    assert counter == 1
    # make sure that they cover everything.
    assert np.array_equal(np.sort(np.concatenate((train_val_idx, test_idx))), np.arange(labels.size))
    return np.sort(train_val_idx), np.sort(test_idx)


def _save_one_set(X, y, train_idx, val_idx, test_idx, dataset_name, data_file):
    assert X.ndim == 4 and y.ndim == 2
    assert y.shape[0] == X.shape[0]
    assert X.shape[1:] == (1, 20, 20)

    for (index_this, index_name) in zip((train_idx, val_idx, test_idx), ('train', 'val', 'test')):
        if index_this is None:
            continue
        print(index_name, index_this.size)

        X_this = X[index_this]
        y_this = y[index_this]

        with h5py.File(data_file) as f_out:
            f_out.create_dataset(dataset_name + '/' + index_name + '/' + 'X', data=X_this, compression='gzip')
            f_out.create_dataset(dataset_name + '/' + index_name + '/' + 'y', data=y_this)
            f_out[dataset_name + '/' + index_name].attrs['index'] = index_this


def _output_counter_list(labels):
    assert labels.shape == (labels.size,)
    ctr = Counter(labels)
    counter_vec = []
    for label in sorted(ctr.keys()):
        counter_vec.append(ctr[label])
    counter_vec = np.asarray(counter_vec)
    counter_vec = counter_vec / counter_vec.sum()
    assert np.all(np.isfinite(counter_vec))
    return counter_vec


def _check_idx_sanity(train_idx, val_idx, test_idx, labels, has_val):
    if has_val:
        assert val_idx is not None
        check_list = (train_idx, val_idx, test_idx)
    else:
        assert val_idx is None
        check_list = (train_idx, test_idx)

    # first, let's see proportion of labels.
    counter_vec_ref = _output_counter_list(labels)
    everything = []
    for index_this in check_list:
        assert isinstance(index_this, np.ndarray)
        assert index_this.ndim == 1
        # and they are sorted
        # and they are unique.
        assert np.array_equal(np.unique(index_this), index_this)
        # check in range
        assert index_this.min() >= 0 and index_this.max() < labels.size
        everything.append(index_this.copy())

        # check label proportion
        counter_vec_this = _output_counter_list(labels[index_this])
        assert counter_vec_ref.shape == counter_vec_this.shape
        # this is good enough to account for roundoff errors.
        assert abs(counter_vec_this - counter_vec_ref).max() < 0.01

    # check no overlap
    everything = np.concatenate(everything)
    assert np.unique(everything).shape == everything.shape


def save_one(neural_dataset_key,
             subset, has_val, train_percentage,
             data_file, seed):
    assert isinstance(train_percentage, int) and 1 <= train_percentage <= 100
    assert isinstance(seed, int) and seed >= 0

    dataset_name = '/'.join((neural_dataset_key, subset, 'with_val' if has_val else 'without_val',
                             str(train_percentage), str(seed)))

    with h5py.File(data_file, 'r') as f_out:
        if dataset_name in f_out:
            print(f'{dataset_name} done before')
            return

    print(dataset_name)
    image_dataset_key = neural_dataset_dict[neural_dataset_key]['image_dataset_key']
    subset_slice = get_subset_slice(image_dataset_key, subset)
    labels = stimulus_label_dict_tang[image_dataset_key][('middle', 'str')][subset_slice]
    # create shuffler.
    # 20% for testing, consistent with previous setup (5 fold).
    train_val_idx, test_idx = one_shuffle(labels, 0.2, seed)
    if has_val:
        # use 10% of training for val.
        train_idx, val_idx = one_shuffle(labels[train_val_idx], 0.1, seed)
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]
    else:
        train_idx = train_val_idx
        val_idx = None

    # finally, another shuffle.
    if train_percentage != 100:
        train_idx = train_idx[one_shuffle(labels[train_idx], train_percentage / 100, seed)[1]]

    # ok. test check the counters check.
    # and they don't overlap.

    # finally, load image,
    # and store everything.
    # always use positive version. so that I can fit anything.
    # all these indices are relative to the subset version.
    _check_idx_sanity(train_idx, val_idx, test_idx, labels, has_val)

    # first, load image
    image_this = load_image_dataset(image_dataset_key, trans=True, scale=0.5, subset=subset)
    neural_response_this = load_neural_dataset(neural_dataset_key, return_positive=True, subset=subset)

    _save_one_set(image_this, neural_response_this, train_idx, val_idx, test_idx, dataset_name, data_file)


if __name__ == '__main__':
    main()
