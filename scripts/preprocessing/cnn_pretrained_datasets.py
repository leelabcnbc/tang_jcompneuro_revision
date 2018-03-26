"""generate GLM-ready datasets, from split_datasets"""

from itertools import product

import h5py
import numpy as np
import os.path
from tang_jcompneuro import dir_dictionary
from tang_jcompneuro.data_preprocessing import (subset_list,
                                                split_data_file, neural_dataset_to_process,
                                                split_dataset_name_gen, split_file_name_gen)
from tang_jcompneuro.io import neural_dataset_dict
from tang_jcompneuro.glm_data_preprocessing import GLMDataPreprocesser, max_total_dim

from sklearn.decomposition import PCA
from tang_jcompneuro.cnn_pretrained import blob_corresponding_info


def get_q_model_pca_trans(x_flat_all, max_total_dim_this):
    # check
    # https://github.com/leelabcnbc/tang_jcompneuro/blob/master/scripts/neuron_fitting/cnn_feature_legacy/dataset_preprocessing.py#L46-L62
    old_shape = x_flat_all.shape
    x_flat_all = x_flat_all.reshape(x_flat_all.shape[0], -1)
    num_feature = x_flat_all.shape[1]
    pca_feature = min(num_feature,
                      max_total_dim_this,
                      x_flat_all.shape[0])

    assert pca_feature > 1
    print(old_shape, x_flat_all.shape, 'to', pca_feature)
    pca_obj = PCA(svd_solver='randomized', n_components=pca_feature,
                  random_state=0)
    pca_obj.fit(x_flat_all)

    print('preserved variance:', pca_obj.explained_variance_ratio_.sum())

    def transformer(x):
        assert x.ndim == 2
        return pca_obj.transform(x)

    return transformer, pca_obj.explained_variance_ratio_.copy()


class CNNPreprocessor(GLMDataPreprocesser):
    # let's first try 1032 for debugging purpose.
    def __init__(self, max_total_dim_this):
        super().__init__(check_original_data=False)
        self.max_total_dim = max_total_dim_this

    def get_transformer(self, X_train_no_val_full):
        # first, perform FP transformation.
        # and then return the remixing one.

        transformer_q_pca, explaind_var_ratio_q = get_q_model_pca_trans(X_train_no_val_full,
                                                                        self.max_total_dim)

        def transformer(X):
            # t1 = time.time()
            X = X.reshape(X.shape[0], -1)
            # t2 = time.time()
            combined = transformer_q_pca(X)
            return combined

        self.transformer = transformer
        self.per_dim_var = explaind_var_ratio_q


def handle_one_case_outer(key_to_extract, f_in_idx: h5py.File, f_in_feature: h5py.File, f_out: h5py.File):
    # seed is not needed either.
    # specify only first two, to speed things up.
    # lower layers take a lot of time.
    for neural_dataset_key, subset, seed in product(neural_dataset_to_process, subset_list, range(2)):
        handle_one_case(neural_dataset_key, subset, seed, key_to_extract, f_in_idx, f_in_feature, f_out)


def handle_one_case(neural_dataset_key, subset, seed, key_to_extract,
                    f_in_idx: h5py.File, f_in_feature: h5py.File, f_out: h5py.File):
    print(f'handle {neural_dataset_key}/{subset}/{seed} for {key_to_extract}')
    dataset_main_name = split_dataset_name_gen(neural_dataset_key, subset, False, 100, seed)
    #
    # # then get transformer.
    # # check
    # # https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/b48bcb7d18bddba032163bba8d4f63f08b054844/results_ipynb/debug/glm_preprocessing.ipynb
    train_set_idx = f_in_idx[dataset_main_name]['train'].attrs['index']
    # sorted and unique.
    assert np.array_equal(np.unique(train_set_idx), train_set_idx)
    # get data
    img_key = neural_dataset_dict[neural_dataset_key]['image_dataset_key']
    # use float64 all the way to get higher precision, and make things compatible with glmnet.
    features_all = f_in_feature[f'{img_key}/{key_to_extract}'][...].astype(np.float64, copy=False)
    print(features_all.shape)
    assert np.all(features_all >= 0)
    print('indeed non negative')
    #
    transformer_this = CNNPreprocessor(max_total_dim_this=max_total_dim)
    # this should always be a copy, due to use of fancy indexing.
    transformer_this.get_transformer(features_all[train_set_idx])
    #
    # ok. let's apply to every one.
    # don't need 25, 50, 75.
    # need to keep no val. otherwise, no place to store attrs in the last line.
    for has_val, train_percentage in product((True, False), (100,)):
        handle_one_case_inner(neural_dataset_key, subset, has_val, train_percentage,
                              seed, f_in_idx, features_all, f_out, transformer_this)

    # finally, for dataset_main_name,
    # save per_dim_var in it.
    f_out[dataset_main_name].attrs['per_dim_var'] = transformer_this.per_dim_var


def handle_one_case_inner(neural_dataset_key, subset, has_val, train_percentage,
                          seed, f_in_idx: h5py.File, features_all: np.ndarray,
                          f_out: h5py.File, transformer: CNNPreprocessor):
    dataset_main_name = split_dataset_name_gen(neural_dataset_key, subset, has_val, train_percentage, seed)

    print(f'handle {dataset_main_name}')

    sets_to_handle = ('train', 'val', 'test') if has_val else ('train', 'test')

    for set_to_handle_this in sets_to_handle:
        data_to_save = dataset_main_name + f'/{set_to_handle_this}/X'
        if data_to_save not in f_out:
            set_original_idx = f_in_idx[dataset_main_name][set_to_handle_this].attrs['index']
            assert np.array_equal(np.unique(set_original_idx), set_original_idx)
            set_original = features_all[set_original_idx]
            set_transformed = transformer.transform(set_original)
            # then save
            f_out.create_dataset(data_to_save, data=set_transformed)
            f_out.flush()
            print(f'{set_to_handle_this} done')
        else:
            print(f'{set_to_handle_this} done before')


def main():
    with h5py.File(split_data_file, 'r') as f_in_idx, h5py.File(
            os.path.join(dir_dictionary['features'], 'cnn_feature_extraction.hdf5'), 'r') as f_in_feature:
        # I will handle each net, each blob separately.
        #
        #

        # for name, config in transformer_dict.items():
        #     print(f'========== handle {name} ==========')
        #     with h5py.File(split_file_name_gen(name)) as f_out:
        #         handle_one_case_outer(config, f_in_idx, f_out)
        for net_name, blob_names in blob_corresponding_info.items():
            blob_names_keys = list(blob_names.keys())
            for blob_idx, blob_name in enumerate(blob_names_keys):
                with h5py.File(split_file_name_gen('/'.join([net_name, 'legacy', blob_name]))) as f_out:
                    handle_one_case_outer('/'.join([net_name, 'legacy', str(blob_idx)]),
                                          f_in_idx, f_in_feature, f_out)


if __name__ == '__main__':
    main()
