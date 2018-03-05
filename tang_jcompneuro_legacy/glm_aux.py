import os.path
from collections import OrderedDict

import h5py

from strflab.feature_transformation import quadratic_features
# from . import dir_dictionary
from .glm import glmnet_cv_best_result, convert_sklearn_kfold_to_glmnet_foldid
# from .stimulus_classification import get_reference_subset_list
from scipy.stats import pearsonr
# from .io import neural_dataset_dict, load_image_dataset, load_neural_dataset
# from .fitting_util import num_fold_global

num_fold_global = 5
import numpy as np
from scipy.signal import hann
from scipy.fftpack import fft2


def hann_2d(size):
    # I use False version, since FFT will be applied.
    window_1d = hann(size, sym=False)[np.newaxis]
    window_2d = window_1d.T @ window_1d
    window_2d /= window_2d.sum()
    return window_2d


# make sure it's there.
# os.makedirs(os.path.join(dir_dictionary['datasets'], 'glm'), exist_ok=True)
#
# subsets_to_work_on = get_reference_subset_list()
# file_to_save_dict = OrderedDict(
#     [(subset, os.path.join(dir_dictionary['datasets'], 'glm',
#                            f'glm_fitting_preprocessed_{subset}.hdf5')) for subset in subsets_to_work_on]
# )

# scale_and_locality_dict = OrderedDict()
# # not needed any more, as I find 0.25 underperforms 0.5 most of the time.
# # see <https://github.com/leelabcnbc/tang-paper-2017/blob/master/neuron_fitting/model_comparison_all.ipynb>
# # scale_and_locality_dict[0.25] = ('fpower', -1, 0, 2, 4, 8)
# scale_and_locality_dict[0.5] = ('fpower', -1, 0, 2, 4, 8)
# datasets_to_work_on = ('MkA_Shape', 'MkE2_Shape')


# def scale_locality_name(scale, locality):
#     if locality == -1:
#         loc_name = 'linear'
#     else:
#         # works for both fpower and numbered ones.
#         loc_name = f'{locality}'
#
#     if scale == 0.25:
#         scale_name = 'quarter'
#     elif scale == 0.5:
#         scale_name = 'half'
#     else:
#         raise ValueError('unsupported scale!')
#
#     return '/'.join([scale_name, loc_name])
#
#
# def get_power_model(X):
#     N, c, h, w = X.shape
#     # assert N in {9500, 4605, 800, 1600, 9500 - 1600,
#     #              4605 - 800} and c == 1 and h == w
#     assert c == 1 and h == w
#     assert h in {10, 20}
#     X_windowed = X * hann_2d(h)
#     X_fft = abs(fft2(X_windowed)) ** 2
#     assert X_fft.shape == X.shape == X_windowed.shape
#     assert np.all(np.isfinite(X_fft))
#     return X_fft

#
# def load_X_and_y_by_name(subset, dataset, scale, locality, method):
#     # these names are names already in str. not those in `scale_and_locality_dict`.
#     # those scale_and_locality_dict are just internal names.
#     X_name_to_fetch = '/'.join([dataset, scale, locality, method])
#     y_name_to_fetch = dataset + '/y'
#     # print(X_name_to_fetch, y_name_to_fetch)
#     file_to_fetch = file_to_save_dict[subset]
#     with h5py.File(file_to_fetch, 'r') as f_out:
#         X = f_out[X_name_to_fetch][...]
#         y = f_out[y_name_to_fetch][...]
#     return X, y

#
# def load_data_at_this_scale_and_locality(dataset, scale, loc, subset,
#                                          return_original_size=False):
#     # X, y = load_data_for_cnn_fitting(dataset, scale=scale, subset=subset)
#     image_key = neural_dataset_dict[dataset]['image_dataset_key']
#     X = load_image_dataset(image_key, trans=True, scale=scale, subset=subset)
#     y = load_neural_dataset(dataset, return_positive=True, subset=subset)
#     print(X.shape, y.shape)
#
#     # let's do some PCA on X.
#     x_flat = X.reshape(len(X), -1)
#     if (not isinstance(loc, str)) and loc >= 0:
#         X_flat_q = quadratic_features(X, locality=(0, loc, loc))
#         # let's check amount of variance contributed by X_flat and X_flat_q
#         var_original = np.var(x_flat, axis=0).sum()
#         var_additional = np.var(X_flat_q, axis=0).sum()
#         print('var L: {:.4f}, Q: {:.4f}, percentage of Q in Q+L: {:.2f}'.format(
#             var_original, var_additional, var_additional / (var_original + var_additional)
#         ))
#         x_flat_all = np.concatenate((x_flat, X_flat_q), axis=1)
#     elif loc == -1:
#         x_flat_all = x_flat
#     elif loc == 'fpower':  # this is the Fourier power model
#         x_flat_all = get_power_model(X).reshape(len(X), -1)
#     else:
#         # I can do fourier power model as well.
#         raise NotImplementedError('not supported loc!')
#
#     # notice that I will have different bias added for different subset. but it should be fine,
#     # for cross-validation purpose.
#     assert x_flat_all.dtype == np.float64
#
#     if not return_original_size:
#         return x_flat_all, y
#     else:
#         return x_flat_all, y, x_flat.shape[1]
#

def compute_all_r(y_true, y_predict, foldid):
    assert np.array_equal(np.unique(foldid), np.arange(num_fold_global) + 1)
    # print(y_true.shape, y_predict.shape, foldid.shape)
    assert y_true.shape == y_predict.shape == foldid.shape == (foldid.size,)
    r_all = []
    for fold in np.arange(num_fold_global):
        mask_this = (foldid == fold + 1)
        r_all.append(pearsonr(y_true[mask_this],
                              y_predict[mask_this])[0])
    r_all = np.array(r_all)
    return r_all


def one_train_loop_cv(X, y, cv_seed, training_tool_params):
    assert training_tool_params.keys() >= {'reg', 'family'}
    family = training_tool_params['family']
    assert family in {'gaussian', 'poisson'}
    reg = training_tool_params['reg']
    if reg == 'L1':
        alpha_value = 1.0
        standardize = False
    elif reg == 'L2':
        alpha_value = 0.0
        standardize = False
    # add these to try more variations on regularization.
    elif reg == 'L1_ns':
        alpha_value = 1.0
        standardize = True
    elif reg == 'L2_ns':
        alpha_value = 0.0
        standardize = True
    else:
        raise NotImplementedError('wrong value {}'.format(reg))
    foldid_to_use = convert_sklearn_kfold_to_glmnet_foldid(len(X), num_fold_global, seed=cv_seed)
    # then pass in everything.
    # I think standardize = False makes sense, as scale in PCAed data carry information
    y_this_predicted_debug = glmnet_cv_best_result(X, y, s='lambda.min',
                                                   standardize=standardize, type_measure='mse',
                                                   family=family,
                                                   foldid=foldid_to_use, alpha=alpha_value)
    # then let's compute mean r2.
    if family == 'poisson':
        y_to_return = np.exp(y_this_predicted_debug)
    elif family == 'gaussian':
        y_to_return = y_this_predicted_debug
    else:
        raise NotImplementedError('no such family')
    r_all = compute_all_r(y.ravel(), y_to_return, foldid_to_use)
    return y_to_return, r_all
