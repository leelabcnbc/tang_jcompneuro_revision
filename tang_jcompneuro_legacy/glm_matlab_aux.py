import os.path
from collections import OrderedDict

import h5py

from .glm_matlab import glmnet_cv_best_result, convert_sklearn_kfold_to_glmnet_foldid
from scipy.stats import pearsonr
# from .fitting_util import num_fold_global
num_fold_global = 5
import numpy as np


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
    assert family in {'gaussian', 'poisson', 'softplus'}
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

    y_to_return = glmnet_cv_best_result(X, y, standardize=standardize,
                                        family=family,
                                        foldid=foldid_to_use, alpha=alpha_value)

    # # then let's compute mean r2.
    # if family == 'poisson':
    #     y_to_return = np.exp(y_this_predicted_debug)
    # elif family == 'gaussian':
    #     y_to_return = y_this_predicted_debug
    # else:
    #     raise NotImplementedError('no such family')
    r_all = compute_all_r(y.ravel(), y_to_return, foldid_to_use)
    return y_to_return, r_all
