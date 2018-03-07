"""overall wrapper of GLM functions"""

# basically, use either R or MATLAB backends to get coefficients,
# and then find the one with highest correlation coefficient on validation set.
# and then return the parameters.

import numpy as np

from .eval import eval_fn_corr_raw


def glm_predict(X, coeff, bias, family):
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert isinstance(coeff, np.ndarray) and coeff.shape == (X.shape[1],)
    assert np.isscalar(bias)

    linear_term = X @ coeff + bias
    if family == 'gaussian':
        return linear_term
    elif family == 'poisson':
        return np.exp(linear_term)
    elif family == 'softplus':
        return np.log(np.exp(linear_term) + 1)
    else:
        raise NotImplementedError


def glm_fit(X, y, backend, **kwargs):
    if backend == 'R':
        from .glm_r import glmnet_interface as glmnet_r
        result = glmnet_r(X, y, **kwargs)
    elif backend == 'MATLAB':
        from .glm_matlab import glmnet_interface as glmnet_matlab
        result = glmnet_matlab(X, y, **kwargs)
    else:
        raise NotImplementedError
    # the interface should have
    # list of lambdas
    # corresponding coefficients
    # corresponding biases.
    check_glm_fit_result(result, X.shape[1])
    return result


def check_glm_fit_result(result, num_feature):
    lambda_list, coeff_list, bias_list = result
    assert isinstance(lambda_list, np.ndarray) and np.all(np.isfinite(lambda_list))
    n_lam = lambda_list.size
    assert n_lam > 0
    assert lambda_list.shape == (n_lam,)
    assert isinstance(coeff_list, np.ndarray) and np.all(np.isfinite(coeff_list))
    assert coeff_list.shape == (n_lam, num_feature)
    assert isinstance(bias_list, np.ndarray) and np.all(np.isfinite(bias_list))
    assert bias_list.shape == (n_lam,)


def _glm_fit_and_select(X_train, y_train, X_val, y_val, backend, debug=False, **kwargs):
    result = glm_fit(X_train, y_train, backend, **kwargs)
    # select the best one.

    lam_list, coeff_list, bias_list = result
    best_result = -np.inf
    best_lam = None
    best_coeff = None
    best_bias = None

    if debug:
        print('train size', X_train.shape, y_train.shape)
        print('val size', X_val.shape, y_val.shape)
        corr_every_one = []

    for lam, coeff, bias in zip(lam_list, coeff_list, bias_list):
        y_val_predict = glm_predict(X_val, coeff, bias, kwargs['family'])
        score_this = eval_fn_corr_raw(y_val_predict[:, np.newaxis], y_val,
                                      data_type=np.float64)
        if score_this > best_result:
            best_result = score_this
            # all these to avoid aliasing.
            best_lam = float(lam)
            best_coeff = coeff.copy()
            best_bias = float(bias)

        if debug:
            corr_every_one.append(score_this)

    if debug:
        from matplotlib import pyplot as plt
        plt.close('all')
        plt.figure()
        plt.plot(lam_list, corr_every_one)
        plt.xlabel('lambda')
        plt.ylabel('corr')
        plt.show()

    assert best_lam is not None and best_coeff is not None and best_bias is not None
    return {
        'lambda': best_lam,
        'coeff': best_coeff,
        'bias': best_bias,
        'corr_val': best_result,
    }


def _check_dataset_shape(X: np.ndarray, y: np.ndarray, n_feature, nonnegative):
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 2 and y.ndim == 2
    # print(X.shape, y.shape)
    assert X.shape[0] == y.shape[0] and X.shape[0] > 0
    assert X.shape[1] == n_feature and y.shape[1] == 1

    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))

    if nonnegative:
        assert np.all(y >= 0)


def glm_wrapper_check_datasets(datasets, family):
    # family is to constrain y.
    # check everything is float64.
    assert len(datasets) == 6
    if family in {'poisson', 'softplus'}:
        nonnegative = True
    elif family in {'gaussian'}:
        nonnegative = False
    else:
        raise NotImplementedError

    n_feature = datasets[0].shape[1]
    _check_dataset_shape(datasets[0], datasets[1], n_feature, nonnegative)
    _check_dataset_shape(datasets[2], datasets[3], n_feature, nonnegative)
    _check_dataset_shape(datasets[4], datasets[5], n_feature, nonnegative)


family_backend_selector = {
    'gaussian': ('R', 'MATLAB'),
    'poisson': ('R', 'MATLAB'),
    'softplus': ('MATLAB',),
}


def glm_wrapper(datasets, *, backend=None, debug=False, return_detailed=False, **kwargs):
    # lambda is always selected automatically.
    assert {'family'} <= kwargs.keys() <= {'alpha', 'standardize', 'family'}
    default_params = {
        'alpha': 1.0,
        'standardize': False,
    }

    params_to_use = default_params
    params_to_use.update(kwargs)
    assert params_to_use.keys() == {'alpha', 'standardize', 'family'}

    if backend is None:
        backend = family_backend_selector[params_to_use['family']][0]
    else:
        assert backend in family_backend_selector[params_to_use['family']]

    # check datasets
    glm_wrapper_check_datasets(datasets, params_to_use['family'])

    # then call glmfit
    result = _glm_fit_and_select(*datasets[:4], backend=backend,
                                 debug=debug, **params_to_use)

    # then get prediction on test set
    if debug:
        print('best lambda', result['lambda'])
        print('test size', datasets[4].shape, datasets[5].shape)

    y_test_predict = glm_predict(datasets[4], result['coeff'],
                                 result['bias'], params_to_use['family'])
    assert np.all(np.isfinite(y_test_predict))
    assert datasets[5].shape == y_test_predict.shape + (1,)

    if not return_detailed:
        return y_test_predict, eval_fn_corr_raw(y_test_predict[:, np.newaxis],
                                                datasets[5], np.float64)
    else:
        return y_test_predict, eval_fn_corr_raw(y_test_predict[:, np.newaxis],
                                                datasets[5], np.float64), result
