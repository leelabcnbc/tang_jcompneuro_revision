"""one variable generalized linear model

a wrapper of R glmnet at <https://cran.r-project.org/web/packages/glmnet/index.html>

the Python one <https://github.com/bbalasub1/glmnet_python/> is not so well-written, with many bugs.

it's easier to use the R version instead.
"""

# old stuff

# """one variable generalized linear model (GLM)
#
# essentially a wrapper of glmnet at <https://github.com/bbalasub1/glmnet_python/>
# """
#
# # somehow, this line will make all glmnet packages available. it will affect namespace globally.
# # bad design.
# # check <https://github.com/bbalasub1/glmnet_python/blob/master/glmnet_python/__init__.py>
# # really bad design.
# import glmnet_python

import numpy as np
from rpy2 import robjects, rinterface
# I don't need this, as I will try to convert everything manually.
# see <https://bitbucket.org/rpy2/rpy2/src/62be5f3e447776bcb808f48cb68cbbcd8a75d4d3/rpy/robjects/numpy2ri.py>
# line 76-82
# import rpy2.robjects.numpy2ri as n2r
from rpy2.rinterface import SexpVector, REALSXP, INTSXP
from sklearn.model_selection import KFold

# load glmnet
r = robjects.r
r.library('glmnet')

_fields_to_convert = {'lambda', 'foldid'}


def convert_to_r_array(x, force_2d=False):
    # see <https://bitbucket.org/rpy2/rpy2/src/62be5f3e447776bcb808f48cb68cbbcd8a75d4d3/rpy/robjects/numpy2ri.py>
    # line 76-82
    # copy to avoid any problem.
    x = np.asarray(x).copy()
    if force_2d:
        assert x.ndim in {1, 2}
        # this is necessary for y variable. otherwise, it would fail for cv.glmnet.
        if x.ndim == 1:
            x = x[:, np.newaxis]
    if x.dtype == np.float64:
        vec = SexpVector(x.ravel("F"), REALSXP)
    elif x.dtype == np.int64:
        vec = SexpVector(x.ravel("F"), INTSXP)
    else:
        raise RuntimeError('not implemented for this dtype {}'.format(x.dtype))
    dim = SexpVector(x.shape, INTSXP)
    return rinterface.baseenv['array'](vec, dim=dim)


def return_converted_fields(dict_in):
    dict_out = dict()
    for name, value in dict_in.items():
        if name in _fields_to_convert:
            value_to_use = convert_to_r_array(value)
        else:
            value_to_use = value
        dict_out[name] = value_to_use
    return dict_out


def glmnet(X, y, **kwargs):
    X = convert_to_r_array(X, force_2d=True)
    y = convert_to_r_array(y, force_2d=True)
    kwargs = return_converted_fields(kwargs)
    # then call it.
    return r['glmnet'](X, y, **kwargs)


def glmnet_cv(X, y, **kwargs):
    X = convert_to_r_array(X, force_2d=True)
    y = convert_to_r_array(y, force_2d=True)
    kwargs = return_converted_fields(kwargs)
    # then call it.
    return r['cv.glmnet'](X, y, **kwargs)


def _fill_and_check_fit_preval(x):
    # I need to check that this all_vals really has good structure.
    # that is, for each row, there is at least one finite value, and all NaN are contiguous on the right.
    assert x.ndim == 2 and x.size > 0
    # simplest thing is to fill in all nan with rightmost finite value, for each row.
    for x_row in x:
        finite_idx = np.flatnonzero(np.isfinite(x_row))
        assert finite_idx.size > 0
        finite_idx = finite_idx.max()
        x_row[finite_idx + 1:] = x_row[finite_idx]
    assert np.all(np.isfinite(x))


def glmnet_cv_fill_preval(fit):
    # use np.array to make sure it's a copy, without shared storage.
    x = np.array(fit.rx2('fit.preval'))
    _fill_and_check_fit_preval(x)
    return x


def glmnet_extract_preval(fit, lambda_to_use, preval=None):
    if preval is None:
        preval = glmnet_cv_fill_preval(fit)

    if isinstance(lambda_to_use, str):
        lambda_to_use = np.array(fit.rx2(lambda_to_use)).ravel()[0]
    assert np.isscalar(lambda_to_use)

    lambda_idx = np.flatnonzero(lambda_to_use == np.array(fit.rx2('lambda')))
    assert lambda_idx.shape == (1,)
    lambda_idx = lambda_idx[0]
    # then extract that column.

    return preval[:, lambda_idx]


def glmnet_coef(fit, s):
    # s can be 'lambda.1se' or 'lambda.min' for a fit obtained from glmnet_cv
    # bias is given first, followed by coefficients
    return np.array(r['as.matrix'](r['coef'](fit, s=s if isinstance(s, str) else convert_to_r_array(s),
                                             exact=False)))


def glmnet_predict(fit, newx, s, type_):
    return np.array(r['predict'](fit, newx=convert_to_r_array(newx, force_2d=True),
                                 s=s if isinstance(s, str) else convert_to_r_array(s),
                                 exact=False, type=type_))


# this is helper function. just to return best glmnet_cv cross-correlated model
# it's the main thing for our neural data fitting.
def glmnet_cv_best_result(X, y, s='lambda.1se', **kwargs):
    if 'keep' not in kwargs:
        kwargs['keep'] = True
    assert kwargs['keep'], 'you must keep data!'
    fit = glmnet_cv(X, y, **kwargs)
    return glmnet_extract_preval(fit, s)


def glmnet_best_result(X, y, **kwargs):
    # use smallest lambda, so that it overfits.
    kwargs['lambda'] = np.asarray([0.0])
    # then call it.
    fit = glmnet(X, y, **kwargs)
    # then predict on itself.
    resp = glmnet_predict(fit, X, [0.0], 'response')
    return resp, fit


def convert_sklearn_kfold_to_glmnet_foldid(N, K, seed=0):
    cv_obj = KFold(n_splits=K, shuffle=True, random_state=seed)
    fold_test_all = np.full(N, fill_value=-1, dtype=np.int64)
    for fold_idx, (train, test) in enumerate(cv_obj.split(np.ones((N, 1)))):
        fold_test_all[test] = fold_idx + 1
    assert np.array_equal(np.arange(K) + 1, np.unique(fold_test_all))
    return fold_test_all
