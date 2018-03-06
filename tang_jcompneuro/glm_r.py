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

# load glmnet
r = robjects.r
r.library('glmnet')


def _convert_to_r_array(x, force_2d=False):
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


def _glmnet(X, y, **kwargs):
    X = _convert_to_r_array(X, force_2d=True)
    y = _convert_to_r_array(y, force_2d=True)
    # then call it.
    return r['glmnet'](X, y, **kwargs)


def _glmnet_coef(fit, return_bias=False):
    # by default, this extracts all coef.
    # first column is bias. (assuming we always fit bias).
    if not return_bias:
        return np.array(r['as.matrix'](r['coef'](fit)))[1:]
    else:
        return np.array(r['as.matrix'](r['coef'](fit)))


def _glmnet_lambda(fit):
    return np.array(fit.rx2('lambda'))


def _glmnet_bias(fit):
    return np.array(fit.rx2('a0'))


def _glmnet_predict(fit, newx, s, type_='response'):
    return np.array(r['predict'](fit, newx=_convert_to_r_array(newx, force_2d=True),
                                 s=s if isinstance(s, str) else _convert_to_r_array(s),
                                 exact=False, type=type_))


def glmnet_interface(X, y, *, alpha, standardize, family):
    fit = _glmnet(X, y, alpha=alpha, standardize=standardize, family=family)
    coef_seq = _glmnet_coef(fit).T
    bias_seq = _glmnet_bias(fit)
    lam_seq = _glmnet_lambda(fit)

    return lam_seq, coef_seq, bias_seq
