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
from scipy.io import savemat, loadmat
from tempfile import TemporaryDirectory
import os.path
from subprocess import check_output, STDOUT
from tang_jcompneuro import dir_dictionary

# __matlab_bin = '/Applications/MATLAB_R2017b.app/bin/matlab'
# __matlab_bin = '/opt/matlab/8.6/bin/matlab'
__matlab_bin = '/usr/local/MATLAB/R2016b/bin/matlab'
__matlab_path = os.path.join(dir_dictionary['package'], '..', 'matlab')

demo_script = """
s = load('{file_to_save}');
% then add path; genpath is not needed.
addpath('{matlab_path}');
% then call
[B, FitInfo] = glmnet(s.X, s.y, s.standardize, s.family, double(s.alpha));
save('{file_to_save}', 'B', 'FitInfo');

""".strip()

glmnet_lambda_seq_script = """
s = load('{file_to_save}');
% then add path; genpath is not needed.
addpath('{matlab_path}');
% then call
[lambda] = glmnet_lambda_sequence(s.X, s.y, s.standardize, s.family, double(s.alpha), s.numlambda);
save('{file_to_save}', 'lambda');

""".strip()


# this is helper function. just to return best glmnet_cv cross-correlated model
# it's the main thing for our neural data fitting.
def glmnet_lambda_sequence(X, y, *, nlambda=100, **kwargs):
    assert {'standardize', 'alpha', 'family'} == kwargs.keys()

    assert X.ndim == 2 and y.ndim == 2
    assert y.shape == (X.shape[0], 1)

    with TemporaryDirectory() as dir_name:
        file_to_save = os.path.join(dir_name, 'input.mat')
        mat_to_save = {
            'X': X,
            'y': y,
        }
        mat_to_save.update(kwargs)
        mat_to_save.update({'numlambda': nlambda})
        # save X, y, and all kwargs.
        savemat(file_to_save, mat_to_save)
        # savemat('/tmp/hahaha.mat', mat_to_save)
        script = glmnet_lambda_seq_script.format(file_to_save=file_to_save,
                                                 matlab_path=__matlab_path)
        # print(script)
        check_output([__matlab_bin, '-nosplash', '-nodisplay'], encoding='utf-8',
                     input=script, stderr=STDOUT)
        # print(a)
        # match R glmnet style.
        lambda_seq = loadmat(file_to_save)['lambda'].ravel()[::-1]
    assert lambda_seq.shape == (nlambda,)
    assert np.all(np.isfinite(lambda_seq))

    return lambda_seq


def _glmnet(X, y, **kwargs):
    # first create temp file to save results.
    assert {'standardize', 'alpha', 'family'} == kwargs.keys()
    assert X.ndim == 2 and y.ndim == 2
    assert y.shape == (X.shape[0], 1)

    with TemporaryDirectory() as dir_name:
        file_to_save = os.path.join(dir_name, 'input.mat')
        mat_to_save = {
            'X': X,
            'y': y,
        }
        mat_to_save.update(kwargs)
        # save X, y, and all kwargs.
        savemat(file_to_save, mat_to_save)
        # savemat('/tmp/hahaha.mat', mat_to_save)
        script = demo_script.format(file_to_save=file_to_save,
                                    matlab_path=__matlab_path)
        # print(script)
        check_output([__matlab_bin, '-nosplash', '-nodisplay'], encoding='utf-8',
                     input=script, stderr=STDOUT)
        result = loadmat(file_to_save)
        coeff_matrix, fit_info = result['B'], result['FitInfo']

    return coeff_matrix, fit_info


def glmnet_interface(X, y, *, alpha, standardize, family):
    coeff_matrix, fit_info = _glmnet(X, y, alpha=alpha, standardize=standardize, family=family)
    # the order match R version.
    coef_seq = coeff_matrix.T[::-1]
    bias_seq = fit_info['Intercept'][0, 0].ravel()[::-1]
    lam_seq = fit_info['Lambda'][0, 0].ravel()[::-1]

    return lam_seq, coef_seq, bias_seq
