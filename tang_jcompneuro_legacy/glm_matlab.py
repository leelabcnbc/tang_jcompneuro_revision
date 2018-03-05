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
from sklearn.model_selection import KFold
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
if ~isfield(s, 'lambda')
    s.lambda = [];
end
% then add path; genpath is not needed.
addpath('{matlab_path}');
% then call
[predicted_y, B, FitInfo, devsum] = glmnet_cv_best_result(s.X, s.y, ...
    s.standardize, s.family, double(s.foldid), double(s.alpha), false, ...
    s.lambda);
save('{file_to_save}', 'predicted_y', 'B', 'FitInfo');

""".strip()


# this is helper function. just to return best glmnet_cv cross-correlated model
# it's the main thing for our neural data fitting.
def glmnet_cv_best_result(X, y, *, debug=False, **kwargs):
    # first create temp file to save results.
    assert {'standardize', 'alpha',
            'foldid', 'family'} <= kwargs.keys() <= {'standardize', 'alpha',
                                                     'foldid', 'family', 'lambda'}
    assert X.ndim == 2 and y.ndim == 2
    assert y.shape == (X.shape[0], 1)
    # print(X.shape, y.shape)
    # print(X.std(axis=0))
    # print(y.std(), y.mean())
    #
    # print(kwargs)

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
        a = check_output([__matlab_bin, '-nosplash', '-nodisplay'], encoding='utf-8',
                         input=script, stderr=STDOUT)
        print(a)
        predicted_y = loadmat(file_to_save)['predicted_y'].ravel()
        if debug:
            mat_return = loadmat(file_to_save)
        else:
            mat_return = None
    assert predicted_y.shape == (y.size,)

    # print(predicted_y.shape)
    # print(predicted_y.std(), predicted_y.mean())

    if debug:
        return predicted_y, mat_return
    else:
        return predicted_y


def convert_sklearn_kfold_to_glmnet_foldid(N, K, seed=0):
    cv_obj = KFold(n_splits=K, shuffle=True, random_state=seed)
    fold_test_all = np.full(N, fill_value=-1, dtype=np.int64)
    for fold_idx, (train, test) in enumerate(cv_obj.split(np.ones((N, 1)))):
        fold_test_all[test] = fold_idx + 1
    assert np.array_equal(np.arange(K) + 1, np.unique(fold_test_all))
    return fold_test_all
