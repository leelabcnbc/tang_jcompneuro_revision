"""interface functions for model_fitting of GLMs"""

from itertools import product

import numpy as np
from .glm import glm_wrapper

_actfn_to_train = (
    # 'linear', 'fpower',
    'gqm.2', 'gqm.4', 'gqm.8'
)
_family_to_train = (
    'poisson', 'gaussian',
    # I will run this somewhere else.
    # 'softplus',
)


def _decompose_subtype(subtype):
    subtype_actfn, subtype_family = subtype.split('_')
    assert subtype_actfn in _actfn_to_train
    assert subtype_family in _family_to_train
    return subtype_actfn, subtype_family


def suffix_fn(model_subtype):
    return _decompose_subtype(model_subtype)[0]


def get_trainer(model_subtype):
    # model_subtype has
    family = _decompose_subtype(model_subtype)[1]

    def trainer(datasets):
        y_predict, corr, model = glm_wrapper(datasets, family=family, return_detailed=True)
        return {
            'y_test_hat': y_predict[:, np.newaxis],
            'corr': corr,
            'model': model,
        }

    return trainer


subtype_to_train = (act_fn + '_' + family for act_fn, family in product(_actfn_to_train, _family_to_train))
