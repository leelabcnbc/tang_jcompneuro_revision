"""interface functions for model_fitting of GLMs"""

import numpy as np
from .glm import glm_wrapper
from .cnn_pretrained import blob_corresponding_info


def suffix_fn(model_subtype):
    # here I use plus sign in subtype,
    # but / in suffix.
    net_name, _, blob = model_subtype.split('+')
    assert _ == 'legacy'
    return '/'.join([net_name, _, blob])


def get_trainer(model_subtype):
    def trainer(datasets):
        y_predict, corr, model = glm_wrapper(datasets, family='poisson', return_detailed=True)
        return {
            'y_test_hat': y_predict[:, np.newaxis],
            'corr': corr,
            'model': model,
        }

    return trainer


def subtypes_to_train_gen():
    all_subtypes = []
    for net_name, blobs_info in blob_corresponding_info.items():
        # if net_name != 'vgg16_bn':
        #     continue
        for blob in blobs_info.keys():
            all_subtypes.append(
                '+'.join([net_name, 'legacy', blob])
            )
    return all_subtypes


subtype_to_train = subtypes_to_train_gen()
