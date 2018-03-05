# this file contains aux functions needed to fit GLMs from CNN features.
# it's using features in https://github.com/leelabcnbc/tang-paper-2017/blob/master/feature_extraction/feature_extraction.ipynb
# or `os.path.join(dir_dictionary['features'], 'cnn_feature_extraction.hdf5')`
# it's legacy in the sense that
# 1) feature extraction details are not well thought out, compared to reference below.
# 2) GLM models are pretty vanilla, without all bells and whistles on regularization.
#
# it's legacy w.r.t. this one.
# Deep convolutional models improve predictions of macaque V1 responses to natural images
# https://doi.org/10.1101/201764
#
# in any case, this legacy one is a good baseline, or good supplement for my current thesis proposal.

# whole structure of this file follow cnn.py
# with arch, subarch, opt
from collections import OrderedDict
import h5py
from .glm_matlab_aux import one_train_loop_cv as one_train_loop_cv_glm
# from tang_jcompneuro.stimulus_classification import get_subset_slice
import os.path
from tang_jcompneuro import dir_dictionary

# this corresponds to training_tool_params for one_train_loop_cv in glm.aux
named_opt_dict = {
    'L1_poisson': {'family': 'poisson',
                   'reg': 'L1', },
    # 'L1_gaussian': {'family': 'gaussian',
    #                 'reg': 'L1'},
    'L1_softplus': {'family': 'softplus',
                    'reg': 'L1', },
    # 'L1_gaussian': {'family': 'gaussian',
    #                 'reg': 'L1'},
}

one_train_loop_cv = one_train_loop_cv_glm

feature_file_legacy_original = '/home/yimengzh/data2/tang-paper-2017/results/features/cnn_feature_extraction.hdf5'
feature_file_legacy = os.path.join(dir_dictionary['datasets'], 'cnn_feature_extraction_pca_legacy.hdf5')

network_layer_dict = OrderedDict()
network_layer_dict['vgg16'] = [
    'conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2',
    'pool2', 'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
    'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1',
    'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7'
]


# network_layer_dict['alexnet'] = ['conv1', 'norm1', 'pool1', 'conv2', 'norm2', 'pool2', 'conv3', 'conv4', 'conv5',
#                                  'pool5', 'fc6', 'fc7']


def load_feature_dataset(dataset_key, subset, network, setting, layer):
    assert isinstance(layer, int)
    layer_str = str(layer)

    # subset_slice = get_subset_slice(dataset_key, subset)

    with h5py.File(feature_file_legacy, 'r') as f:
        x = f[f'{dataset_key}/{network}/{setting}/{layer_str}/{subset}'][...]
    assert x.ndim == 2
    return x


def _cnn_params_to_string_opt(opt_param: dict, strict=True) -> str:
    for (k, v) in named_opt_dict.items():
        if v == opt_param:
            return k
    if not strict:
        return '<undefined>'
    else:
        raise NotImplementedError('no such opt named yet!')


def cnn_params_to_string_tuple(architecture_param: str, submodel_param, opt_param,
                               strict=True) -> (str, str, str):
    architecture_param_str = architecture_param
    assert len(submodel_param) == 2
    # I should get back the layer number.
    submodel_param_str = (submodel_param[0], '{:02}'.format(submodel_param[1]),
                          network_layer_dict[architecture_param][submodel_param[1]])
    submodel_param_str = ','.join(submodel_param_str)
    opt_param_str = _cnn_params_to_string_opt(opt_param, strict=strict)

    return architecture_param_str, submodel_param_str, opt_param_str
