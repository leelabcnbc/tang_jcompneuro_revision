"""handles feature extraction"""
# rewrite of https://github.com/leelabcnbc/tang-paper-2017/blob/master/tang_2017/feature_extraction.py
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn

from leelabtoolbox.feature_extraction.cnn import (cnnsizehelper, generic_network_definitions)
from leelabtoolbox.preprocessing import pipeline

from collections import defaultdict, OrderedDict
from torch import nn
from torch.autograd import Variable
from functools import partial
from skimage.transform import rescale
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import FloatTensor
import h5py


def blobinfo():
    # copied from
    # https://github.com/leelabcnbc/leelab-toolbox/blob/fe57c8577993c9c9883eee1ca0b527cb8300226f/leelabtoolbox/feature_extraction/cnn/pytorch_network_definitions.py
    blob_corresponding_info_inner = dict()
    blob_corresponding_info_inner['vgg16'] = OrderedDict([('conv1_1', 'features.1'),
                                                          ('conv1_2', 'features.3'),
                                                          ('pool1', 'features.4'),
                                                          ('conv2_1', 'features.6'),
                                                          ('conv2_2', 'features.8'),
                                                          ('pool2', 'features.9'),
                                                          ('conv3_1', 'features.11'),
                                                          ('conv3_2', 'features.13'),
                                                          ('conv3_3', 'features.15'),
                                                          ('pool3', 'features.16'),
                                                          ('conv4_1', 'features.18'),
                                                          ('conv4_2', 'features.20'),
                                                          ('conv4_3', 'features.22'),
                                                          ('pool4', 'features.23'),
                                                          ('conv5_1', 'features.25'),
                                                          ('conv5_2', 'features.27'),
                                                          ('conv5_3', 'features.29'),
                                                          ('pool5', 'features.30'),
                                                          ('fc6', 'classifier.1'),
                                                          ('fc7', 'classifier.4')])

    blob_corresponding_info_inner['vgg16_bn'] = OrderedDict([('conv1_1', 'features.2'),
                                                             ('conv1_2', 'features.5'),
                                                             ('pool1', 'features.6'),
                                                             ('conv2_1', 'features.9'),
                                                             ('conv2_2', 'features.12'),
                                                             ('pool2', 'features.13'),
                                                             ('conv3_1', 'features.16'),
                                                             ('conv3_2', 'features.19'),
                                                             ('conv3_3', 'features.22'),
                                                             ('pool3', 'features.23'),
                                                             ('conv4_1', 'features.26'),
                                                             ('conv4_2', 'features.29'),
                                                             ('conv4_3', 'features.32'),
                                                             ('pool4', 'features.33'),
                                                             ('conv5_1', 'features.36'),
                                                             ('conv5_2', 'features.39'),
                                                             ('conv5_3', 'features.42'),
                                                             ('pool5', 'features.43'),
                                                             ('fc6', 'classifier.1'),
                                                             ('fc7', 'classifier.4')])

    blob_corresponding_info_inner['vgg19'] = OrderedDict([('conv1_1', 'features.1'),
                                                          ('conv1_2', 'features.3'),
                                                          ('pool1', 'features.4'),
                                                          ('conv2_1', 'features.6'),
                                                          ('conv2_2', 'features.8'),
                                                          ('pool2', 'features.9'),
                                                          ('conv3_1', 'features.11'),
                                                          ('conv3_2', 'features.13'),
                                                          ('conv3_3', 'features.15'),
                                                          ('conv3_4', 'features.17'),
                                                          ('pool3', 'features.18'),
                                                          ('conv4_1', 'features.20'),
                                                          ('conv4_2', 'features.22'),
                                                          ('conv4_3', 'features.24'),
                                                          ('conv4_4', 'features.26'),
                                                          ('pool4', 'features.27'),
                                                          ('conv5_1', 'features.29'),
                                                          ('conv5_2', 'features.31'),
                                                          ('conv5_3', 'features.33'),
                                                          ('conv5_4', 'features.35'),
                                                          ('pool5', 'features.36'),
                                                          ('fc6', 'classifier.1'),
                                                          ('fc7', 'classifier.4')])

    blob_corresponding_info_inner['vgg19_bn'] = OrderedDict([('conv1_1', 'features.2'),
                                                             ('conv1_2', 'features.5'),
                                                             ('pool1', 'features.6'),
                                                             ('conv2_1', 'features.9'),
                                                             ('conv2_2', 'features.12'),
                                                             ('pool2', 'features.13'),
                                                             ('conv3_1', 'features.16'),
                                                             ('conv3_2', 'features.19'),
                                                             ('conv3_3', 'features.22'),
                                                             ('conv3_4', 'features.25'),
                                                             ('pool3', 'features.26'),
                                                             ('conv4_1', 'features.29'),
                                                             ('conv4_2', 'features.32'),
                                                             ('conv4_3', 'features.35'),
                                                             ('conv4_4', 'features.38'),
                                                             ('pool4', 'features.39'),
                                                             ('conv5_1', 'features.42'),
                                                             ('conv5_2', 'features.45'),
                                                             ('conv5_3', 'features.48'),
                                                             ('conv5_4', 'features.51'),
                                                             ('pool5', 'features.52'),
                                                             ('fc6', 'classifier.1'),
                                                             ('fc7', 'classifier.4')])

    blob_corresponding_reverse_info_inner = dict()
    for net_name, net_info in blob_corresponding_info_inner.items():
        blob_corresponding_reverse_info_inner[net_name] = OrderedDict()
        for x, y in net_info.items():
            blob_corresponding_reverse_info_inner[net_name][y] = x
        assert len(blob_corresponding_reverse_info_inner[net_name]) == len(net_info)

    return blob_corresponding_info_inner, blob_corresponding_reverse_info_inner


blob_corresponding_info, blob_corresponding_reverse_info = blobinfo()


def get_one_network_meta(net_name, ec_size=22, blobs_to_extract=None):
    if blobs_to_extract is None:
        blobs_to_extract = list(blob_corresponding_info[net_name].keys())

    # get meta info needed for this network
    if net_name.endswith('_bn'):
        net_name_for_check = net_name[:-3]
    else:
        net_name_for_check = net_name
    input_size = generic_network_definitions.input_size_info[net_name_for_check]
    blob_info = generic_network_definitions.blob_info[net_name_for_check]
    helper_this = cnnsizehelper.CNNSizeHelper(blob_info, input_size)

    # 22 is the original setting, since we rescale image to 2/3 of original
    # (which is found to be (marginally?) better for RSA analysis, in both Corentin's case and Tang)

    top_bottom = input_size[0] / 2 - ec_size / 2, input_size[0] / 2 + ec_size / 2
    left_right = input_size[1] / 2 - ec_size / 2, input_size[1] / 2 + ec_size / 2

    slicing_dict = defaultdict(lambda: ((None, None), (None, None)))

    # compute how many columns to extract.
    for layer in helper_this.layer_info_dict:
        slicing_dict[layer] = helper_this.compute_minimum_coverage(layer, top_bottom, left_right)

    slicing_dict = cnnsizehelper.get_slice_dict(slicing_dict, blobs_to_extract=blobs_to_extract)

    def correspondence_func(x):
        # handle no correspondence case
        return blob_corresponding_reverse_info[net_name].get(x, None)

    return helper_this, slicing_dict, blobs_to_extract, correspondence_func


def get_pretrained_network(net_name):
    a = {'vgg16': vgg16, 'vgg19': vgg19, 'vgg16_bn': vgg16_bn, 'vgg19_bn': vgg19_bn}[net_name](pretrained=True)
    # a.cuda()
    a = a.eval()
    return a


def _forward_hook(m, in_, out_, module_name, callback_dict, slice_this):
    assert isinstance(out_, Variable)
    data_all = out_.data.cpu().numpy()
    # then slice it
    slice_r, slice_c = slice_this
    if data_all.ndim == 4:
        data_this_to_use = data_all[:, :, slice_r, slice_c]
    else:
        assert data_all.ndim == 2
        data_this_to_use = data_all
    # print(f'{data_all.shape} -> {data_this_to_use.shape}')
    # extra copy to guard against weird things.
    callback_dict[module_name]['output'].append(data_this_to_use.copy())


def augment_module_pre(net: nn.Module, module_names: set, module_correspondence=None, slice_dict=None) -> (dict, list):
    callback_dict = OrderedDict()  # not necessarily ordered, but this can help some readability.

    if module_correspondence is None:
        # this maps internal PyTorch name to standard names (in Caffe).
        module_correspondence = lambda x_: x_

    forward_hook_remove_func_list = []
    for x, y in net.named_modules():
        if module_correspondence(x) in module_names:
            callback_dict[module_correspondence(x)] = {}
            callback_dict[module_correspondence(x)]['output'] = []
            forward_hook_remove_func_list.append(
                y.register_forward_hook(
                    partial(_forward_hook, module_name=module_correspondence(x), callback_dict=callback_dict,
                            slice_this=slice_dict[module_correspondence(x)])))

    def remove_handles():
        for h in forward_hook_remove_func_list:
            h.remove()

    return callback_dict, remove_handles


def preprocess_dataset(images, bgcolor, input_size, rescale_ratio=None):
    # rescale
    if rescale_ratio is not None:
        images = np.asarray([rescale(im, scale=rescale_ratio, order=1, mode='edge') for im in images])

    # make sure images are 3D
    if images.ndim == 3:
        images = np.concatenate((images[..., np.newaxis],) * 3, axis=-1)
    assert images.ndim == 4 and images.shape[-1] == 3
    assert np.all(images <= 1) and np.all(images >= 0)

    # use leelab-toolbox pipeline
    steps_naive = ['putInCanvas']
    pars_naive = {'putInCanvas': {'canvas_size': input_size,
                                  'canvas_color': bgcolor,
                                  },
                  }
    pipeline_naive, realpars_naive, order_naive = pipeline.preprocessing_pipeline(steps_naive, pars_naive,
                                                                                  order=steps_naive)
    images_new = pipeline_naive.transform(images.astype(np.float32, copy=False))

    # normalize
    # check
    # http://pytorch.org/docs/master/torchvision/models.html
    images_new -= np.array([0.485, 0.456, 0.406])
    images_new /= np.array([0.229, 0.224, 0.225])
    # transpose
    images_new = np.transpose(images_new, (0, 3, 1, 2))
    # done
    return images_new


def extract_features_one_case(net, dataset_preprocessed, blobs_to_extract, correspondence_func, slicing_dict,
                              batch_size, verbose=True):
    callback_dict, remove_handles = augment_module_pre(net, blobs_to_extract,
                                                       module_correspondence=correspondence_func,
                                                       slice_dict=slicing_dict)

    # then create tensor dataset
    loader_this = DataLoader(TensorDataset(FloatTensor(dataset_preprocessed),
                                           FloatTensor(np.zeros(len(dataset_preprocessed), dtype=np.float32))),
                             batch_size=batch_size)
    for batch_idx, (inputs, _) in enumerate(loader_this):
        inputs = Variable(inputs.cuda(), volatile=True)  # pure inference mode.
        net(inputs)
        if (batch_idx + 1) % 20 == 0 and verbose:
            print(f'[{batch_idx}/{len(loader_this)}]')

    # then collect data
    features_all = OrderedDict()
    for blob_name, blob in callback_dict.items():
        features_all[blob_name] = np.concatenate(blob['output'])
        if verbose:
            print(blob_name, features_all[blob_name].shape)
    # maybe save some memory.
    del callback_dict
    # remove handles.
    remove_handles()
    return features_all


def process_one_case_wrapper(net_name_this, net_this, dataset_np_this, grp_name,
                             setting_this, bg_color, batch_size, file_to_save_input, file_to_save_feature):
    (helper_this, slicing_dict,
     blobs_to_extract, correspondence_func) = get_one_network_meta(net_name_this, setting_this['ec_size'])
    print(grp_name, blobs_to_extract)

    with h5py.File(file_to_save_feature) as f_feature:
        if grp_name not in f_feature:

            # then preproces dataset
            # wrap this in hdf5, so that loading can be a lot faster.
            # for why `ascontiguousarray`, see <https://discuss.pytorch.org/t/problem-with-reading-pfm-image/2924>
            with h5py.File(file_to_save_input) as f_input:
                if grp_name not in f_input:
                    dataset_preprocessed = preprocess_dataset(dataset_np_this, bg_color,
                                                              input_size=helper_this.input_size,
                                                              rescale_ratio=setting_this['scale'])
                    f_input.create_dataset(grp_name, data=dataset_preprocessed, compression="gzip")
                    f_input.flush()
                    print(f'{grp_name} input computation done')
                else:
                    dataset_preprocessed = f_input[grp_name][...]
                    print(f'{grp_name} input computation done before')
            dataset_preprocessed = np.ascontiguousarray(dataset_preprocessed)

            print(dataset_preprocessed.shape)

            features_all = extract_features_one_case(net_this, dataset_preprocessed,
                                                     blobs_to_extract, correspondence_func, slicing_dict, batch_size)
            for blob_idx, blob_data in enumerate(features_all.values()):
                f_feature.create_dataset(f'{grp_name}/{blob_idx}', data=blob_data, compression="gzip")
                f_feature.flush()
            print(f'{grp_name} feature extraction done')
            # save blob names
            f_feature[grp_name].attrs['blobs_to_extract'] = np.array([np.string_(x) for x in blobs_to_extract])
        else:
            print(f'{grp_name} feature extraction done before')
