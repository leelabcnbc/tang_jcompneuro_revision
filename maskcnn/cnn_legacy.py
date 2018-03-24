"""the file that defines CNN.
a PyTorch rewrite and merge of `CNN.py` and `v1data/convnet.py` of original repo.

I think the best way is to define a PyTorch module for that factorized fc layer (with or without bias)
"""

import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from collections import OrderedDict


class FactoredLinear2D(nn.Module):
    """
    skeleton copied from implementation of nn.Linear from PyTorch 0.3.1
    """

    def __init__(self, in_channels, map_size, out_features, bias=True,
                 weight_feature_constraint=None, weight_spatial_constraint=None):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0
        self.in_channels = in_channels

        map_size = (map_size, map_size) if isinstance(map_size, int) else map_size
        assert isinstance(map_size, tuple) and len(map_size) == 2
        assert isinstance(map_size[0], int) and map_size[0] > 0
        assert isinstance(map_size[1], int) and map_size[1] > 0

        self.map_size = map_size

        assert isinstance(out_features, int) and out_features > 0
        self.out_features = out_features

        assert weight_feature_constraint in {None, 'abs'}
        self.weight_feature_constraint = weight_feature_constraint
        assert weight_spatial_constraint in {None, 'abs'}
        self.weight_spatial_constraint = weight_spatial_constraint

        self.weight_spatial: nn.Parameter = nn.Parameter(
            torch.Tensor(self.out_features, self.map_size[0], self.map_size[1]))
        self.weight_feature: nn.Parameter = nn.Parameter(torch.Tensor(self.out_features, self.in_channels))
        if bias:
            self.bias: nn.Parameter = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # print('changed impl')

    def reset_parameters(self):
        # this is simply adapted from nn.Linear. should always be initialized by hand.
        stdv = 1. / math.sqrt(self.in_channels * self.map_size[0] * self.map_size[1])
        self.weight_spatial.data.uniform_(-stdv, stdv)
        self.weight_feature.data.fill_(1.0)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # I assume that input has shape (N, in_channels, map_size[0], map_size[1]
        # first get the weights.

        weight_spatial_view = self.weight_spatial
        weight_feature_view = self.weight_feature

        if self.weight_feature_constraint is not None:
            if self.weight_feature_constraint == 'abs':
                # weight_spatial_view = torch.abs(weight_spatial_view)
                weight_feature_view = torch.abs(weight_feature_view)
            else:
                raise RuntimeError

        if self.weight_spatial_constraint is not None:
            if self.weight_spatial_constraint == 'abs':
                weight_spatial_view = torch.abs(weight_spatial_view)
            else:
                raise RuntimeError

        weight_spatial_view = weight_spatial_view.view(self.out_features, 1, self.map_size[0], self.map_size[1])
        weight_feature_view = weight_feature_view.view(self.out_features, self.in_channels, 1, 1)

        # then broadcast to get new weight.
        if self.in_channels != 1:
            weight = weight_spatial_view * weight_feature_view
        else:
            # feature weighting not needed
            # this is for both quicker learning, as well as being compatible with `CNN.py` in the original repo.
            weight = weight_spatial_view.expand(self.out_features, self.in_channels, self.map_size[0], self.map_size[1])
        weight = weight.view(self.out_features, self.in_channels * self.map_size[0] * self.map_size[1])
        return F.linear(input.view(input.size(0), -1), weight, self.bias)


def get_maskcnn_v1_conv_config(*, out_channels=48, layer1_kernel=13,
                               layerhigh_kernel=3):
    _maskcnn_v1_conv_config_original = {
        'filter_sizes': [layer1_kernel, layerhigh_kernel, layerhigh_kernel],
        'out_channels': [out_channels, out_channels, out_channels],
        'strides': [1, 1, 1],
        'paddings': ['VALID', 'SAME', 'SAME'],
        'smooth_weights': [0.03, 0, 0],
        'sparse_weights': [0, 0.05, 0.05],
    }

    num_layer = len(_maskcnn_v1_conv_config_original['filter_sizes'])
    _maskcnn_v1_conv_config_new = []
    for idx in range(num_layer):
        dict_this = {
            k: v[idx] for k, v in _maskcnn_v1_conv_config_original.items()
        }
        _maskcnn_v1_conv_config_new.append(dict_this)
    return _maskcnn_v1_conv_config_new


# filter_sizes = [13, 3, 3],
# out_channels = [48, 48, 48],
# strides = [1, 1, 1],
# paddings = ['VALID', 'SAME', 'SAME'],
# smooth_weights = [0.03, 0, 0],
# sparse_weights = [0, 0.05, 0.05],

def inv_softplus(x):
    # copied from original code.
    # I think numerically it's not very stable.
    return np.log(np.exp(x) - 1)


class MaskCNNV1(nn.Module):
    """replicate `v1data/convnet.py` in the original repo"""

    def __init__(self, n, input_size=(31, 31),
                 conv_config_list=None, init_dict_tf=None, bn_eps=0.001,
                 weight_spatial_constraint='abs',
                 weight_feature_constraint='abs',
                 init_config=None,
                 mean_response=None):
        # filter_sizes = [13, 3, 3],
        # out_channels = [48, 48, 48],
        # strides = [1, 1, 1],
        # paddings = ['VALID', 'SAME', 'SAME'],
        # smooth_weights = [0.03, 0, 0],
        # sparse_weights = [0, 0.05, 0.05],
        super().__init__()
        # image_size, kernel_size = s
        # self.conv = nn.Conv2d(1, 1, kernel_size, bias=False)
        # # I think BN in TF and BN in PyTorch have different implementations.
        # # for example, I think eps in TF is added to variance directly,
        # # whereas in PyTorch, it's added outside sqrt-ed variance.
        # # Here I can only do a best-effort match.
        # #
        # #
        # # NONONO. My falut.
        # # For BN, PyTorch as eps inside, just as TF.
        # # For Instance normalization, PyTorch as eps outside.
        # #
        # self.bn = nn.BatchNorm2d(1, momentum=0.002, affine=False, eps=bn_eps)
        # map_size = image_size - kernel_size + 1

        # build conv layer

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
            input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        assert isinstance(input_size, tuple) and len(input_size) == 2
        assert isinstance(input_size[0], int) and input_size[0] > 0
        assert isinstance(input_size[1], int) and input_size[1] > 0
        self.input_size = input_size
        self.no_learning_params = {'conv.bn0.weight', 'conv.bn1.weight', 'conv.bn2.weight'}

        if conv_config_list is None:
            conv_config_list = get_maskcnn_v1_conv_config()

        map_size = input_size
        conv_all = []
        self.smooth_weights = []
        self.sparse_weights = []
        for idx, conv_this_layer in enumerate(conv_config_list):

            self.smooth_weights.append(conv_this_layer['smooth_weights'])
            self.sparse_weights.append(conv_this_layer['sparse_weights'])

            kernel_size = conv_this_layer['filter_sizes']
            in_channels = 1 if idx == 0 else conv_config_list[idx - 1]['out_channels']
            stride = conv_this_layer['strides']
            if conv_this_layer['paddings'] == 'VALID':
                padding = 0
            else:
                assert conv_this_layer['paddings'] == 'SAME'
                assert stride == 1, 'only implemented for stride=1 for now.'
                padding = (kernel_size - 1) // 2

            map_size = ((map_size[0] - kernel_size + 2 * padding) // stride + 1,
                        (map_size[1] - kernel_size + 2 * padding) // stride + 1)

            conv_all.append(
                (f'conv{idx}', nn.Conv2d(in_channels=in_channels,
                                         out_channels=conv_this_layer['out_channels'],
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False))
            )
            conv_all.append(
                # notice that, to match behavior of original code,
                # for the optimizer, I need to set learning rate for gamma to be 0.
                # or .weight here.
                (f'bn{idx}', nn.BatchNorm2d(num_features=conv_this_layer['out_channels'],
                                            eps=bn_eps, momentum=0.1, affine=True))
            )
            if idx != len(conv_config_list) - 1:
                conv_all.append(
                    (f'act{idx}',
                     # this is essentially what `elu` (which is NOT the ELU in standard usage)
                     # means in the original code.
                     nn.Softplus())
                )
        self.conv = nn.Sequential(OrderedDict(conv_all))

        self.factored_fc_2d = FactoredLinear2D(conv_config_list[-1]['out_channels'],
                                               map_size, n, bias=True,
                                               weight_spatial_constraint=weight_spatial_constraint,
                                               weight_feature_constraint=weight_feature_constraint)
        self.final_act = nn.Softplus()

        if init_config is None:
            # match what's in original code.
            init_config = {
                'conv_std': 0.01,
                'factored_fc_2d_spatial_std': 0.01,
                'factored_fc_2d_feature_std': 0.01,
            }

        self.init_weights(init_dict_tf, init_config)
        if mean_response is not None:
            self.init_bias(mean_response)

        # helper for computing loss.
        self.conv_module_list = [x for x in self.conv.children() if isinstance(x, nn.Conv2d)]

    def init_bias(self, mean_response):
        b = inv_softplus(mean_response)
        assert b.shape == self.factored_fc_2d.bias.size()
        assert np.all(np.isfinite(b))
        self.factored_fc_2d.bias.data[...] = torch.Tensor(b)

    def init_weights(self, init_dict_tf, init_config):

        name_mapping = {
            'conv.conv0.weight': 'conv0/weights',
            'conv.conv1.weight': 'conv1/weights',
            'conv.conv2.weight': 'conv2/weights',
            'conv.bn0.bias': 'conv0/BatchNorm/beta',
            'conv.bn1.bias': 'conv1/BatchNorm/beta',
            'conv.bn2.bias': 'conv2/BatchNorm/beta',
            'factored_fc_2d.weight_feature': 'W_features',
            'factored_fc_2d.weight_spatial': 'W_spatial',
            'factored_fc_2d.bias': 'b_out',
        }

        name_mapping_random = {
            'conv.conv0.weight': 'conv_std',
            'conv.conv1.weight': 'conv_std',
            'conv.conv2.weight': 'conv_std',
            'conv.bn0.bias': 0,
            'conv.bn1.bias': 0,
            'conv.bn2.bias': 0,
            'factored_fc_2d.weight_feature': 'factored_fc_2d_feature_std',
            'factored_fc_2d.weight_spatial': 'factored_fc_2d_spatial_std',
            # this will be left in the end.
            # 'factored_fc_2d.bias': 'b_out',
        }

        value_mapping = {
            # conv2d TF to conv2d PyTorch.
            # PyTorch has shape (check official doc of Conv2d)
            # out_channels, in_channels, kernel_size[0], kernel_size[1]
            # TF has shape
            # (see https://discuss.pytorch.org/t/how-to-transfer-pretained-model-from-tensorflow-to-pytorch/6173)
            # H, W, INPUT_C, OUTPUT_C
            # (regardless of `data_format`=NCHW or NHWC for conv2d)
            # see https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
            'conv0/weights': lambda x: np.transpose(x, (3, 2, 0, 1)),
            'conv1/weights': lambda x: np.transpose(x, (3, 2, 0, 1)),
            'conv2/weights': lambda x: np.transpose(x, (3, 2, 0, 1)),
            'conv0/BatchNorm/beta': lambda x: x,
            'conv1/BatchNorm/beta': lambda x: x,
            'conv2/BatchNorm/beta': lambda x: x,
            'b_out': lambda x: x,
            'W_features': lambda x: x.T,
            'W_spatial': lambda x: np.reshape(x, (
                np.int(np.round(np.sqrt(x.shape[0]))), np.int(np.round(np.sqrt(x.shape[0]))),
                x.shape[1])).transpose((2, 0, 1))
        }

        if init_dict_tf is not None:
            for param_name, param_value in self.named_parameters():
                # print(param_name, type(param_value), param_value.size())
                if param_name in name_mapping:
                    name_in_tf = name_mapping[param_name]
                    data_to_fill = value_mapping[name_in_tf](init_dict_tf[name_in_tf])
                    assert data_to_fill.shape == param_value.size()
                    param_value.data[...] = torch.Tensor(data_to_fill)
                else:
                    # fill in 1.
                    # print(param_name)
                    # YIMENG: here, no_learning_params only specify what parameters should have zero
                    # learning rate, but not their initial values. whatever. it will be fixed later if necessary.
                    assert param_name in self.no_learning_params
                    param_value.data[...] = 1
        else:
            # use init_config
            for param_name, param_value in self.named_parameters():
                # print(param_name, type(param_value), param_value.size())
                if param_name in name_mapping_random:
                    data_to_fill = name_mapping_random[param_name]
                    if not isinstance(data_to_fill, str):
                        param_value.data[...] = data_to_fill
                    else:
                        # Pytorch doesn't have truncated normal yet.
                        # but should be fine...
                        param_value.data.normal_(0, init_config[data_to_fill])
                else:
                    # fill in 1.
                    # print(param_name)
                    # YIMENG: here, no_learning_params only specify what parameters should have zero
                    # learning rate, but not their initial values. whatever. it will be fixed later if necessary.
                    if param_name in self.no_learning_params:
                        param_value.data[...] = 1
                    else:
                        assert param_name == 'factored_fc_2d.bias'
                        # this should be initlaized later.
                        param_value.data.zero_()
        pass

    def forward(self, input):
        x = self.conv(input)
        x = self.factored_fc_2d(x)
        x = self.final_act(x)

        return x
