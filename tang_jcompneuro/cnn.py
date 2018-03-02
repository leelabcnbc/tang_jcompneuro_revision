"""a rewrite of cnn.py

this version is mostly inspired by
NIPS2017 (mask cnn).
see https://github.com/leelabcnbc/thesis-proposal-yimeng/blob/master/thesis_proposal/population_neuron_fitting/maskcnn/cnn.py
"""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import init as nn_init
import math
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from .configs.cnn_arch import sanity_check_arch_config
from .configs.cnn_init import sanity_check_init_config
from .configs.cnn_opt import sanity_check_opt_config, sanity_check_one_optimizer_opt_config
from torch.nn.functional import mse_loss


class FactoredLinear2D(nn.Module):
    """
    skeleton copied from implementation of nn.Linear from PyTorch 0.3.1
    # just copied from my maskcnn implementation.
    """

    def __init__(self, in_channels, map_size, out_features, bias=True,
                 weight_feature_constraint=None, weight_spatial_constraint=None):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0
        self.in_channels = in_channels

        self.map_size = _check_input_size(map_size)

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


def _check_input_size(input_size):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
    assert isinstance(input_size, tuple) and len(input_size) == 2
    assert isinstance(input_size[0], int) and input_size[0] > 0
    assert isinstance(input_size[1], int) and input_size[1] > 0
    return input_size


def inv_softplus(x):
    # copied from original code.
    # I think numerically it's not very stable.
    return np.log(np.exp(x) - 1)


class CNN(nn.Module):
    """
    a class that can handle all CNN variants in the paper.
    """

    def __init__(self,
                 arch_config,
                 init_config,
                 input_size=20,
                 n=1,
                 bn_eps=0.001,
                 mean_response=None,
                 seed=None):
        super().__init__()
        # ====== parameter check start ======

        # if init_config is None:
        #     # match what's in original code.
        #     init_config = {
        #         'conv_std': 0.01,
        #         'fc_std': 0.01,
        #     }

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        sanity_check_arch_config(arch_config)
        sanity_check_init_config(init_config)

        self.input_size = _check_input_size(input_size)
        self.act_fn = arch_config['act_fn']
        # ====== parameter check end   ======

        # ====== define conv layers    ======
        self.conv, map_size = self._generate_conv(arch_config['conv'], bn_eps)

        # ====== define fc layer       ======
        self.reshape_conv = not arch_config['fc']['factored']
        self.fc = self._generate_fc(map_size, arch_config['conv'][-1]['out_channel'],
                                    arch_config['fc'], n)

        # ====== define last act fn    ======
        if not arch_config['linear_output']:
            if self.act_fn == 'softplus':
                self.final_act = nn.Softplus()
            elif self.act_fn == 'relu':
                self.final_act = nn.ReLU()
            else:
                raise NotImplementedError
        else:
            self.final_act = None

        self.init_weights(init_config)
        if mean_response is not None:
            self.init_bias(mean_response)

        # helper for computing loss.
        self.conv_module_list = [x for x in self.conv.children() if isinstance(x, nn.Conv2d)]

    def _generate_conv(self, conv_config, bn_eps):
        map_size = self.input_size
        conv_all = []
        for idx, conv_this_layer in enumerate(conv_config):

            kernel_size = conv_this_layer['kernel_size']
            in_channels = 1 if idx == 0 else conv_config[idx - 1]['out_channel']
            stride = conv_this_layer['stride']
            padding = conv_this_layer['padding']

            map_size = ((map_size[0] - kernel_size + 2 * padding) // stride + 1,
                        (map_size[1] - kernel_size + 2 * padding) // stride + 1)

            conv_all.append(
                (f'conv{idx}', nn.Conv2d(in_channels=in_channels,
                                         out_channels=conv_this_layer['out_channel'],
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=not conv_this_layer['bn']))
            )
            if conv_this_layer['bn']:
                conv_all.append(
                    # notice that, to match behavior of original code,
                    # for the optimizer, I need to set learning rate for gamma to be 0.
                    # or .weight here.
                    (f'bn{idx}', nn.BatchNorm2d(num_features=conv_this_layer['out_channel'],
                                                eps=bn_eps, momentum=0.1, affine=True))
                )
            if self.act_fn == 'softplus':
                conv_all.append(
                    (f'act{idx}',
                     # this is essentially what `elu` (which is NOT the ELU in standard usage)
                     # means in the original code.
                     nn.Softplus()
                     )
                )
            elif self.act_fn == 'relu':
                conv_all.append(
                    (f'act{idx}',
                     nn.ReLU()
                     )
                )
            else:
                # to implement other nonlinearities
                # such as HalfSquaring, etc.
                # check http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Sigmoid
                # to define a new module.
                # it's not difficult.
                raise NotImplementedError

            # finally, add pooling.
            pool_config = conv_this_layer['pool']
            if pool_config is not None:
                conv_all.append(
                    (f'pool{idx}', nn.MaxPool2d(kernel_size=pool_config['kernel_size'],
                                                stride=pool_config['stride'],
                                                padding=pool_config['padding']))
                )
                map_size = ((map_size[0] - pool_config['kernel_size'] + 2
                             * pool_config['padding']) // pool_config['stride'] + 1,
                            (map_size[1] - pool_config['kernel_size'] + 2
                             * pool_config['padding']) // pool_config['stride'] + 1)

        return nn.Sequential(OrderedDict(conv_all)), map_size

    def _generate_fc(self, map_size, out_channel, fc_config, n):
        module_list = []
        if fc_config['factored']:
            module_list.append(('fc', FactoredLinear2D(out_channel,
                                                       map_size, n, bias=True)))
        else:
            module_list.append(('fc', nn.Linear(map_size[0] * map_size[1] * out_channel, n)))
            if fc_config['dropout'] is not None:
                module_list.append(('dropout', nn.Dropout(p=fc_config['dropout'])))

        return nn.Sequential(OrderedDict(module_list))

    def init_bias(self, mean_response):
        if self.act_fn == 'softplus':
            b = inv_softplus(mean_response)
        elif self.act_fn == 'relu':
            b = mean_response
        else:
            raise NotImplementedError
        assert b.shape == self.factored_fc_2d.bias.size()
        assert np.all(np.isfinite(b))
        self.fc.fc.bias.data[...] = torch.Tensor(b)

    def init_weights(self, init_config):

        name_mapping_random = {
            'conv.conv0.weight': 'conv_init',
            'conv.conv1.weight': 'conv_init',
            'conv.conv2.weight': 'conv_init',
            'conv.conv0.bias': 0,
            'conv.conv1.bias': 0,
            'conv.conv2.bias': 0,
            'conv.bn0.bias': 0,
            'conv.bn1.bias': 0,
            'conv.bn2.bias': 0,
            'conv.bn0.weight': 1,
            'conv.bn1.weight': 1,
            'conv.bn2.weight': 1,
            # for factored case
            'fc.fc.weight_feature': 'fc_init',
            'fc.fc.weight_spatial': 'fc_init',
            # for unfactored
            'fc.fc.weight': 'fc_init',
            # for both
            # also, this param will be intialized by mean params.
            'fc.fc.bias': 0,
        }

        # use init_config
        for param_name, param_value in self.named_parameters():
            # print(param_name, type(param_value), param_value.size())
            data_to_fill = name_mapping_random[param_name]
            if not isinstance(data_to_fill, str):
                param_value.data[...] = data_to_fill
            else:
                # simple. with preconditioning of mean parameters,
                # it should not be that bad.
                fill_value = init_config[data_to_fill]
                if isinstance(fill_value, float):
                    param_value.data.normal_(0, fill_value)
                else:
                    assert fill_value == 'kaiming_fan_out'
                    nn_init.kaiming_normal(param_value.data, mode='fan_out')

        pass

    def forward(self, input):
        x = self.conv(input)
        if self.reshape_conv:
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.final_act is not None:
            x = self.final_act(x)

        return x


def get_optimizer(model: CNN, optimizer_config: dict):
    assert sanity_check_one_optimizer_opt_config(optimizer_config)
    # always learn everything.
    if optimizer_config['optimizer_type'] == 'sgd':
        optimizer_this = optim.SGD(model.parameters(), lr=optimizer_config['lr'],
                                   momentum=optimizer_config['momentum'])
    elif optimizer_config['optimizer_type'] == 'adam':
        optimizer_this = optim.Adam(model.parameters(), lr=optimizer_config['lr'])
    else:
        raise NotImplementedError
    return optimizer_this


def get_conv_loss(opt_conv_config, conv_module_list):
    sum_list = []
    for m, s in zip(conv_module_list, opt_conv_config):
        w_this: nn.Parameter = m.weight

        if s['l2'] != 0:
            sum_list.append(s['l2'] * 0.5 * torch.sum(w_this ** 2))
        if s['l1'] != 0:
            sum_list.append(s['l1'] * torch.sum(torch.abs(w_this)))
        if m.bias is not None:
            if s['l2_bias'] != 0:
                sum_list.append(s['l2_bias'] * 0.5 * torch.sum(m.bias ** 2))
            if s['l1_bias'] != 0:
                sum_list.append(s['l1_bias'] * torch.sum(torch.abs(m.bias)))

    return sum(sum_list)


def get_fc_loss(opt_fc_config, fc_module):
    sum_list = []
    if isinstance(fc_module, nn.Linear):
        # simple
        w_this: nn.Parameter = fc_module.weight
        if opt_fc_config['l2'] != 0:
            sum_list.append(opt_fc_config['l2'] * 0.5 * torch.sum(w_this ** 2))
        if opt_fc_config['l1'] != 0:
            sum_list.append(opt_fc_config['l1'] * torch.sum(torch.abs(w_this)))
    elif isinstance(fc_module, FactoredLinear2D):
        w_this_1: nn.Parameter = fc_module.weight_feature
        w_this_2: nn.Parameter = fc_module.weight_spatial
        if opt_fc_config['l2'] != 0:
            sum_list.append(opt_fc_config['l2'] * 0.5 * torch.sum(w_this_1 ** 2))
            sum_list.append(opt_fc_config['l2'] * 0.5 * torch.sum(w_this_2 ** 2))
        if opt_fc_config['l1'] != 0:
            sum_list.append(opt_fc_config['l1'] * torch.sum(torch.abs(w_this_1)))
            sum_list.append(opt_fc_config['l1'] * torch.sum(torch.abs(w_this_2)))
    else:
        raise NotImplementedError

    if fc_module.bias is not None:
        if opt_fc_config['l2_bias'] != 0:
            sum_list.append(opt_fc_config['l2_bias'] * 0.5 * torch.sum(fc_module.bias ** 2))
        if opt_fc_config['l1_bias'] != 0:
            sum_list.append(opt_fc_config['l1_bias'] * torch.sum(torch.abs(fc_module.bias)))
    return sum(sum_list)


def get_output_loss(yhat, y, loss_type):
    if loss_type == 'mse':
        return mse_loss(yhat, y)
    elif loss_type == 'poisson':
        # 1e-5 is for numerical stability.
        # same in NIPS2017 (mask CNN) code.
        return torch.mean(yhat - y * torch.log(yhat + 1e-5))
    else:
        raise NotImplementedError


def get_loss(opt_config: dict, model: CNN = None, strict=True):
    assert sanity_check_opt_config(opt_config)
    opt_config = deepcopy(opt_config)
    # we don't need model. but that can be of help.
    if strict:
        assert model is not None
    if model is not None:
        assert len(model.conv_module_list) == len(opt_config['conv'])

    def loss_func_inner(yhat, y, model_this: CNN):
        conv_loss = get_conv_loss(opt_config['conv'], model_this.conv_module_list)
        fc_loss = get_fc_loss(opt_config['fc'], model_this.fc.fc)
        output_loss = get_output_loss(yhat, y, opt_config['loss'])
        return conv_loss + fc_loss + output_loss

    return loss_func_inner
