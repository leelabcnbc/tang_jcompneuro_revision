"""optimizer for CNN with FactoredLinear2D modules"""

import torch
from torch import nn, autograd
from torch.nn import functional
from typing import List
from tang_jcompneuro.cnn import FactoredLinear2D
from torch.nn.functional import mse_loss
import numpy as np


def maskcnn_loss_pilot_kernel(module_list: List[nn.Conv2d],
                              scale=1e-4):
    if scale == 0:
        return 0
    # should be a list of conv module to apply the loss
    # https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/2
    return 0.5 * scale * sum([torch.sum(m.weight ** 2) for m in module_list])


def maskcnn_loss_pilot_weight_spatial(module: FactoredLinear2D,
                                      scale=1):
    if scale == 0:
        return 0
    # should be a list of conv module to apply the loss
    weight_reshape = module.weight_spatial.view(module.weight_spatial.size(0), -1)
    return scale * torch.mean(torch.sum(torch.abs(weight_reshape), 1))


def maskcnn_loss_pilot_weight_feature(module: FactoredLinear2D,
                                      scale=1):
    if scale == 0:
        return 0
    # should be a list of conv module to apply the loss
    weight_reshape = module.weight_feature.view(module.weight_feature.size(0), -1)
    return scale * torch.mean(torch.sum(torch.abs(weight_reshape), 1))


def maskcnn_loss_pilot_mse(yhat, y):
    # average over features, and average over neurons.
    # see https://github.com/pytorch/pytorch/issues/3322 for tricky stuff of MSELoss
    return mse_loss(yhat, y)


def maskcnn_loss_v1_weight_readout(module: FactoredLinear2D,
                                   scale=1):
    if scale == 0:
        return 0
    weight_feature_flat = module.weight_feature.view(module.weight_feature.size()[0], -1)
    weight_spatial_flat = module.weight_spatial.view(module.weight_spatial.size()[0], -1)

    return scale * torch.sum(torch.sum(torch.abs(weight_feature_flat), 1) *
                             torch.sum(torch.abs(weight_spatial_flat), 1))


def maskcnn_loss_v1_kernel_group_sparsity(module_list: List[nn.Conv2d],
                                          scale_list: list):
    """group_sparsity_regularizer_2d in original code"""
    # basically, sum up each HxW slice individually.
    sum_list = []
    for m, s in zip(module_list, scale_list):
        if s == 0:
            continue
        w_this: nn.Parameter = m.weight
        c_out, c_in, h, w = w_this.size()
        w_this = w_this.view(c_out, c_in, h * w)
        sum_to_add = s * torch.sum(torch.sqrt(torch.sum(w_this ** 2, -1)))
        sum_list.append(sum_to_add)
    return sum(sum_list)


_maskcnn_loss_v1_kernel_smoothness_kernel = {'data': None}


def maskcnn_loss_v1_kernel_smoothness(module_list: List[nn.Conv2d],
                                      scale_list: list, gpu=True):
    if _maskcnn_loss_v1_kernel_smoothness_kernel['data'] is None:
        _maskcnn_loss_v1_kernel_smoothness_kernel['data'] = torch.FloatTensor(
            np.array(
                [[[[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]]]
            )
        )
        # 1 x 1 x 3 x 3
        if gpu:
            _maskcnn_loss_v1_kernel_smoothness_kernel['data'] = _maskcnn_loss_v1_kernel_smoothness_kernel['data'].cuda()
        _maskcnn_loss_v1_kernel_smoothness_kernel['data'] = autograd.Variable(
            _maskcnn_loss_v1_kernel_smoothness_kernel['data'])
    kernel = _maskcnn_loss_v1_kernel_smoothness_kernel['data']

    """group_sparsity_regularizer_2d in original code"""
    # basically, sum up each HxW slice individually.
    sum_list = []
    for m, s in zip(module_list, scale_list):
        if s == 0:
            continue
        w_this: nn.Parameter = m.weight
        c_out, c_in, h, w = w_this.size()

        w_this = w_this.view(c_out * c_in, 1, h, w)
        w_this_conved = functional.conv2d(w_this, kernel, padding=1).view(c_out, c_in, -1)
        w_this = w_this.view(c_out, c_in, -1)

        sum_to_add = s * torch.sum(torch.sum(w_this_conved ** 2, -1) / torch.sum(w_this ** 2, -1))
        sum_list.append(sum_to_add)
    return sum(sum_list)


def maskcnn_loss_v1_poisson(yhat, y):
    return torch.mean(torch.sum(yhat - y * torch.log(yhat + 1e-5), 1))
