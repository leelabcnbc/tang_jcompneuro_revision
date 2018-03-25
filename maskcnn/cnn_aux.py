from copy import deepcopy

import torch
from torch import optim
from torch.nn.functional import mse_loss
from .cnn_opt import (sanity_check_one_optimizer_opt_config,
                      sanity_check_opt_config,
                      generate_one_conv_layer_opt_config,
                      generate_one_opt_config,
                      generate_one_fc_layer_opt_config,
                      generate_one_optimizer_config
                      )
from . import optim as optim_maskcnn
from tang_jcompneuro.cnn import CNN
from tang_jcompneuro.configs import cnn_arch
from tang_jcompneuro.configs.cnn_init import generate_init_config


def get_maskcnn_v1_arch_config(out_channel=48, kernel_size_l1=13,
                               kernel_size_l23=3, factored_constraint='abs',
                               act_fn='softplus', last_conv_linear=True):
    # regenerate V1 mask cnn in NIPS2017 paper
    conv1_config = cnn_arch.generate_one_conv_config(kernel_size=kernel_size_l1,
                                                     out_channel=out_channel,
                                                     bn=True, bn_affine=True)
    conv2_config = cnn_arch.generate_one_conv_config(kernel_size=kernel_size_l23,
                                                     out_channel=out_channel,
                                                     bn=True, padding=kernel_size_l23 // 2)
    # repeat.
    conv3_config = cnn_arch.generate_one_conv_config(kernel_size=kernel_size_l23,
                                                     out_channel=out_channel,
                                                     bn=True, padding=kernel_size_l23 // 2)
    fc_config = cnn_arch.generate_one_fc_config(factored=True,
                                                factored_constraint=factored_constraint)
    return cnn_arch.generate_one_config(
        [conv1_config, conv2_config, conv3_config], fc_config, act_fn, False,
        conv_last_no_act=last_conv_linear)


def get_maskcnn_v1_opt_config(layer=3, group=0.05, smoothness=0.03, scale=0.02,
                              legacy=True, bn_scale_nolearning=True, loss_type='poisson'):
    assert layer >= 1
    conv1_config = generate_one_conv_layer_opt_config(0.0, smoothness)
    conv2_higher_config = generate_one_conv_layer_opt_config(group, 0.0)

    return generate_one_opt_config(
        [conv1_config, ] + [deepcopy(conv2_higher_config) for _ in range(layer - 1)],
        generate_one_fc_layer_opt_config(scale),
        loss_type, generate_one_optimizer_config('adam',
                                                 bn_scale_nolearning=bn_scale_nolearning),
        legacy=legacy
    )


def get_optimizer(model: CNN, optimizer_config: dict):
    assert sanity_check_one_optimizer_opt_config(optimizer_config)
    # always learn everything.
    # collect lr groups.
    # this is default in V1maskCNN. But I think it's useless.
    standard_grp = []
    no_learning_grp = []
    if optimizer_config['bn_scale_nolearning']:
        for x, y in model.named_parameters():
            if x in {'conv.bn0.weight', 'conv.bn1.weight', 'conv.bn2.weight'}:
                no_learning_grp.append(y)
            else:
                standard_grp.append(y)
        params_to_learn = [{'params': standard_grp}, {'params': no_learning_grp, 'lr': 0}]
    else:
        params_to_learn = model.parameters()

    if optimizer_config['optimizer_type'] == 'sgd':
        optimizer_this = optim.SGD(params_to_learn, lr=optimizer_config['lr'],
                                   momentum=optimizer_config['momentum'])
    elif optimizer_config['optimizer_type'] == 'adam':
        optimizer_this = optim.Adam(params_to_learn, lr=optimizer_config['lr'])
    else:
        raise NotImplementedError
    return optimizer_this


def _get_output_loss(yhat, y, loss_type, legacy):
    if loss_type == 'mse':
        # return mse_loss(yhat, y)
        if legacy:
            return torch.mean(torch.sum((yhat - y) ** 2, 1))
        else:
            return mse_loss(yhat, y)
    elif loss_type == 'poisson':
        # 1e-5 is for numerical stability.
        # same in NIPS2017 (mask CNN) code.
        if not legacy:
            return torch.mean(yhat - y * torch.log(yhat + 1e-5))
        else:
            # I don't like it. but this is for reproducing their code
            # TODO: actually this is probably correct, as their readout loss
            # for fc layer uses sum (which scales with # of neurons).
            # given that in my experiments I set conv kernel penalty to zero,
            # I think a loss that scales is good.
            return torch.mean(torch.sum(yhat - y * torch.log(yhat + 1e-5), 1))
    else:
        raise NotImplementedError


def get_loss(opt_config: dict, model: CNN = None, strict=True,
             return_dict=False):
    assert sanity_check_opt_config(opt_config)
    opt_config = deepcopy(opt_config)
    # we don't need model. but that can be of help.
    if strict:
        assert model is not None
    if model is not None:
        assert len(model.conv_module_list) == len(opt_config['conv'])

    group_list = [x['group'] for x in opt_config['conv']]
    smooth_list = [x['smoothness'] for x in opt_config['conv']]

    def loss_func_inner(yhat, y, model_this: CNN):
        # use their way instead.
        group_sparsity = optim_maskcnn.maskcnn_loss_v1_kernel_group_sparsity(model.conv_module_list, group_list)
        smooth_sparsity = optim_maskcnn.maskcnn_loss_v1_kernel_smoothness(model.conv_module_list, smooth_list)
        readout_reg = optim_maskcnn.maskcnn_loss_v1_weight_readout(model_this.fc.fc,
                                                                   scale=opt_config['fc']['scale'])
        output_loss = _get_output_loss(yhat, y, opt_config['loss'], opt_config['legacy'])

        if return_dict:
            # for debugging
            result_dict = {
                'group_sparsity': group_sparsity,
                'smooth_sparsity': smooth_sparsity,
                'readout_reg': readout_reg,
                'poisson': output_loss,
                'total_loss': group_sparsity + smooth_sparsity + readout_reg + output_loss
            }
            for x1, x2 in result_dict.items():
                result_dict[x1] = x2.data.cpu().numpy()
                assert result_dict[x1].shape == (1,)
                result_dict[x1] = result_dict[x1][0]
            return result_dict
        else:
            return group_sparsity + smooth_sparsity + readout_reg + output_loss

    return loss_func_inner


def v1_maskcnn_generator():
    return generate_init_config(0.01, 0.01)
