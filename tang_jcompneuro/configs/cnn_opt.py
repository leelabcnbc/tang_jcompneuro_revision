"""optimization related configs

L1, L2 of conv layers (not bn), bias or not. (separately for L1 and L2)
    separate for each layer.
L1, L2 of fc layers, bias or not (separately for L1 and L2)

loss type: MSE or Poisson

optimizer: Adam or SGD (with momentum of 0.9, which is always the case)
    - lr: (starting) learning rate.
    - other things such as momentum and etc. are left there.

I want batch size, early stopping, epoch, etc. to be the same.
also, they are irrelevant for me to generate a PyTorch optimizer.
"""
from collections import OrderedDict
from . import type_check_wrapper


def sanity_check_opt_config(config):
    return type_check_wrapper(config, _type_checker,
                              {'fc', 'conv', 'loss', 'optimizer'})


def sanity_checl_conv_opt_config(config):
    assert isinstance(config, list)
    for x in config:
        assert sanity_check_one_layer_opt_config(x)
    return True


def sanity_check_one_layer_opt_config(config):
    return type_check_wrapper(config, _type_checker,
                              {'l1', 'l2', 'l1_bias', 'l2_bias'})


def sanity_check_one_optimizer_opt_config(config):
    assert isinstance(config, dict)
    optimizer_type = config['optimizer_type']
    if optimizer_type == 'sgd':
        assert type_check_wrapper(config, _type_checker, {'optimizer_type',
                                                          'lr', 'momentum'})
    elif optimizer_type == 'adam':
        assert type_check_wrapper(config, _type_checker, {'optimizer_type',
                                                          'lr'})
    else:
        raise NotImplementedError
    return True


def generate_one_optimizer_config(optimizer_type, lr=None):
    config = {'optimizer_type': optimizer_type}
    if optimizer_type == 'sgd':
        config['lr'] = 0.1 if lr is None else lr
        config['momentum'] = 0.9
    elif optimizer_type == 'adam':
        config['lr'] = 0.001 if lr is None else lr
    else:
        raise NotImplementedError

    assert sanity_check_one_optimizer_opt_config(config)
    return config


def generate_one_opt_config(conv_config_list, fc_config, loss, optimizer):
    config = dict()
    config['conv'] = conv_config_list
    config['fc'] = fc_config
    config['loss'] = loss
    config['optimizer'] = optimizer
    assert sanity_check_opt_config(config)
    return config


def generate_one_layer_opt_config(l1, l2, l1_bias=None, l2_bias=None):
    # by default, l1_bias = l1, l2_bias = l2.
    # so to stop them from learning you need to set them to zero.

    if l1_bias is None:
        l1_bias = l1
    if l2_bias is None:
        l2_bias = l2

    config = dict()
    config['l1'] = l1
    config['l2'] = l2
    config['l1_bias'] = l1_bias
    config['l2_bias'] = l2_bias

    assert sanity_check_one_layer_opt_config(config)
    return config


_type_checker = {
    'l1': float,
    'l2': float,
    'l1_bias': float,
    'l2_bias': float,
    'conv': sanity_checl_conv_opt_config,
    'fc': sanity_check_one_layer_opt_config,
    'loss': lambda x: x in {'mse', 'poisson'},
    'optimizer': sanity_check_one_optimizer_opt_config,
    'optimizer_type': lambda x: x in {'adam', 'sgd'},
    'lr': float,
    'momentum': float,

}


#
def legacy_opt_generator():
    opt_dict_this = OrderedDict()
    """implement adam_longer, baseline, and middle_decay strategies"""

    def layer_gen(x):
        return generate_one_layer_opt_config(l1=0.0, l2=x)

    def optimizer_config_gen(x):
        return generate_one_optimizer_config(x)

    opt_dict_this['baseline'] = generate_one_opt_config([layer_gen(0.0001)], layer_gen(0.0001), 'mse',
                                                        optimizer_config_gen('sgd'))
    opt_dict_this['middle_decay'] = generate_one_opt_config([layer_gen(0.001)],
                                                            layer_gen(0.001), 'mse',
                                                            optimizer_config_gen('sgd'))
    opt_dict_this['adam_longer'] = generate_one_opt_config([layer_gen(0.0001)],
                                                           layer_gen(0.0001), 'mse',
                                                           optimizer_config_gen('adam'))
    return opt_dict_this


def generate_all_opt():
    opt_dict_ = OrderedDict()

    opt_dict_['legacy'] = legacy_opt_generator()

    return opt_dict_


opt_dict = generate_all_opt()
