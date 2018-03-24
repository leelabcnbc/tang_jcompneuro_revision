"""optimization related configs

group + smooth for conv
sparseness for readout.
"""
from tang_jcompneuro.configs import type_check_wrapper


def sanity_check_opt_config(config):
    return type_check_wrapper(config, _type_checker,
                              {'fc', 'conv', 'loss', 'optimizer', 'legacy'})


def sanity_check_conv_opt_config(config):
    assert isinstance(config, list)
    for x in config:
        assert sanity_check_one_conv_layer_opt_config(x)
    return True


def sanity_check_one_conv_layer_opt_config(config):
    return type_check_wrapper(config, _type_checker, {'group', 'smoothness'})


def sanity_check_one_fc_layer_opt_config(config):
    return type_check_wrapper(config, _type_checker, {'scale'})


def sanity_check_one_optimizer_opt_config(config):
    assert isinstance(config, dict)
    optimizer_type = config['optimizer_type']
    if optimizer_type == 'sgd':
        assert type_check_wrapper(config, _type_checker, {'optimizer_type',
                                                          'lr', 'momentum',
                                                          'bn_scale_nolearning'})
    elif optimizer_type == 'adam':
        assert type_check_wrapper(config, _type_checker, {'optimizer_type',
                                                          'lr',
                                                          'bn_scale_nolearning'})
    else:
        raise NotImplementedError
    return True


def generate_one_optimizer_config(optimizer_type, lr=None, bn_scale_nolearning=False):
    config = {'optimizer_type': optimizer_type}
    if optimizer_type == 'sgd':
        config['lr'] = 0.01 if lr is None else lr
        config['momentum'] = 0.9
    elif optimizer_type == 'adam':
        config['lr'] = 0.001 if lr is None else lr
    else:
        raise NotImplementedError

    config['bn_scale_nolearning'] = bn_scale_nolearning

    assert sanity_check_one_optimizer_opt_config(config)
    return config


def generate_one_opt_config(conv_config_list, fc_config, loss, optimizer, legacy=False):
    config = dict()
    config['conv'] = conv_config_list
    config['fc'] = fc_config
    config['loss'] = loss
    config['optimizer'] = optimizer
    config['legacy'] = legacy
    assert sanity_check_opt_config(config)
    return config


def generate_one_conv_layer_opt_config(group, smoothness):
    # by default, l1_bias = l1, l2_bias = l2.
    # so to stop them from learning you need to set them to zero.

    config = dict()
    config['group'] = group
    config['smoothness'] = smoothness

    assert sanity_check_one_conv_layer_opt_config(config)
    return config


def generate_one_fc_layer_opt_config(scale):
    # by default, l1_bias = l1, l2_bias = l2.
    # so to stop them from learning you need to set them to zero.

    config = dict()
    config['scale'] = scale

    assert sanity_check_one_fc_layer_opt_config(config)
    return config


_type_checker = {
    'group': float,
    'smoothness': float,
    'scale': float,
    'conv': sanity_check_conv_opt_config,
    'fc': sanity_check_one_fc_layer_opt_config,
    'loss': lambda x: x in {'mse', 'poisson'},
    'optimizer': sanity_check_one_optimizer_opt_config,
    'optimizer_type': lambda x: x in {'adam', 'sgd'},
    'lr': float,
    'momentum': float,
    'legacy': bool,
    'bn_scale_nolearning': bool,
}

#
