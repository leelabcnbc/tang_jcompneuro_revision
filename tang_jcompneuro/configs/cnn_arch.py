"""this file stores all CNN setups we will try

# seperate structure and optimization.
that is we have both arch_configs and opt_configs.
"""
from collections import OrderedDict

from . import type_check_wrapper


def sanity_check_arch_config(config):
    # should have three parts
    return type_check_wrapper(config, _type_checker, {'fc', 'conv', 'act_fn', 'linear_output',
                                                      'conv_last_no_act'})


def generate_one_conv_config(kernel_size, out_channel, stride=1, bn=False, padding=0, pool=None,
                             dilation=1, bn_affine=True):
    config = {
        'kernel_size': kernel_size,
        'out_channel': out_channel,
        'stride': stride,
        'bn': bn,
        'padding': padding,
        'pool': pool,
        'dilation': dilation,
        'bn_affine': bn_affine,
    }

    assert _sanity_check_conv_list_config([config])
    return config


def generate_one_pool_config(kernel_size, stride=None, padding=0, pool_type='max'):
    if stride is None:
        # this is default setting of pytorch. also for VGG networks.
        stride = kernel_size
    config = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'pool_type': pool_type,
    }
    assert _sanity_pool_checker(config)
    return config


def generate_one_fc_config(factored=False, dropout=None, mlp=None, factored_constraint=None):
    config = {
        'factored': factored,
        'dropout': dropout,
        'mlp': mlp,
        'factored_constraint': factored_constraint,
    }
    assert _sanity_check_fc_config(config)
    return config


# actual generators
def generate_one_config(conv, fc, act_fn, linear_output, conv_last_no_act=False):
    config = dict()
    config['conv'] = conv
    config['fc'] = fc
    config['act_fn'] = act_fn
    config['linear_output'] = linear_output
    config['conv_last_no_act'] = conv_last_no_act
    assert sanity_check_arch_config(config)
    return config


def legacy_one_layer_generator(num_channel):
    pool_config = generate_one_pool_config(8, 4)

    return generate_one_config([
        generate_one_conv_config(9, num_channel, pool=pool_config)
    ], generate_one_fc_config(), 'relu', True)


def generate_all_arch():
    arch_dict_ = OrderedDict()

    # all legacy one layer ones.
    arch_dict_legacy_1l = OrderedDict()
    for num_channel in (6, 12, 18):
        # key is always str so that JSONize is easier.
        arch_dict_legacy_1l[str(num_channel)] = legacy_one_layer_generator(num_channel)

    arch_dict_['legacy_1L'] = arch_dict_legacy_1l

    return arch_dict_


# checkers

def _sanity_pool_checker(pool_config):
    if pool_config is not None:
        assert type_check_wrapper(pool_config,
                                  _type_checker,
                                  {'kernel_size', 'stride', 'padding', 'pool_type'})
    return True


def _sanity_check_conv_list_config(conv_config_list):
    assert isinstance(conv_config_list, list)
    for x in conv_config_list:
        assert type_check_wrapper(x, _type_checker, {
            'kernel_size', 'out_channel',
            'stride', 'bn',
            'padding',
            'pool',
            'dilation',
            'bn_affine',
        })
    return True


def _sanity_check_fc_config(fc_config):
    assert isinstance(fc_config, dict)
    assert fc_config.keys() == {'factored', 'dropout', 'mlp', 'factored_constraint'}
    assert isinstance(fc_config['factored'], bool)
    if fc_config['factored']:
        assert fc_config['dropout'] is None
        assert fc_config['mlp'] is None
        assert fc_config['factored_constraint'] in {None, 'abs'}
    else:
        assert fc_config['factored_constraint'] is None
        dropout = fc_config['dropout']
        assert dropout is None or isinstance(dropout, float)
        mlp = fc_config['mlp']
        assert mlp is None or isinstance(mlp, int)
    return True


# here, non-type things should return True on success.
_type_checker = {
    'kernel_size': int,
    'out_channel': int,
    'bn': bool,
    'bn_affine': bool,
    'stride': int,
    'padding': int,
    'pool': _sanity_pool_checker,
    'pool_type': lambda x: x in {'max', 'avg'},
    'conv': _sanity_check_conv_list_config,
    'fc': _sanity_check_fc_config,
    'act_fn': lambda x: x in {'relu', 'softplus', None, 'sq', 'halfsq', 'abs'},
    'linear_output': bool,
    'dilation': int,
    'conv_last_no_act': bool,
}

# actual arch dict
arch_dict = generate_all_arch()
