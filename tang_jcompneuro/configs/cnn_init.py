"""initialization related configs"""

from collections import OrderedDict


def sanity_check_init_config(config):
    assert isinstance(config, dict)
    assert config.keys() == {'conv_init', 'fc_init'}
    for y in config.values():
        # kaiming fan out is used by my legacy code.
        assert isinstance(y, float) or y == 'kaiming_fan_out'
    return True


def generate_init_config(conv_init, fc_init):
    config = {
        'conv_init': conv_init,
        'fc_init': fc_init,
    }
    assert sanity_check_init_config(config)

    return config


def legacy_generator():
    return generate_init_config('kaiming_fan_out', 0.0001)


def generate_all_init():
    init_dict_ = OrderedDict()

    # all legacy one layer ones.
    init_dict_['legacy'] = legacy_generator()

    return init_dict_


init_dict = generate_all_init()
