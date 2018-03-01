"""
OLD CNN for debugging.
https://github.com/leelabcnbc/tang_jcompneuro/blob/master/tang_jcompneuro/cnn.py

rewritten CNN module

each CNN training, can be paramterized by three sets of parameters.

1. architecture parameters. 1 layer, 2 layer, 3 layer, what unit type, etc.
2. sub model parameters, number of channels, etc.  I feel 1 and 2 are twisted together. but still, it's to
   have them separated. In practice, model size parameter can be very simple, such as (12,) (36,) (192,), etc.
   or can be very complicated. I need to write a special model size parameter decoder for every architecture.
   for simplicity, I also put weight initilization parameters here.
3. optimization parameters. I will leave this None first, just using default settings.
   later on, if needed, we can investigate this part.


so, every model will be encoded as a three part tuple.
I hope I can make this whole thing representable by a single string, connected by '/'.
1, 2 should be easy.
3 can be tricky, but since I don't work on it right now, it's fine.
I can think about it later.

I think it can done like this.
momentum,0.5,batch,200,lr,0.001.

I need some function to properly output the float numbers in correct number of digits.

or for pure precision, I can represent float numbers in terms of integer time some small number, say 0.00001

so momentum=0.5 should be written as 50000. this has benefit of guaranteed preserved precision.
anyway, they will be worked on later.
"""

import math
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch import nn, optim, FloatTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import h5py

num_fold_global = 5


def _fwd_hook(_m, _input, _output, name):
    # used to print out blob shape, for debugging.
    print(name, _output.data.size())


def save_model_to_hdf5_group(grp: h5py.Group, net: torch.nn.Module):
    # this will work regardless whether
    for x, y in net.named_parameters():
        if x not in grp:
            # currently, it's not big. So I don't do compression at all, for speed.
            grp.create_dataset(x, data=y.data.cpu().numpy())
            grp.file.flush()


def load_model_from_hdf5_group(grp: h5py.Group, net: torch.nn.Module, strict=True):
    # this should work when net is on CPU.
    for x, y in net.named_parameters():
        if x in grp:
            # print(x, type(y), type(y.data), y.data.size())
            # currently, it's not big. So I don't do compression at all, for speed.
            y.data[...] = FloatTensor(grp[x][...])
            # print(x, type(y), type(y.data), y.data.size())
        else:
            print(f'warning, {x} not initialized')
            if strict:
                raise RuntimeError('some variable not initialized')


class SingleNeuronNetCommon(nn.Module):
    def __init__(self):
        super().__init__()
        self.__hooks = []

    @staticmethod
    @abstractmethod
    def default_params() -> dict:
        raise NotImplementedError()

    # use this debug_(de)register pair to show feature map size during forward pass.
    def debug_register(self):
        # hack to show size of feature map.
        assert self.__hooks == []
        for x, y in self.named_modules():
            if not isinstance(y, nn.Sequential) and y is not self:
                self.__hooks.append(y.register_forward_hook(partial(_fwd_hook, name=(x, str(type(y))))))

    def debug_deregister(self):
        assert self.__hooks
        for x in self.__hooks:
            x.remove()

    def num_parameters(self):
        total_number_count = 0
        for (x, y) in self.named_parameters():
            total_number_count += np.product(y.size())
        return total_number_count

    # copied from <https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L42-L55>
    def initialize_weights(self, weight_std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, weight_std)
                m.bias.data.zero_()


class SingleNeuronNetBaseline(SingleNeuronNetCommon):
    def __init__(self, n_channel, num_filter, nonlinear_type, weight_std, pool_type):
        self.num_filter = num_filter
        super().__init__()
        pool_class = _get_pool_class(pool_type)
        self.nonlinear_type = nonlinear_type
        if nonlinear_type == 'ReLU':
            self.features = nn.Sequential(
                nn.Conv2d(n_channel, num_filter, kernel_size=9),
                nn.ReLU(inplace=True),
                # added a pooling layer, to reduce number of parameters. this also works as a regularization.
                pool_class(kernel_size=8, stride=4),
            )
        elif nonlinear_type == 'TanH':
            # bad performance. maybe, relatively, 10%+ worse than ReLU.
            self.features = nn.Sequential(
                nn.Conv2d(n_channel, num_filter, kernel_size=9),
                nn.Tanh(),
                # added a pooling layer, to reduce number of parameters. this also works as a regularization.
                pool_class(kernel_size=8, stride=4),
            )
        elif nonlinear_type == 'None':
            # bad performance. maybe, relatively, 10%+ worse than ReLU.
            self.features = nn.Sequential(
                nn.Conv2d(n_channel, num_filter, kernel_size=9),
                # added a pooling layer, to reduce number of parameters. this also works as a regularization.
                # simply conv then pool.
                pool_class(kernel_size=8, stride=4),
            )
        elif nonlinear_type == 'HalfSquare':
            # bad performance. maybe, relatively, 10%+ worse than ReLU.
            self.features1 = nn.Sequential(
                nn.Conv2d(n_channel, num_filter, kernel_size=9),
                nn.ReLU(),
                # added a pooling layer, to reduce number of parameters. this also works as a regularization.
            )

            self.features2 = nn.Sequential(
                pool_class(kernel_size=8, stride=4),
            )
        elif nonlinear_type == 'Square':
            # bad performance. maybe, relatively, 10%+ worse than ReLU.
            self.features1 = nn.Sequential(
                nn.Conv2d(n_channel, num_filter, kernel_size=9),
            )

            self.features2 = nn.Sequential(
                pool_class(kernel_size=8, stride=4),
            )
        else:
            raise ValueError('non recognizable non linearity!')

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * num_filter, 1),
        )

        self.initialize_weights(weight_std)

    def forward(self, x):
        if self.nonlinear_type not in {'HalfSquare', 'Square'}:
            x = self.features(x)
        else:
            x = self.features2(self.features1(x) ** 2)
        x = x.view(x.size(0), 2 * 2 * self.num_filter)
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        return dict(n_channel=1, num_filter=12, nonlinear_type='ReLU', weight_std=0.0001,
                    pool_type='max')


class SingleNeuronNetBaselineBigFilter(SingleNeuronNetCommon):
    def __init__(self, n_channel, num_filter, weight_std, pool_type):
        self.num_filter = num_filter
        super().__init__()
        pool_class = _get_pool_class(pool_type)
        self.features = nn.Sequential(
            nn.Conv2d(n_channel, num_filter, kernel_size=14),
            nn.ReLU(inplace=True),
            # added a pooling layer, to reduce number of parameters. this also works as a regularization.
            pool_class(kernel_size=5, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * num_filter, 1),
        )

        self.initialize_weights(weight_std)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2 * 2 * self.num_filter)
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        return dict(n_channel=1, num_filter=5, weight_std=0.0001, pool_type='max')


class SingleNeuronNetMLP(SingleNeuronNetCommon):
    def __init__(self, num_filter, weight_std):
        self.num_filter = num_filter
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(400, num_filter),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_filter, 1),
        )
        self.initialize_weights(weight_std)

    def forward(self, x):
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.features(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        return dict(num_filter=3, weight_std=0.0001)


class SingleNeuronNetBaselineBigFilterDilation(SingleNeuronNetCommon):
    def __init__(self, n_channel, num_filter, weight_std, pool_type):
        self.num_filter = num_filter
        super().__init__()
        pool_class = _get_pool_class(pool_type)
        self.features = nn.Sequential(
            nn.Conv2d(n_channel, num_filter, kernel_size=8, dilation=2),
            nn.ReLU(inplace=True),
            # added a pooling layer, to reduce number of parameters. this also works as a regularization.
            pool_class(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * num_filter, 1),
        )

        self.initialize_weights(weight_std)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2 * 2 * self.num_filter)
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        return dict(n_channel=1, num_filter=15, weight_std=0.0001, pool_type='max')


class SingleNeuronNetBaselineSmallFilter(SingleNeuronNetCommon):
    def __init__(self, n_channel, num_filter, weight_std, pool_type):
        self.num_filter = num_filter
        super().__init__()

        pool_class = _get_pool_class(pool_type)

        self.features = nn.Sequential(
            nn.Conv2d(n_channel, num_filter, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            # added a pooling layer, to reduce number of parameters. this also works as a regularization.
            pool_class(kernel_size=5, stride=3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * num_filter, 1),
        )

        self.initialize_weights(weight_std)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2 * 2 * self.num_filter)
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        return dict(n_channel=1, num_filter=27, weight_std=0.0001, pool_type='max')


class SingleNeuronNetBaselineStride2(SingleNeuronNetCommon):
    def __init__(self, n_channel, num_filter, weight_std, pool_type):
        self.num_filter = num_filter
        super().__init__()

        pool_class = _get_pool_class(pool_type)

        # this works poorly, probably due to non-dense sampling.
        self.features = nn.Sequential(
            # this would leave one pixel left. but should be fine...
            nn.Conv2d(n_channel, num_filter, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            # 6x6
            pool_class(kernel_size=4, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * num_filter, 1),
        )

        self.initialize_weights(weight_std)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2 * 2 * self.num_filter)
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        return dict(n_channel=1, num_filter=12, weight_std=0.0001, pool_type='max')


def _get_pool_class(pool_type):
    return {
        'max': nn.MaxPool2d,
        'avg': nn.AvgPool2d,
    }[pool_type]


class SingleNeuronNetTwoLayer(SingleNeuronNetCommon):
    def __init__(self, n_channel, num_filter_1, num_filter_2, weight_std, second_layer, pool_type):
        self.num_filter_1 = num_filter_1
        self.num_filter_2 = num_filter_2
        self.second_layer = second_layer
        super().__init__()

        pool_class = _get_pool_class(pool_type)

        if second_layer:
            self.features = nn.Sequential(
                nn.Conv2d(n_channel, num_filter_1, kernel_size=5, dilation=2),
                nn.ReLU(inplace=True),
                # added a pooling layer, to reduce number of parameters. this also works as a regularization.
                nn.Conv2d(num_filter_1, num_filter_2, kernel_size=3, dilation=2, padding=2),
                nn.ReLU(inplace=True),
                pool_class(kernel_size=8, stride=4),
            )

            self.classifier = nn.Sequential(
                nn.Linear(2 * 2 * num_filter_2, 1),
            )
        else:
            assert num_filter_2 is None
            self.features = nn.Sequential(
                # this gives very poor result. seems that dense sampling (stride=1) is important.
                # nn.Conv2d(n_channel, num_filter_1, kernel_size=8, stride=2),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=5, stride=2),

                nn.Conv2d(n_channel, num_filter_1, kernel_size=5, dilation=2),
                nn.ReLU(inplace=True),
                pool_class(kernel_size=8, stride=4),
            )

            self.classifier = nn.Sequential(
                nn.Linear(2 * 2 * num_filter_1, 1),
            )

        self.initialize_weights(weight_std)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2 * 2 * (self.num_filter_2 if self.second_layer else self.num_filter_1))
        x = self.classifier(x)
        return x

    @staticmethod
    def default_params() -> dict:
        # 9 and 9 gives roughly same number of parameters as baseline 12 channel models.
        return dict(n_channel=1, num_filter_1=9, num_filter_2=9, weight_std=0.0001, second_layer=True,
                    pool_type='max')


model_class_dict = {
    'baseline': SingleNeuronNetBaseline,
    'baseline_stride2': SingleNeuronNetBaselineStride2,
    '2layer': SingleNeuronNetTwoLayer,
    'baseline_bigfilter': SingleNeuronNetBaselineBigFilter,
    'baseline_bigfilter_dilation': SingleNeuronNetBaselineBigFilterDilation,
    'baseline_smallfilter': SingleNeuronNetBaselineSmallFilter,
    'mlp': SingleNeuronNetMLP,
}

max_keys_set_dict = {
    'baseline': {'num_filter', 'nonlinear_type', 'pool_type'},
    'baseline_bigfilter': {'num_filter', 'pool_type'},
    'baseline_bigfilter_dilation': {'num_filter', 'pool_type'},
    'baseline_smallfilter': {'num_filter', 'pool_type'},
    'baseline_stride2': {'num_filter', 'pool_type'},
    '2layer': {'num_filter_1', 'num_filter_2', 'second_layer', 'pool_type'},
    'mlp': {'num_filter'},
}


def _cnn_params_to_string_submodel(architecture_param: str, submodel_param) -> str:
    # this can handle most cases, as long as there's no float number.
    if submodel_param is None:
        submodel_param = {}
    model_default_params = model_class_dict[architecture_param].default_params()

    max_keys = sorted(max_keys_set_dict[architecture_param])
    assert submodel_param.keys() <= set(max_keys)
    for key in max_keys:
        if key not in submodel_param:
            submodel_param[key] = model_default_params[key]
            # then print it. this should be easy.
    # keys to throw away, for compatibility.
    assert submodel_param.keys() == set(max_keys)
    if 'pool_type' in submodel_param and submodel_param['pool_type'] == 'max':
        del submodel_param['pool_type']
        max_keys = [x for x in max_keys if x != 'pool_type']
    assert submodel_param.keys() == set(max_keys)
    submodel_param_str = '+'.join([f'{k},{submodel_param[k]}' for k in max_keys])
    return submodel_param_str


# a handful of default opts.
named_opt_dict = {
    'baseline': {},
    'adam_longer': {'momentum': None, 'opt_type': 'Adam', 'lr': 0.001, 'num_epoch': 150},
    'adam_longest': {'momentum': None, 'opt_type': 'Adam', 'lr': 0.001, 'num_epoch': 300},
    'middle_decay': {'weight_decay': 0.001},
    'middle_decay_shorter': {'weight_decay': 0.001, 'num_epoch': 50},
}


def get_named_submodel_param_list():
    named_submodel_param_list_list = OrderedDict()
    # for baseline
    submodel_param_list = OrderedDict()
    submodel_param_list['01_avg'] = {'num_filter': 1, 'pool_type': 'avg'}
    submodel_param_list['02_avg'] = {'num_filter': 2, 'pool_type': 'avg'}
    submodel_param_list['03_avg'] = {'num_filter': 3, 'pool_type': 'avg'}
    # submodel_param_list['04_avg'] = {'num_filter': 4, 'pool_type': 'avg'}
    submodel_param_list['06'] = {'num_filter': 6}
    submodel_param_list['12'] = None  # ~1000
    submodel_param_list['12_square'] = {'nonlinear_type': 'HalfSquare'}
    submodel_param_list['12_avg'] = {'pool_type': 'avg'}
    submodel_param_list['12_avg_square'] = {'pool_type': 'avg', 'nonlinear_type': 'HalfSquare'}
    submodel_param_list['12_square'] = {'nonlinear_type': 'HalfSquare'}

    submodel_param_list['12_avg_puresquare'] = {'pool_type': 'avg', 'nonlinear_type': 'Square'}
    submodel_param_list['12_puresquare'] = {'nonlinear_type': 'Square'}
    submodel_param_list['12_avg_linear'] = {'pool_type': 'avg', 'nonlinear_type': 'None'}
    submodel_param_list['12_linear'] = {'nonlinear_type': 'None'}

    submodel_param_list['18'] = {'num_filter': 18}  # ~1600
    submodel_param_list['24'] = {'num_filter': 24}  # ~2000
    submodel_param_list['30'] = {'num_filter': 30}  # ~2600
    named_submodel_param_list_list['baseline'] = submodel_param_list

    submodel_param_list = OrderedDict()
    # these two are for debugging purpose, showing model performance increase.
    submodel_param_list['01'] = {'num_filter': 1}
    submodel_param_list['02'] = {'num_filter': 2}
    submodel_param_list['03'] = {'num_filter': 3}
    # submodel_param_list['04'] = {'num_filter': 4}
    submodel_param_list['05'] = None  # 1000
    submodel_param_list['08'] = {'num_filter': 8}  # 1600
    submodel_param_list['10'] = {'num_filter': 10}  # 2000
    submodel_param_list['13'] = {'num_filter': 13}  # 2600

    submodel_param_list['01_avg'] = {'num_filter': 1, 'pool_type': 'avg'}
    submodel_param_list['02_avg'] = {'num_filter': 2, 'pool_type': 'avg'}
    submodel_param_list['03_avg'] = {'num_filter': 3, 'pool_type': 'avg'}
    # submodel_param_list['04_avg'] = {'num_filter': 4, 'pool_type': 'avg'}
    submodel_param_list['05_avg'] = {'pool_type': 'avg'}
    named_submodel_param_list_list['baseline_bigfilter'] = submodel_param_list

    # for 2layer
    submodel_param_list = OrderedDict()
    submodel_param_list['06+06'] = {'num_filter_1': 6, 'num_filter_2': 6}
    submodel_param_list['09+09'] = None  # 1000
    submodel_param_list['12+12'] = {'num_filter_1': 12, 'num_filter_2': 12}  # 1600
    submodel_param_list['15+15'] = {'num_filter_1': 15, 'num_filter_2': 15}  # ~2600
    named_submodel_param_list_list['2layer'] = submodel_param_list

    submodel_param_list = OrderedDict()
    submodel_param_list['01'] = {'num_filter': 1}
    submodel_param_list['03'] = None
    named_submodel_param_list_list['mlp'] = submodel_param_list

    return named_submodel_param_list_list


named_submodel_param_dict_dict = get_named_submodel_param_list()


def get_named_submodel_num_param():
    all_dict = OrderedDict()
    for architecture_param, subdict_this in named_submodel_param_dict_dict.items():
        dict_this = OrderedDict()
        for submodel_param_this_name, submodel_param_this_value in subdict_this.items():
            print(architecture_param, submodel_param_this_name)
            # load one for each, and show size of feature map
            net_this = _get_init_net(architecture_param, submodel_param_this_value, {'seed': 0}, cuda=False)
            # print all parameters.
            total_number_count = 0
            for (x, y) in net_this.named_parameters():
                print(x, y.size())
                total_number_count += np.product(y.size())
            assert total_number_count == net_this.num_parameters()
            del net_this
            dict_this[submodel_param_this_name] = total_number_count
        all_dict[architecture_param] = dict_this
    return all_dict


def named_submodel_param_str_dict_dict():
    named_submodel_param_str_list_list = deepcopy(named_submodel_param_dict_dict)
    for model_arch, submodel_dict in named_submodel_param_str_list_list.items():
        for submodel_name in submodel_dict:
            submodel_dict[submodel_name] = _cnn_params_to_string_submodel(model_arch, submodel_dict[submodel_name])
    return named_submodel_param_str_list_list


# then generate str version.
named_submodel_param_str_dict_dict = named_submodel_param_str_dict_dict()


def _cnn_params_to_string_opt(opt_param: dict, strict=True) -> str:
    for (k, v) in named_opt_dict.items():
        if v == opt_param:
            return k
    if not strict:
        return '<undefined>'
    else:
        raise NotImplementedError('no such opt named yet!')


def cnn_params_to_string_tuple(architecture_param: str, submodel_param=None, opt_param=None,
                               strict=True) -> (str, str, str):
    architecture_param_str = architecture_param
    submodel_param_str = _cnn_params_to_string_submodel(architecture_param, submodel_param)
    if opt_param is None:
        opt_param = {}
    opt_param_str = _cnn_params_to_string_opt(opt_param, strict=strict)

    return architecture_param_str, submodel_param_str, opt_param_str


def _get_init_net(architecture_param, submodel_param, opt_param: dict,
                  cuda=True):
    model_class: SingleNeuronNetCommon = model_class_dict[architecture_param]
    if submodel_param is None:
        submodel_param = dict()

    assert submodel_param.keys() <= model_class.default_params().keys()
    params_to_use = deepcopy(model_class.default_params())
    params_to_use.update(submodel_param)

    # for total determinism, you should disable cuDNN as well.
    #
    torch.manual_seed(opt_param['seed'])
    # from pytorch doc
    # If you are working with a multi-GPU model,
    # this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
    torch.cuda.manual_seed(opt_param['seed'])

    net_this = model_class(**params_to_use)
    if cuda:
        net_this.cuda()

    return net_this


def _get_optimizer(net_this: nn.Module, opt_param: dict):
    (opt_type, lr,
     momentum, weight_decay) = opt_param['opt_type'], opt_param['lr'], opt_param['momentum'], opt_param['weight_decay']
    if opt_type == 'SGD':
        optimizer_this = optim.SGD(net_this.parameters(), lr=lr,
                                   momentum=momentum, weight_decay=weight_decay)
    elif opt_type == 'Adam':
        assert momentum is None
        optimizer_this = optim.Adam(net_this.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'LBFGS':
        # no weight decay.
        assert momentum is None and weight_decay is None
        optimizer_this = optim.LBFGS(net_this.parameters(), lr=lr)
    else:
        raise ValueError('unrecognized optimizer')
    return optimizer_this


def one_train_loop(architecture_param, trainset, X_test=None, y_test=None,
                   submodel_param=None, opt_param=None, loss_every=25, verbose=False):
    """
    for simplicity, I assumed GPU models.
    :param architecture_param: what model class.
    :param trainset:
    :param X_test:
    :param y_test:
    :param submodel_param: detailed params for that model class.
    :param opt_param: params for optimization.
    :param loss_every:
    :return:
    """

    if X_test is None and y_test is None:
        test_flag = False
    elif X_test is not None and y_test is not None:
        test_flag = True
    else:
        raise ValueError('either both X_test and y_test be None or not None')

    opt_param_default = {
        'seed': 0,  # seed for torch.
        'batch_size': 128,
        'num_epoch': 75,
        'weight_decay': 0.0001,
        'lr': 0.1,
        'opt_type': 'SGD',
        'momentum': 0.9,
    }

    if opt_param is None:
        opt_param = {}
    opt_param_to_use = deepcopy(opt_param_default)
    opt_param_to_use.update(opt_param)
    net_this: SingleNeuronNetCommon = _get_init_net(architecture_param, submodel_param, opt_param_to_use)
    optimizer_this = _get_optimizer(net_this, opt_param_to_use)

    if verbose:
        print(opt_param_to_use)

    batch_size, num_epoch = opt_param_to_use['batch_size'], opt_param_to_use['num_epoch']

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion_this = nn.MSELoss(size_average=True)

    # then the main loop
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        loss = np.nan
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            # print(inputs.size(), labels.size())
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # zero the parameter gradients
            optimizer_this.zero_grad()
            # forward + backward + optimize
            outputs = net_this(inputs)
            loss = criterion_this(outputs, labels)
            loss.backward()
            optimizer_this.step()
        if loss_every is not None and (epoch + 1) % loss_every == 0:
            print('epoch {}, loss {}'.format(epoch + 1, loss))

    if test_flag:
        y_test_predict = net_this(Variable(FloatTensor(X_test).cuda())).data.cpu().numpy()
        final_result = pearsonr(y_test.ravel(), y_test_predict.ravel())[0]
    else:
        final_result = None
        y_test_predict = None

    return net_this, final_result, y_test_predict


def one_train_loop_cv(architecture_param, X: np.ndarray, y: np.ndarray,
                      cv_seed=0, submodel_param=None, opt_param=None, loss_every=None,
                      verbose=True, num_fold=num_fold_global):
    assert y.ndim == 2
    cv_obj = KFold(n_splits=num_fold, shuffle=True, random_state=cv_seed)
    cv_obj = cv_obj.split(y)  # I use y as it's 2d. 4d is not supported by sklearn.
    cv_results = []
    y_recon = np.full_like(y, fill_value=np.nan)

    for split_idx, (train, test) in enumerate(cv_obj):
        if verbose:
            print('working on fold {}/5'.format(split_idx + 1))
        X_train = FloatTensor(X[train])
        y_train = FloatTensor(y[train])
        X_test = X[test]
        y_test = y[test]
        finished_net, final_result, data_this = one_train_loop(architecture_param, TensorDataset(X_train, y_train),
                                                               X_test=X_test, y_test=y_test,
                                                               submodel_param=submodel_param, opt_param=opt_param,
                                                               loss_every=loss_every, verbose=verbose)
        # then compute.
        cv_results.append(final_result)
        y_recon[test] = data_this

    assert np.isfinite(y_recon).all()
    cv_results = np.asarray(cv_results)
    results = (y_recon, cv_results)

    return results
