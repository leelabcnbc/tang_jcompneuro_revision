from copy import deepcopy
from collections import OrderedDict

import numpy as np
import h5py
import torch
from torch import FloatTensor
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import relu
from torch.optim import Adam
from torch.nn import MSELoss
from scipy.stats import pearsonr


def save_model_to_hdf5_group(grp: h5py.Group, saved_params: dict):
    # this will work regardless whether
    for x, y in saved_params.items():
        if isinstance(y, dict):
            grp_this = grp.create_group(x)
            save_model_to_hdf5_group(grp_this, y)
        else:
            grp.create_dataset(x, data=y)


def load_model_from_hdf5_group(grp: h5py.Group):
    final_dict = {}
    for x in grp:
        if isinstance(grp[x], h5py.Group):
            final_dict[x] = load_model_from_hdf5_group(grp[x])
        elif isinstance(grp[x], h5py.Dataset):
            scalar_flag = grp[x].shape == ()
            final_dict[x] = grp[x][()] if scalar_flag else grp[x][...]
        else:
            raise RuntimeError
    return final_dict


def get_named_submodel_param_list():
    named_submodel_param_list_list = OrderedDict()
    # for baseline
    # potentially because loading data takes time.
    named_submodel_param_list_list['simple'] = (None,)
    named_submodel_param_list_list['complex'] = (None,)
    named_submodel_param_list_list['multi'] = (
        '1,1',  # 1 simple, 1 complex
        '1,2',  # 1 simple, 2 complex
        # # '1,3',  # 1 simple, 3 complex
        '2,1',  # 2 simple, 1 complex.
    )
    # TODO: multi ones should have non-None list.
    return named_submodel_param_list_list


named_submodel_param_list_dict = get_named_submodel_param_list()


# by checking the ImaGen Gabor code, we know there are two steps.
# 1. generating the coordinate system based on x,y,xdensity,ydensity,
# 2. evaluate gabor fn on those points.
# we will also follow this when implementing our pytorch version.

# this should not be called directly.
class GaborBase(nn.Module):
    def __init__(self, imsize,
                 batch_size, num_unit,
                 init_dict=None,
                 # these should be turned off when embedding Gabor modules in a multi Gabor module.
                 # well, currently, my code can handle multi complex and multi simple already.
                 # so probably those cases are rare.
                 output_b=True,
                 output_a=True,
                 seed=None):
        super().__init__()
        # according to doc http://pytorch.org/docs/master/nn.html#torch.nn.Module.register_buffer
        # "Buffers can be accessed as attributes using given names."
        # http://pytorch.org/docs/0.2.0/_modules/torch/nn/modules/module.html#Module.register_buffer

        # 1. generate coordinate system
        # this is at
        # https://github.com/ioam/imagen/blob/v2.1.0/imagen/patterngenerator.py#L154
        # and SheetCoordinateSystem is used.
        # let's first check result obtained by SheetCoordinateSystem and try to write one that mimics it.
        # another reference on this is at
        # http://ioam.github.io/topographica/User_Manual/coords.html
        # also
        # check
        # https://github.com/leelabcnbc/tang-paper-2017/blob/a6b91bd92e33892d1afaa4910dc17d312ce31725/feature_extraction/test_imagen.ipynb

        self.register_buffer('_basegrid_x',
                             Variable(torch.linspace(-imsize / 2 + 0.5, imsize / 2 - 0.5, imsize) / imsize, ))
        self.register_buffer('_basegrid_y',
                             Variable(-torch.linspace(-imsize / 2 + 0.5, imsize / 2 - 0.5, imsize) / imsize, ))
        # using buffer will handle conversion to GPU automatically.

        self.imsize = imsize
        self.imsize2 = imsize * imsize
        self.unit_array_shape = (batch_size, num_unit)
        assert batch_size > 0 and num_unit > 0
        # this is for generating init values.
        self._rng_state = np.random.RandomState(seed=seed)

        # unit_array_shape should a 2d thing,
        # num_trial x num_unit.
        # num_unit should be 1 for complex / simple, and multiple for multiple Gabor.
        # num_trial is basically a way to parallelize multiple runs from different initial locations.
        # it should be set relative

        # then set params.

        self.loc = nn.Parameter(torch.zeros(2, *self.unit_array_shape))  # x and y
        # sigmax and sigmay (normalized, not in pixels, but w.r.t. length of image frame)
        # will be passed into width and height in gabor in
        # follow https://github.com/ioam/imagen/blob/v2.1.0/imagen/patternfn.py#L95
        self.sigma = nn.Parameter(torch.zeros(2, *self.unit_array_shape))
        self.orientation = nn.Parameter(torch.zeros(*self.unit_array_shape))
        self.frequency = nn.Parameter(torch.zeros(*self.unit_array_shape))

        # I should have some way to initilize the weights.
        if init_dict is None:
            init_dict = self.__default_init()

        self._initialize_params(init_dict)

        self.output_a = nn.Parameter(torch.ones(*self.unit_array_shape)) if output_a else None
        self.output_b = nn.Parameter(torch.zeros(batch_size)) if output_b else None

        self._initialize_params_bias(init_dict)

        self._param_to_fix_dict = dict()
        # collect all params to fix later on.
        for (x, y) in self.named_parameters():
            if x in {'sigma', 'frequency', 'loc'}:
                self._param_to_fix_dict[x] = y

    # def _preprocess_input(self, x):
    #     return x.view(-1, self.imsize2) + self.im_bias
    def _initialize_params(self, init_dict):
        self.loc.data[...] = FloatTensor(np.asarray(init_dict['loc'], dtype=np.float32))
        self.orientation.data[...] = FloatTensor(np.asarray(init_dict['orientation'], dtype=np.float32))
        self.sigma.data[...] = FloatTensor(np.asarray(init_dict['sigma'], dtype=np.float32))
        self.frequency.data[...] = FloatTensor(np.array([init_dict['frequency']], dtype=np.float32))

    def _initialize_params_bias(self, init_dict):
        if self.output_a is not None and 'output_a' in init_dict:
            self.output_a.data[...] = FloatTensor(np.asarray(init_dict['output_a'], dtype=np.float32))
        if self.output_b is not None and 'output_b' in init_dict:
            self.output_b.data[...] = FloatTensor(np.asarray(init_dict['output_b'], dtype=np.float32))

    def _postprocess_output(self, y: Variable):
        # y has shape *self.unit_array_shape + (len(x),)
        # should return (unit_array_shape[0], len(x))
        if self.output_a is not None:
            output_scaled = y * self.output_a.view(*self.unit_array_shape, 1)
        else:
            output_scaled = y
        output_scaled = output_scaled.sum(dim=1)
        if self.output_b is not None:
            output_biased = output_scaled + self.output_b.view(self.unit_array_shape[0], 1)
        else:
            output_biased = output_scaled
        return output_biased

    def _generate_grid(self):
        # given current x and y (loc),
        # generate equivalent of _setup_xy in imagen.
        # equivalent of
        # https://github.com/ioam/imagen/blob/v2.1.0/imagen/patterngenerator.py#L246
        # use pytorch broadcasting mechanism.
        # None is pytorch's np.newaxis
        # basegrid is 1d. let's say (imsize,) loc is of shape *unit_array_shape + (1,).
        # so result should be *unit_array_shape + (imsize,)

        # don't use None stuff due to bug in https://github.com/pytorch/pytorch/issues/2741

        x_points = self._basegrid_x - self.loc[0].view(*self.unit_array_shape, 1)
        y_points = self._basegrid_y - self.loc[1].view(*self.unit_array_shape, 1)
        # print('xypoints', x_points.size(), y_points.size())
        # pattern_x and pattern_y should have shape *unit_array_shape + (imsize, imsize)
        # term_cos = torch.cos(self.orientation)[..., None, None]  # *unit_array_shape + (1,1)
        # term_sin = torch.sin(self.orientation)[..., None, None]  # *unit_array_shape + (1,1)
        # print('termsizes', term_cos.size(), term_sin.size())
        # # print('expanded xypoints',  y_points[..., None].size(), x_points[..., None, :].size())
        # # hack due to bug https://github.com/pytorch/pytorch/issues/2741
        # pattern_y = y_points[..., None] * term_cos - x_points[..., None, :] * term_sin
        # pattern_x = y_points[..., None] * term_sin + x_points[..., None, :] * term_cos
        # return pattern_x, pattern_y

        term_cos = torch.cos(self.orientation).view(*self.unit_array_shape, 1, 1)  # *unit_array_shape + (1,1)
        term_sin = torch.sin(self.orientation).view(*self.unit_array_shape, 1, 1)  # *unit_array_shape + (1,1)
        # print('termsizes', term_cos.size(), term_sin.size())
        # print('expanded xypoints',  y_points[..., None].size(), x_points[..., None, :].size())
        # hack due to bug https://github.com/pytorch/pytorch/issues/2741
        y_points_exp = y_points.view(*self.unit_array_shape, -1, 1)
        x_points_exp = x_points.view(*self.unit_array_shape, 1, -1)
        # print('expanded xypoints', y_points_exp.size(), x_points_exp.size())
        # pattern_y = y_points[..., None] * term_cos - x_points[..., None, :] * term_sin
        # pattern_x = y_points[..., None] * term_sin + x_points[..., None, :] * term_cos
        pattern_y = y_points_exp * term_cos - x_points_exp * term_sin
        pattern_x = y_points_exp * term_sin + x_points_exp * term_cos
        return pattern_x, pattern_y

    def _generate_gabor_inner(self, phase):
        # generate one actual Gabor.
        # follow https://github.com/ioam/imagen/blob/v2.1.0/imagen/patternfn.py#L95

        # https://discuss.pytorch.org/t/tensor-math-logical-operations-any-and-all-functions/6624/2
        assert (self.sigma > 0).data.all()
        # phase should have same size as self.frequency
        if isinstance(phase, Variable):
            phase = phase.view(*self.unit_array_shape, 1, 1)
        # otherwise, I would assume phase is some simple numpy or python float scalar.

        pattern_x, pattern_y = self._generate_grid()
        # print(pattern_x.size(), pattern_y.size())

        # both of shape *unit_array_shape + (imsize, imsize)
        # I must make sigma use first dims, as otherwise. 'Tensor.view' won't work, because
        # it's not contiguous.
        x_w = pattern_x / self.sigma[0].view(*self.unit_array_shape, 1, 1)
        y_h = pattern_y / self.sigma[1].view(*self.unit_array_shape, 1, 1)
        p = torch.exp(-0.5 * x_w * x_w + -0.5 * y_h * y_h)
        return p * 0.5 * torch.cos(2 * np.pi * self.frequency.view(*self.unit_array_shape, 1, 1) * pattern_y + phase)
        # return a lot of Gabors.  *unit_array_shape + (imsize, imsize)

    def forward(self, *_):
        # () or not () is the same.
        # https://stackoverflow.com/questions/16706956/is-there-a-difference-between-raise-exception-and-raise-exception-without
        raise NotImplementedError

    def _forward_inner(self, x, gabor_original_shape, bias):
        x_reshaped = self._reshape_input(x)
        #  *unit_array_shape, num_im
        out = torch.matmul(x_reshaped, self._reshape_gabor(gabor_original_shape)).view(*self.unit_array_shape,
                                                                                       x_reshaped.size()[0])
        # print('out shape', out.size())
        # I should try to permute dim to front, to make later computation easier.

        if bias is not None:
            out = out + bias.view(*self.unit_array_shape, 1)
        return out

    def _reshape_gabor(self, x: Variable):
        return x.view(*self.unit_array_shape, self.imsize2, 1)

    def _reshape_input(self, x: Variable):
        # I assumed you give num_im x 1 x imsize x imsize stuff. that 1 thing can be missed or multiple.
        # as it doesn't affect view in this case.
        return x.view(-1, self.imsize2)

    def init_output_bias(self, x: np.ndarray, y: np.ndarray):
        # adjust output_a and output_b, should work when model is on CPU.
        assert self.output_a is not None and self.output_b is not None
        assert y.shape == (y.size,)
        y_now = self.forward(Variable(FloatTensor(x))).data.cpu().numpy()
        assert y_now.shape == (self.unit_array_shape[0], y.size)
        # then, do linear regression for everyone.
        # assert y is not constant.
        # if y_now is constant, then it's probably bad initialization. stop.
        # compute std for every row of y_now, and assert they are all nonzero.
        alpha_beta_array = np.asarray([return_alpha_beta_simple_linear_regression(y_now_i, y) for y_now_i in y_now],
                                      dtype=np.float32)
        assert alpha_beta_array.shape == (self.unit_array_shape[0], 2)
        # for bad points, init to 0 and 1 instead. I don't think such points are optimizable any way.
        bad_index = np.logical_not(np.all(np.isfinite(alpha_beta_array), axis=1))
        print(f'bad index {bad_index.sum()}/{bad_index.size}')
        alpha_beta_array[bad_index] = np.array([0, 1])
        assert np.all(np.isfinite(alpha_beta_array))

        # set bias and scale
        # so they will work even for non 0 b and non 1 a.
        self.output_b.data[...] = self.output_b.data + FloatTensor(alpha_beta_array[:, 0])
        self.output_a.data[...] = self.output_a.data * FloatTensor(alpha_beta_array[:, 1, np.newaxis]).expand(
            *self.unit_array_shape)

    def __default_init(self) -> dict:
        # for each unit in unit_array_shape, generate a set of parameters.
        # check
        # https://github.com/leelabcnbc/sparse-coding-tang-final/blob/e9a4e3dadcc4be1bbc516f65e2f8b26655d0faf9/neuron_fitting/gabor_fitting_debug.ipynb
        # as reference
        # for x and y, I should try to make original 30x30 part covered as much as possible,
        # as that's all true RFs were supposed to be in that area.
        scale_factor = 0.5  # this is assumed.
        actual_portion = 30 / (self.imsize / scale_factor)  # [-actual_portion/2, actual_portion/2] in normalized x, y
        # covers the whole 30x30 RF bound.
        init_loc = self._rng_state.rand(2, *self.unit_array_shape) * actual_portion - actual_portion / 2

        # then frequency
        # use orignal ones. 2 - 8.
        init_freq = self._rng_state.rand(*self.unit_array_shape) * 6 + 2

        # then orientation.
        # easy.
        init_ori = self._rng_state.rand(*self.unit_array_shape) * np.pi

        # then sigma
        # 2 pixel to 4 pixel, as in original ones.
        sigma_lower, sigma_upper = 2 / self.imsize, 4 / self.imsize
        init_sigma = self._rng_state.rand(2, *self.unit_array_shape) * (sigma_upper - sigma_lower) + sigma_lower

        return {
            'loc': init_loc,
            'frequency': init_freq,
            'sigma': init_sigma,
            'orientation': init_ori,
        }

    def adjust_params(self):
        # make sure everything is in range
        self._param_to_fix_dict['loc'].data.clamp_(-1.25, 1.25)
        # at least one pixel.
        self._param_to_fix_dict['sigma'].data.clamp_(min=1 / self.imsize)
        self._param_to_fix_dict['frequency'].data.clamp_(min=0.1)


def init_and_predict_one_net_from_extracted_params(x: np.ndarray, y: np.ndarray, class_this, params,
                                                   class_params=None):
    # only tested when all bias terms are there.
    num_im, num_c, height, width = x.shape
    assert num_c == 1 and num_im > 0 and height == width and height > 0
    assert y.shape == (num_im,)

    # TODO make this work for Gabor multi as well.
    if class_params is None:
        class_params = {}
    init_dict = extend_fetch_params_to_init_dict(params)
    class_params_to_use = _extract_class_params_from_init_dict(init_dict)
    class_params_to_use.update(class_params)

    class_this_to_use = __gabor_class_dict[class_this]
    net_this: GaborBase = class_this_to_use(imsize=height, init_dict=init_dict,
                                            **class_params_to_use)
    net_this.cuda()
    output = net_this(Variable(FloatTensor(x).cuda())).data.cpu().numpy()
    del net_this
    assert output.shape == (1, num_im)
    assert output.dtype == np.float32
    corr = pearsonr(output[0], y)[0]
    if not np.isfinite(corr):
        corr = 0.0
    return corr, output.T


def gabor_training_wrapper(x: np.ndarray, y: np.ndarray, class_this, class_params=None, num_batch=None,
                           batch_size=None,
                           num_epoch=1000, lr=0.01, verbose=False,
                           seed_bias=1000,  # making complex and simple different seeds.
                           optimizer='adam'):
    """

    :param x:
    :param y:
    :param class_this:
    :param class_params:
    :param num_batch: how many batches. more means potentially more accurate fitting.
    :param batch_size:  how large is each batch.
           if None, it's designed to run on a 8G card, with 20x20 input, 10000 images.
    :return:
    """
    num_im, num_c, height, width = x.shape
    assert num_c == 1 and num_im > 0 and height == width and height > 0
    assert y.shape == (num_im,)

    class_this_to_use = __gabor_class_dict[class_this]

    if class_this in {'complex', 'simple'}:
        if class_params is None:
            class_params = {'num_unit': 1}
    else:
        assert class_this == 'multi'
        assert class_params is not None

    if class_this == 'complex':
        class_params_to_use = {
            'batch_size': 192 // class_params['num_unit'],  # 6G
        }
    elif class_this == 'simple':
        class_params_to_use = {
            'batch_size': 384 // class_params['num_unit'],  # 6G
        }
    else:
        assert class_this == 'multi'  # this way, it will consume same amount of memory as complex and simple.
        # When one of them is zero, it degrades to complex and simple.
        class_params_to_use = {
            'batch_size': 384 // (class_params['num_simple'] + 2 * class_params['num_complex']),
        }

    class_params_to_use.update(class_params)
    if batch_size is not None:
        class_params_to_use['batch_size'] = batch_size

    if num_batch is None:
        # 192 for complex and 384 for simple. should take similar amount of time.
        if class_this in {'complex', 'simple'}:
            num_batch = class_params['num_unit']
        else:
            assert class_this == 'multi'
            num_batch = class_params['num_simple'] + class_params['num_complex']
    if class_this in {'simple', 'complex'}:
        assert 'seed' not in class_params_to_use
    else:
        assert class_this == 'multi'
        assert 'seed_complex' not in class_params_to_use
        assert 'seed_simple' not in class_params_to_use

    pytorch_input_x = Variable(FloatTensor(x).cuda())
    pytorch_input_y = Variable(FloatTensor(y).cuda())
    best_corr = -np.inf
    best_params = None
    best_predict = None
    for i_batch in range(num_batch):
        if class_this in {'simple', 'complex'}:
            net_this: GaborBase = class_this_to_use(imsize=height, seed=i_batch, **class_params_to_use)
        else:
            assert class_this == 'multi'
            net_this: GaborBase = class_this_to_use(imsize=height, seed_complex=i_batch + seed_bias,
                                                    seed_simple=i_batch,
                                                    **class_params_to_use)
        # net_this: GaborBase = class_this_to_use(imsize=height, seed=i_batch, **class_params_to_use)
        # intialize
        net_this.init_output_bias(x, y)
        # training.
        net_this.cuda()
        assert optimizer == 'adam'
        optimizer_this = Adam(net_this.parameters(), lr=lr)
        # else:
        #     assert optimizer == 'lbfgs'
        #     optimizer_this = LBFGS(net_this.parameters(), lr=lr)
        criterion_this = MSELoss(size_average=True)
        loss = None
        # if optimizer == 'adam':
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            # zero the parameter gradients
            optimizer_this.zero_grad()
            # forward + backward + optimize
            outputs = net_this(pytorch_input_x)
            loss = criterion_this(outputs, pytorch_input_y.expand_as(outputs))
            loss.backward()
            optimizer_this.step()
            net_this.adjust_params()

            if verbose and (epoch + 1) % 200 == 0:
                print('epoch {}, loss {}'.format(epoch + 1, loss))
        # else:
        #     # THIS DOESN'T WORK, PERIOD.
        #     # it may work if it can exploit the block structure in the hessian.
        #     # as I actually train many sets of parameters together.
        #     # however, I don't know how to do it.
        #     assert optimizer == 'lbfgs'
        #     loss_dict = {'loss': None}
        #
        #     def closure():
        #         # correct the values of updated input image
        #         del loss_dict['loss']
        #         net_this.adjust_params()
        #         optimizer_this.zero_grad()
        #         # forward + backward + optimize
        #         outputs = net_this(pytorch_input_x)
        #         loss_this = criterion_this(outputs, pytorch_input_y.expand_as(outputs))
        #         loss_this.backward()
        #         loss_dict['loss'] = loss_this
        #         return loss_this
        #
        #     for epoch in range(num_epoch):
        #         optimizer_this.step(closure=closure)
        #         if verbose and (epoch + 1) % 20 == 0:
        #             print('epoch {}, loss {}'.format(epoch + 1, loss_dict['loss']))
        #     del loss_dict
        del loss
        # final adjustment.
        net_this.adjust_params()
        outputs = net_this(pytorch_input_x).data.cpu().numpy()
        all_corrs = np.array([pearsonr(x, y)[0] for x in outputs])

        all_corrs[np.logical_not(np.isfinite(all_corrs))] = 0
        # then take the index of the max.
        # save the params.
        assert np.all(np.isfinite(all_corrs))

        best_idx = np.argmax(all_corrs)
        best_corr_this = all_corrs[best_idx]

        if best_corr_this > best_corr:
            if verbose:
                print(f'update best corr from {best_corr} to {best_corr_this}')
            best_params = fetch_params(net_this, best_idx)
            best_corr = best_corr_this
            best_predict = outputs[best_idx].copy()
        else:
            if verbose:
                print(f'no update best corr as {best_corr} >= {best_corr_this}')
        if verbose:
            print(f'batch {i_batch+1}/{num_batch}, high corr cases {(all_corrs>0.99).sum()}/{all_corrs.size}')
            print(f'best corr up to now {best_corr}')
        del net_this
        del optimizer_this
        del criterion_this
    assert np.isfinite(best_corr) and best_params is not None
    # maybe more del helps memory releasing.
    del pytorch_input_x
    del pytorch_input_y
    return best_corr, best_params, best_predict


__params_to_fetch_dim = {
    'loc': (3, 1),
    'sigma': (3, 1),
    'orientation': (2, 0),
    'frequency': (2, 0),
    'output_a': (2, 0),
    'output_b': (1, 0),
    'bias1': (2, 0),
    'bias2': (2, 0),
    'phase': (2, 0),
    'bias': (2, 0),
}
# https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
# https://stackoverflow.com/a/26853961/3692822
__params_to_fetch_dim_mixed = {
    **{'module_simple.' + k: v for k, v in __params_to_fetch_dim.items()},
    **{'module_complex.' + k: v for k, v in __params_to_fetch_dim.items()},
    'output_b': (1, 0),
}


def fetch_params(net: nn.Module, idx: int) -> dict:
    if isinstance(net, GaborBase):
        param_dict = {}
        for (x, y) in net.named_parameters():
            assert x in __params_to_fetch_dim
            y_data = y.data.cpu().numpy()
            total_dim, dim_to_extract = __params_to_fetch_dim[x]
            assert y_data.ndim == total_dim
            param_dict[x] = y_data[(slice(None),) * dim_to_extract + (idx,)].copy()
            assert param_dict[x].ndim == total_dim - 1
    else:
        assert isinstance(net, GaborMixedCell)
        param_dict = {}
        param_dict_complex = {}
        param_dict_simple = {}
        output_b_seen = False
        for (x, y) in net.named_parameters():
            assert x in __params_to_fetch_dim_mixed
            if x != 'output_b':
                prefix, suffix = x.split('.')
                if prefix == 'module_simple':
                    param_dict_this = param_dict_simple
                elif prefix == 'module_complex':
                    param_dict_this = param_dict_complex
                else:
                    raise ValueError('not possible!')
            else:
                param_dict_this = param_dict
                suffix = x
                output_b_seen = True
            y_data = y.data.cpu().numpy()
            total_dim, dim_to_extract = __params_to_fetch_dim[suffix]
            assert y_data.ndim == total_dim
            param_dict_this[suffix] = y_data[(slice(None),) * dim_to_extract + (idx,)].copy()
            assert param_dict_this[suffix].ndim == total_dim - 1
        assert output_b_seen
        assert 'output_b' in param_dict  # we only focus on such case right now.
        if param_dict_complex != {}:
            param_dict['complex'] = param_dict_complex
        if param_dict_simple != {}:
            param_dict['simple'] = param_dict_simple

    return param_dict


def extend_fetch_params_to_init_dict(params):
    if 'simple' not in params and 'complex' not in params:
        params = deepcopy(params)
        for x in params:
            assert x in __params_to_fetch_dim
            total_dim, dim_to_extract = __params_to_fetch_dim[x]
            params[x] = params[x][(slice(None),) * dim_to_extract + (np.newaxis,)]
        return params
    else:
        new_params = dict()
        for x, y in params.items():
            if x in {'simple', 'complex'}:
                new_params[x] = extend_fetch_params_to_init_dict(y)
            else:
                assert x == 'output_b'
                new_params[x] = y[np.newaxis]
        return new_params


def _extract_class_params_from_init_dict(init_dict):
    if 'simple' not in init_dict and 'complex' not in init_dict:
        # should be a GaborSimple or GaborComplex. simply return {}
        return {'num_unit': init_dict['loc'].shape[-1]}
    else:
        assert 'output_b' in init_dict
        if 'simple' not in init_dict:
            num_simple = 0
        else:
            num_simple = init_dict['simple']['loc'].shape[-1]

        if 'complex' not in init_dict:
            num_complex = 0
        else:
            num_complex = init_dict['complex']['loc'].shape[-1]
        return {
            'num_simple': num_simple,
            'num_complex': num_complex,
        }


class GaborSimpleCell(GaborBase):
    def __init__(self, imsize, batch_size=1, num_unit=1,
                 bias=True, init_dict=None, **kwargs):
        super().__init__(imsize, batch_size, num_unit, init_dict, **kwargs)
        self.phase = nn.Parameter(torch.zeros(*self.unit_array_shape))
        self.bias = nn.Parameter(torch.zeros(*self.unit_array_shape)) if bias else None
        if init_dict is None:
            init_dict = self.__default_init()

        self.phase.data[...] = FloatTensor(np.asarray(init_dict['phase'], dtype=np.float32))

        if self.bias is not None and 'bias' in init_dict:
            self.bias.data[...] = FloatTensor(np.asarray(init_dict['bias'], dtype=np.float32))

    def _generate_gabor(self) -> Variable:
        return self._generate_gabor_inner(self.phase)

    def forward(self, x):
        out_inner = self._forward_inner(x, self._generate_gabor(), self.bias)
        out = relu(out_inner) ** 2
        return self._postprocess_output(out)

    def __default_init(self) -> dict:  # avoid clashing, so using __.
        return {
            'phase': self._rng_state.rand(*self.unit_array_shape) * np.pi * 2
        }


class GaborComplexCell(GaborBase):
    def __init__(self, imsize,
                 batch_size=1, num_unit=1,
                 bias=False, init_dict=None, **kwargs):
        super().__init__(imsize, batch_size, num_unit, init_dict, **kwargs)
        # self.bias1 = nn.Parameter(torch.zeros(*self.unit_array_shape)) if bias else None
        # self.bias2 = nn.Parameter(torch.zeros(*self.unit_array_shape)) if bias else None
        # # when bias is not zero, this really matters.
        # self.phase = nn.Parameter(torch.zeros(*self.unit_array_shape)) if bias else 0
        assert not bias, 'you should not have bias in complex cell in the first place!'
        if init_dict is None:
            init_dict = {}

            # if self.bias1 is not None and 'bias2' in init_dict:
            #     self.bias1.data[...] = FloatTensor(np.asarray(init_dict['bias1'], dtype=np.float32))
            # if self.bias2 is not None and 'bias2' in init_dict:
            #     self.bias2.data[...] = FloatTensor(np.asarray(init_dict['bias2'], dtype=np.float32))
            # if isinstance(self.phase, nn.Parameter) and 'phase' in init_dict:
            #     self.phase.data[...] = FloatTensor(np.asarray(init_dict['phase'], dtype=np.float32))

    def _generate_gabor(self):
        a1: Variable = self._generate_gabor_inner(0)
        a2: Variable = self._generate_gabor_inner(np.pi / 2)
        return a1, a2

    def forward(self, x: Variable):
        a1, a2 = self._generate_gabor()
        out1 = self._forward_inner(x, a1, None)
        out2 = self._forward_inner(x, a2, None)
        return self._postprocess_output(out1 ** 2 + out2 ** 2)


class GaborMixedCell(nn.Module):
    def __init__(self, imsize,
                 batch_size=1, num_simple=1, num_complex=1,
                 bias_simple=True,  # well, complex bias should not be allowed in the first place.
                 output_a=True,
                 output_b=True,
                 # you should use different seed for simple and complex.
                 init_dict=None, seed_simple=None, seed_complex=None):
        super().__init__()
        self.num_simple = num_simple
        self.num_complex = num_complex
        self.batch_size = batch_size

        # simplest way is to generate have two modules. this way,
        # I won't deal with slicing parameters to two parts,
        # one for complex, one for simple.
        # slicing will make parameter tensors not contiguous,
        # and that can cause great problems.
        if init_dict is None:
            init_dict = {'complex': None, 'simple': None}

        if self.num_simple > 0:
            self.module_simple = GaborSimpleCell(imsize, batch_size=batch_size,
                                                 num_unit=self.num_simple, bias=bias_simple,
                                                 init_dict=init_dict['simple'], output_a=output_a,
                                                 output_b=False, seed=seed_simple)
        else:
            self.module_simple = None
        if self.num_complex > 0:
            self.module_complex = GaborComplexCell(imsize, batch_size=batch_size, num_unit=self.num_complex,
                                                   bias=False, init_dict=init_dict['complex'], output_a=output_a,
                                                   output_b=False, seed=seed_complex)
        else:
            self.module_complex = None

        self.output_a = output_a

        self.output_b = nn.Parameter(torch.zeros(batch_size)) if output_b else None

        if self.output_b is not None and 'output_b' in init_dict:
            self.output_b.data[...] = FloatTensor(np.asarray(init_dict['output_b'], dtype=np.float32))

    def forward(self, x: Variable):
        if self.module_simple is not None:
            resp_simple = self.module_simple.forward(x)
        else:
            resp_simple = 0
        if self.module_complex is not None:
            resp_complex = self.module_complex.forward(x)
        else:
            resp_complex = 0
        # 0 should not matter in addition.
        if self.output_b is not None:
            output_biased = resp_simple + resp_complex + self.output_b.view(self.batch_size, 1)
        else:
            output_biased = resp_simple + resp_complex
        return output_biased

    def adjust_params(self):
        if self.module_simple is not None:
            self.module_simple.adjust_params()
        if self.module_complex is not None:
            self.module_complex.adjust_params()

    def init_output_bias(self, x: np.ndarray, y: np.ndarray):
        # adjust output_a and output_b, should work when model is on CPU.
        assert self.output_a and self.output_b is not None
        assert y.shape == (y.size,)
        y_now = self.forward(Variable(FloatTensor(x))).data.cpu().numpy()
        assert y_now.shape == (self.batch_size, y.size)
        # then, do linear regression for everyone.
        # assert y is not constant.
        # if y_now is constant, then it's probably bad initialization. stop.
        # compute std for every row of y_now, and assert they are all nonzero.
        alpha_beta_array = np.asarray([return_alpha_beta_simple_linear_regression(y_now_i, y) for y_now_i in y_now],
                                      dtype=np.float32)
        assert alpha_beta_array.shape == (self.batch_size, 2)
        # for bad points, init to 0 and 1 instead. I don't think such points are optimizable any way.
        bad_index = np.logical_not(np.all(np.isfinite(alpha_beta_array), axis=1))
        print(f'bad index {bad_index.sum()}/{bad_index.size}')
        alpha_beta_array[bad_index] = np.array([0, 1])
        assert np.all(np.isfinite(alpha_beta_array))

        # set bias and scale
        # so they will work even for non 0 b and non 1 a.
        self.output_b.data[...] = self.output_b.data + FloatTensor(alpha_beta_array[:, 0])
        # propagate the operation to sub modules.
        # self.output_a.data[...] = self.output_a.data * FloatTensor(alpha_beta_array[:, 1, np.newaxis]).expand(
        #     *self.unit_array_shape)
        if self.module_simple is not None:
            self.module_simple.output_a.data[...] = self.module_simple.output_a.data * FloatTensor(
                alpha_beta_array[:, 1, np.newaxis]).expand(self.batch_size,
                                                           self.num_simple)

        if self.module_complex is not None:
            self.module_complex.output_a.data[...] = self.module_complex.output_a.data * FloatTensor(
                alpha_beta_array[:, 1, np.newaxis]).expand(self.batch_size,
                                                           self.num_complex)


def return_alpha_beta_simple_linear_regression(x: np.ndarray, y: np.ndarray):
    # assert xstd != 0 and ystd != 0
    # <https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line>
    assert x.shape == y.shape and x.ndim == 1
    xstd = x.std()
    ystd = y.std()
    norm_beta = pearsonr(x, y)[0] * ystd / xstd
    norm_alpha = y.mean() - norm_beta * x.mean()
    # alpha is intercept, beta is scaling.
    assert np.isscalar(norm_alpha) and np.isscalar(norm_beta)
    # assert np.isfinite(norm_alpha) and np.isfinite(norm_beta)
    return norm_alpha, norm_beta


__gabor_class_dict = {
    'complex': GaborComplexCell,
    'simple': GaborSimpleCell,
    # 'complex_multi': GaborComplexCell,
    # 'simple_multi': GaborSimpleCell,
    'multi': GaborMixedCell,
}
