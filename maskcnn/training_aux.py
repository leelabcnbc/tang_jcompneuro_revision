"""training should invoke this function"""

import numpy as np
from functools import partial
import torch
from torch import nn, FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from tang_jcompneuro import training
from .cnn_aux import get_loss, get_optimizer
from scipy.stats import pearsonr


def eval_fn_wrapper(yhat_all, y_all, loss_type, return_corr=False):
    # use poisson loss.
    # torch.mean(torch.sum(yhat - y * torch.log(yhat + 1e-5), 1))
    assert yhat_all.shape == y_all.shape
    assert y_all.ndim == 2
    corr_each = []
    for yhat, y in zip(yhat_all.T, y_all.T):
        assert yhat.shape == y.shape == (y_all.shape[0],)
        pearson_this = pearsonr(yhat, y)[0]
        if not np.isfinite(pearson_this):
            pearson_this = 0.0
        assert np.isfinite(pearson_this) and np.isscalar(pearson_this)
        corr_each.append(pearson_this)
    corr_each = np.array(corr_each)
    if loss_type == 'poisson':
        return {'loss': np.mean(yhat_all - y_all * np.log(yhat_all + 1e-5)),
                # this complicated stuff is coped from standalone.py
                'corr': corr_each if return_corr else None,
                'corr2_mean': (corr_each ** 2).mean(),
                }
    elif loss_type == 'mse':
        return {'loss': np.mean((yhat_all - y_all) ** 2),
                # this complicated stuff is coped from standalone.py
                'corr': corr_each if return_corr else None,
                'corr2_mean': (corr_each ** 2).mean(),
                }
    else:
        raise NotImplementedError


def count_params(model: nn.Module):
    count = 0
    for y in model.parameters():
        count += np.product(y.size())
    return count


def _check_dataset_shape(X: np.ndarray, y: np.ndarray):
    if not (X is None and y is None):
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert X.ndim == 4 and y.ndim == 2
        # print(X.shape, y.shape)
        assert X.shape[0] == y.shape[0] and X.shape[0] > 0
        return FloatTensor(X), FloatTensor(y)
    else:
        return None, None


def generate_datasets(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray = None, y_test: np.ndarray = None,
                      X_val: np.ndarray = None, y_val: np.ndarray = None, *,
                      batch_size=256, per_epoch_train=True, shuffle_train=True):
    X_train, y_train = _check_dataset_shape(X_train, y_train)
    X_test, y_test = _check_dataset_shape(X_test, y_test)
    X_val, y_val = _check_dataset_shape(X_val, y_val)

    if per_epoch_train:
        assert X_train.size()[0] >= per_epoch_train

    dataset_train = TensorDataset(X_train, y_train)
    if per_epoch_train:
        dataset_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train,
                                   drop_last=True)  # since we drop_last, that's why X_train has to be long enough.
        dataset_train = training.infinite_n_batch_loader(dataset_train, n=1)
    else:
        dataset_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                   drop_last=False)

    if X_test is not None and y_test is not None:
        dataset_test = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    else:
        dataset_test = None
    if X_val is not None and y_val is not None:
        dataset_val = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    else:
        dataset_val = None

    return dataset_train, dataset_test, dataset_val


def train_one_case(model, datasets, opt_config,
                   seed=None,
                   shuffle_train=True, show_every=1000, return_val_perf=False,
                   max_epoch=20000,
                   eval_loss_type='poisson',
                   ):
    assert len(datasets) == 6
    dataset_train, dataset_test, dataset_val = generate_datasets(
        *datasets, per_epoch_train=True, shuffle_train=shuffle_train
    )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    loss_fn = get_loss(opt_config, model)
    optimizer = get_optimizer(model, opt_config['optimizer'])

    # the datasets need to be
    phase1_dict = {
        'max_epoch': max_epoch,
        'lr_config': None,
        'early_stopping_config': {'patience': 10},
    }

    phase2_dict = {
        'max_epoch': max_epoch,
        'lr_config': {'type': 'reduce_by_factor', 'factor': 1 / 3},
        'early_stopping_config': {'patience': 10},
    }

    phase3_dict = {
        'max_epoch': max_epoch,
        'lr_config': {'type': 'reduce_by_factor', 'factor': 1 / 3},
        'early_stopping_config': {'patience': 10},
    }

    phase_config_dict_list = [phase1_dict, phase2_dict, phase3_dict]
    global_config_dict = {
        'convert_data_to_gpu': True,
        'loss_every_iter': 1,
        'val_every': 50,  # 256 is about 12800 stimuli.
        'test_every': 50,
        # 'output_loss' is what we care actually,
        # such as MSE, or corr, etc.
        'early_stopping_field': 'loss',
        'show_every': show_every,
    }
    eval_fn = partial(eval_fn_wrapper, loss_type=eval_loss_type)
    training.train(model, loss_fn, dataset_train, optimizer,
                   phase_config_dict_list, dataset_val=dataset_val,
                   eval_fn=eval_fn,
                   global_config_dict=global_config_dict, dataset_test=dataset_test)

    return_list = []

    eval_fn = partial(eval_fn_wrapper, loss_type=eval_loss_type, return_corr=True)

    if return_val_perf:
        yhat_val, y_val = training.eval_wrapper(model, dataset_val, True, None)
        val_perform = eval_fn(yhat_val, y_val)['corr']
        assert np.all(np.isfinite(val_perform))
    else:
        val_perform = None

    if return_val_perf:
        return_list.append(val_perform)

    # finally, output prediction.
    if dataset_test is not None:
        yhat, y = training.eval_wrapper(model, dataset_test, True, None)
        return_list.append(yhat)
        return_list.append(eval_fn(yhat, y)['corr'])

    if len(return_list) == 0:
        return None
    else:
        return tuple(return_list)
