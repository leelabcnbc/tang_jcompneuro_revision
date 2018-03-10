"""training should invoke this function"""

import numpy as np
import torch
from torch import nn, FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from . import training
from .cnn import get_loss, get_optimizer
from .eval import eval_fn_cnn_training as eval_fn


def count_params(model: nn.Module):
    count = 0
    for y in model.parameters():
        count += np.product(y.size())
    return count


def _check_dataset_shape(X: np.ndarray, y: np.ndarray):
    if not (X is None and y is None):
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (X.ndim == 4 or X.ndim == 2) and y.ndim == 2
        # print(X.shape, y.shape)
        assert X.shape[0] == y.shape[0] and X.shape[0] > 0
        return FloatTensor(X), FloatTensor(y)
    else:
        return None, None


def generate_datasets(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray = None, y_test: np.ndarray = None,
                      X_val: np.ndarray = None, y_val: np.ndarray = None, *,
                      batch_size=128, per_epoch_train=True, shuffle_train=True):
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
                   seed=None, legacy=False, legacy_epoch=75,
                   shuffle_train=True, show_every=1000, return_val_perf=False,
                   max_epoch=20000):
    assert len(datasets) == 6
    dataset_train, dataset_test, dataset_val = generate_datasets(
        *datasets, per_epoch_train=not legacy, shuffle_train=shuffle_train
    )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    loss_fn = get_loss(opt_config, model)
    optimizer = get_optimizer(model, opt_config['optimizer'])

    if legacy:
        # simply 75 iterations as before.
        phase1_dict = {
            'max_epoch': legacy_epoch,
            'lr_config': None,
            'early_stopping_config': None,
        }

        phase_config_dict_list = [phase1_dict, ]
        global_config_dict = {
            'convert_data_to_gpu': True,
            'loss_every_iter': 100000,  # don't show.
            'show_every': 100000,  # don't show.
        }
    else:
        # the datasets need to be
        phase1_dict = {
            'max_epoch': max_epoch,  # at most 200 * 100 minibatches.
            'lr_config': None,
            'early_stopping_config': {'patience': 10},
        }

        phase_config_dict_list = [phase1_dict, ]
        global_config_dict = {
            'convert_data_to_gpu': True,
            'loss_every_iter': 20,  # show loss every 20 iterations,
            'val_every': 100,  # 100x128 is about 12800 stimuli.
            'test_every': 100,
            # 'output_loss' is what we care actually,
            # such as MSE, or corr, etc.
            'early_stopping_field': 'neg_corr',
            'show_every': show_every,
        }

    # then train.
    training.train(model, loss_fn, dataset_train, optimizer,
                   phase_config_dict_list, dataset_val=dataset_val, eval_fn=eval_fn,
                   global_config_dict=global_config_dict, dataset_test=dataset_test)

    return_list = []

    if return_val_perf:
        yhat_val, y_val = training.eval_wrapper(model, dataset_val, True, None)
        val_perform = eval_fn(yhat_val, y_val)['corr']
        assert np.isscalar(val_perform) and np.isfinite(val_perform)
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
