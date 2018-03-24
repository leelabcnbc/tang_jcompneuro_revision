"""a very generic nested training loop

the learning has a nested loop structure.

I call the outer loop phases.

In each phase, I may set a different learning rate
and different early quitting criteria.

each phase has a max iteration, such as 5000.


Inside each phase, there is another loop that loops over epochs.

at end of every epoch, we compute training set stats.
(optionally) at end of every N_val epoch, we compute validation stats.
# we keep track of the best model epoch number. for val.
(optionally) at end of every N_test epoch, we compute testing stats.

validation stats can be used to do early stopping, after `patience` number of iterations.


right now, let's not think about training recovery, which is probably never needed for my projects here.

if we really need that, we just do it manually.

"""

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tempfile import TemporaryFile
import numpy as np


def cycle_reboot(iterable):
    # adapted from https://docs.python.org/3/library/itertools.html#itertools.cycle
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    while True:
        for element in iterable:
            yield element


def infinite_n_batch_loader(loader: DataLoader, n=1):
    # returns a function, that every time you call generates a new function
    # with n batches
    # I think to be safe, each loader should only be used once
    # with such a function.
    loader = iter(cycle_reboot(loader))  # ok. this has converted the loader into an infinite one.

    # if loader has shuffle=True, then it will give different things forever.
    #
    #
    #
    def generate_new_n_batch():
        for _ in range(n):
            next_batch = next(loader)
            yield next_batch

    return generate_new_n_batch


def train(model: nn.Module, loss_func, dataset_train, optimizer: optim.Optimizer,
          phase_config_dict_list: list = None, *,
          dataset_val: DataLoader = None, dataset_test: DataLoader = None, eval_fn=None,
          global_config_dict: dict = None):
    """

    :param model:
    :param dataset: this can be either a DataLoader; otherwise, it's assumed to be some callable
                    that will return a new thing that mimics a DataLoader every time.
                    typically, infinite_n_batch_loader should be made
    :param optimizer: an Optimizer
    :param phase_config_dict_list:
    :return:
    """

    global_config_dict_default = {
        'convert_data_to_gpu': True,
        'loss_every_iter': 20,  # show loss every 20 iterations,
        'val_every': 1,
        'test_every': 1,
        # 'output_loss' is what we care actually,
        # such as MSE, or corr, etc.
        'early_stopping_field': 'output_loss',
        'show_every': 1,
    }

    if global_config_dict is None:
        global_config_dict = dict()
    assert global_config_dict_default.keys() >= global_config_dict.keys()
    global_config_dict_default.update(global_config_dict)
    global_config_dict = global_config_dict_default

    num_phase = len(phase_config_dict_list)

    for i_phase, phase_config_dict in enumerate(phase_config_dict_list):
        print(f'========starting phase {i_phase+1}/{num_phase}==========')

        train_one_phase(model, loss_func, dataset_train,
                        optimizer, phase_config_dict, global_config_dict, dataset_val, dataset_test,
                        eval_fn)

        print(f'========end phase {i_phase+1}/{num_phase}==========')


def _update_lr(optimizer: optim.Optimizer, lr_config):
    update_type = lr_config['type']
    if update_type == 'reduce_by_factor':
        assert lr_config.keys() == {'type', 'factor'}

        # then update.
        # follow pytorch sample.
        factor = lr_config['factor']

        update_func = lambda x, i: x * factor if np.isscalar(factor) else x * factor[i]
    elif update_type == 'fixed':
        assert lr_config.keys() == {'type', 'fixed_lr'}

        fixed_lr = lr_config['fixed_lr']
        update_func = lambda x, i: fixed_lr if np.isscalar(factor) else fixed_lr[i]

    else:
        raise NotImplementedError

    for idx, p in enumerate(optimizer.param_groups):
        old_lr = p['lr']
        # therefore, a old_lr == 0 will stay 0.
        if old_lr > 0:
            new_lr = update_func(old_lr, idx)
            print("for grp of size {}, reducing lr from {:.6f} to {:.6f}".format(len(p['params']),
                                                                                 old_lr, new_lr))
            p['lr'] = new_lr
        else:
            assert old_lr == 0
            print("for grp of size {}, lr stays zero".format(len(p['params'])))
    return


def eval_wrapper(model: nn.Module, dataset: DataLoader, send_to_gpu, eval_fn):
    # some part inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py
    #
    # collect both output and target
    model.eval()
    assert not model.training
    labels_all = []
    outputs_all = []
    for i_minibatch, (inputs, labels) in enumerate(dataset):
        labels_all.append(labels.cpu().numpy().copy())
        if send_to_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model(inputs).data.cpu().numpy()
        outputs_all.append(outputs.copy())

    labels_all = np.concatenate(labels_all, 0)
    outputs_all = np.concatenate(outputs_all, 0)
    model.train()

    if eval_fn is not None:
        return eval_fn(outputs_all, labels_all)
    else:
        return outputs_all, labels_all


def train_one_phase(model, loss_func, dataset_train, optimizer: optim.Optimizer,
                    phase_config_dict, global_config_dict, dataset_val, dataset_test,
                    eval_fn):
    model.train()
    loss_every = global_config_dict['loss_every_iter']
    val_every = global_config_dict['val_every']
    test_every = global_config_dict['test_every']
    show_every = global_config_dict['show_every']

    max_epoch = phase_config_dict['max_epoch']
    early_stopping_config = phase_config_dict['early_stopping_config']
    lr_config = phase_config_dict['lr_config']

    if lr_config is not None:
        _update_lr(optimizer, lr_config)

    es_field = global_config_dict['early_stopping_field']

    if early_stopping_config is not None:
        early_stopping_patience = early_stopping_config['patience']
        # like keras.
        early_stopping_best = np.inf  # should be smaller.
        early_stopping_wait = 0
    else:
        early_stopping_patience = None
        early_stopping_best = None
        early_stopping_wait = None

    early_stopping_break = False

    with TemporaryFile() as f_best:
        for i_epoch in range(max_epoch):

            print_flag = i_epoch != 0 and i_epoch % show_every == 0

            if print_flag:
                print(f'========starting epoch {i_epoch}==========')

            # load dataset
            if isinstance(dataset_train, DataLoader):
                dataset = dataset_train
            else:
                dataset = dataset_train()

            # do the standard thing.
            for i_minibatch, (inputs, labels) in enumerate(dataset):
                # double check that I'm training properly.
                # in other words, I'm not doing dropout improperly.
                assert model.training
                if global_config_dict['convert_data_to_gpu']:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels, model)
                loss.backward()
                optimizer.step()

                # then let's do things.
                if print_flag and loss_every is not None and i_minibatch != 0 and i_minibatch % loss_every == 0:
                    print(f'{i_epoch}-{i_minibatch}, train loss {loss.data.cpu().numpy()[0]}')

            # show fc weights
            # turned up when I run `/results_ipynb/debug/cnn/cnn_wrapper_bn.ipynb`.
            # print(torch.sum(model.fc.fc.weight.data**2))

            if dataset_val is not None and val_every is not None and i_epoch % val_every == 0:
                assert eval_fn is not None
                # then print some data for validation set
                val_metric = eval_wrapper(model, dataset_val, global_config_dict['convert_data_to_gpu'], eval_fn)
                if print_flag:
                    print('val metric\n', val_metric)
                assert val_metric is not None
            else:
                val_metric = None

            if dataset_test is not None and test_every is not None and i_epoch % test_every == 0:
                assert eval_fn is not None
                # then print some data for validation set
                test_metric = eval_wrapper(model, dataset_test, global_config_dict['convert_data_to_gpu'], eval_fn)
                if print_flag:
                    print('test metric\n', test_metric)

            if print_flag:
                print(f'========done epoch {i_epoch}==========')

            # do early stopping stuff.
            # val_metric is not None means we evaluted dataset_val this time.
            if early_stopping_config is not None and val_metric is not None:
                assert np.isfinite(val_metric[es_field]), 'validation metric must be finite'
                if val_metric[es_field] < early_stopping_best:
                    early_stopping_best = val_metric[es_field]  # should be smaller.
                    early_stopping_wait = 0

                    # truncate file and save current one
                    # https://stackoverflow.com/questions/17126037/how-to-delete-only-the-content-of-file-in-python
                    f_best.seek(0)
                    f_best.truncate()
                    # follow https://github.com/pytorch/examples/blob/master/imagenet/main.py
                    torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, f_best)
                    f_best.seek(0)
                    # print('save once')
                else:
                    early_stopping_wait += 1
                    # print(f'patience {early_stopping_wait}')
                    if early_stopping_wait >= early_stopping_patience:
                        print(f'early stopping after epoch {i_epoch}')

                        # recover
                        checkpoint = torch.load(f_best)
                        model.load_state_dict(checkpoint['state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        early_stopping_break = True
                        break
        # load best if there is one.
        if early_stopping_config is not None and early_stopping_best != np.inf and not early_stopping_break:
            # recover.
            print(f'recover best model after {max_epoch} epochs')
            checkpoint = torch.load(f_best)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
