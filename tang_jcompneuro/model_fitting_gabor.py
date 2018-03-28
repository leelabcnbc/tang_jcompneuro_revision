from functools import partial
import numpy as np


def gabor_training_wrapper(x: np.ndarray, y: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray, class_this, class_params=None,
                           num_batch=None,
                           batch_size=None):
    from .gabor import (gabor_training_wrapper,
                        init_and_predict_one_net_from_extracted_params,
                        save_model_to_hdf5_group)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert y.ndim == 1 and x.ndim == 4 and x.dtype == y.dtype == np.float32
    # cv_obj = KFold(n_splits=num_fold_global, shuffle=True, random_state=cv_seed)
    # cv_obj = cv_obj.split(y)  # I use y as it's 2d. 4d is not supported by sklearn.
    # cv_results = []
    # y_recon = np.full_like(y, fill_value=np.nan)
    if x_test is None and y_test is None:
        # copy just for safe
        x_test = x.copy()
        y_test = y.copy()
    assert isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray)
    assert y_test.ndim == 1 and x_test.ndim == 4 and x_test.dtype == y_test.dtype == np.float32

    best_corr, best_params = gabor_training_wrapper(x, y, class_this, class_params,
                                                    num_batch, batch_size)[:2]
    final_result, data_this = init_and_predict_one_net_from_extracted_params(x_test, y_test,
                                                                             class_this, best_params,
                                                                             class_params=None)
    results = {
        'corr': final_result,
        'y_test_hat': data_this,
        # well, this is actually on training.
        'attrs': {'corr_train': best_corr,
                  },
        'model': partial(save_model_to_hdf5_group, saved_params=best_params)
    }

    return results


def decompose_model_subtype(model_subtype):
    if model_subtype == 'simple':
        class_this, class_params = 'simple', None
    elif model_subtype == 'complex':
        class_this, class_params = 'complex', None
    else:
        class_this, num_simple, num_complex = model_subtype.split(',')
        assert class_this == 'multi'
        num_simple = int(num_simple)
        num_complex = int(num_complex)
        assert num_simple > 0 and num_complex > 0
        class_params = {
            'num_simple': num_simple,
            'num_complex': num_complex,
        }
    return class_this, class_params


def get_trainer(model_subtype, cudnn_enabled=False, cudnn_benchmark=False):
    class_this, class_params = decompose_model_subtype(model_subtype)
    print(class_this, class_params)

    def trainer(datasets):
        # best performance in my experiments.
        from torch.backends import cudnn
        cudnn.enabled = cudnn_enabled
        cudnn.benchmark = cudnn_benchmark
        # print(cudnn.enabled, cudnn.benchmark)
        assert cudnn.enabled == cudnn_enabled and cudnn.benchmark == cudnn_benchmark

        x, y, x_test, y_test, x_val, y_val = datasets
        assert y.ndim == 2 and y.shape[1] == 1
        y = y[:, 0]
        y = y.astype(np.float32, copy=False)
        x = x.astype(np.float32, copy=False)
        if y_test is not None:
            assert y_test.ndim == 2 and y_test.shape[1] == 1
            y_test = y_test[:, 0]
            y_test = y_test.astype(np.float32, copy=False)
            assert x_test is not None
            x_test = x_test.astype(np.float32, copy=False)
        else:
            assert x_test is None

        # check input size
        assert x.ndim == 4 and x.shape[1:] == (1, 20, 20)
        assert x_val is None and y_val is None

        return gabor_training_wrapper(x, y, x_test, y_test, class_this, class_params)

    return trainer


models_to_train = (
    'simple',
    'complex',
    'multi,1,1',
    'multi,2,1',
    'multi,1,2',
)
