import numpy as np
from scipy.stats import pearsonr


def eval_fn_handle_datatype(yhat, y, data_type=None):
    if data_type == np.float32:
        assert yhat.dtype == y.dtype == np.float32
    elif data_type == np.float64:
        assert yhat.dtype == y.dtype == np.float64
    else:
        assert data_type is None

    assert yhat.shape == (yhat.size, 1)
    assert y.shape == (y.size, 1)
    assert yhat.shape == y.shape


def _pearson(yhat, y):
    loss = pearsonr(yhat.ravel(), y.ravel())[0]
    if not np.isfinite(loss):
        loss = 0.0
    return loss


def eval_fn_corr_raw(yhat, y, data_type=None):
    eval_fn_handle_datatype(yhat, y, data_type)
    return _pearson(yhat, y)


def eval_fn_cnn_training(yhat, y, data_type=np.float32):
    eval_fn_handle_datatype(yhat, y, data_type)

    loss = _pearson(yhat, y)
    mse_loss = np.mean((yhat - y) ** 2)
    return {'neg_corr': -loss,
            'corr': loss,
            'mse': mse_loss}
