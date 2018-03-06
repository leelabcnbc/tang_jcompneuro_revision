"""helper functions to get convert data for glm fitting"""
import numpy as np
from sklearn.decomposition import PCA

# import time

from scipy.signal import hann
from scipy.fftpack import fft2

from strflab.feature_transformation import quadratic_features


def hann_2d(size):
    # I use False version, since FFT will be applied.
    window_1d = hann(size, sym=False)[np.newaxis]
    window_2d = window_1d.T @ window_1d
    window_2d /= window_2d.sum()
    return window_2d


def remixing_transformer(X):
    # typically, always perform a final remixing.
    assert X.ndim == 2
    pca_obj = PCA(svd_solver='full', n_components=None)
    pca_obj.fit(X)

    return (lambda x: pca_obj.transform(x.reshape(x.shape[0], -1))), pca_obj.explained_variance_ratio_.copy()


def get_power_model(X):
    N, c, h, w = X.shape
    assert c == 1 and h == w == 20
    X_windowed = X * hann_2d(h)
    X_fft = abs(fft2(X_windowed)) ** 2
    assert X_fft.shape == X.shape == X_windowed.shape
    assert np.all(np.isfinite(X_fft))
    return X_fft


def get_q_model(X, loc):
    assert X.ndim == 4
    x_flat = X.reshape(X.shape[0], -1)
    X_flat_q = quadratic_features(X, locality=(0, loc, loc))
    # let's check amount of variance contributed by X_flat and X_flat_q
    x_flat_all = np.concatenate((x_flat, X_flat_q), axis=1)

    return x_flat_all, x_flat.shape[1]


def get_q_model_pca_trans(x_flat_all, size_x_linear, max_total_dim):
    num_feature = x_flat_all.shape[1]
    pca_feature = min(num_feature - size_x_linear,
                      max_total_dim - size_x_linear,
                      x_flat_all.shape[0])
    assert size_x_linear == 400
    assert pca_feature > 1
    pca_obj = PCA(svd_solver='randomized', n_components=pca_feature,
                  random_state=0)
    pca_sep_input = x_flat_all[:, size_x_linear:]
    pca_obj.fit(pca_sep_input)

    def transformer(x):
        # print(x.shape, 'haha')
        assert x.ndim == 2 and x.shape[1] > size_x_linear
        pca_sep_input_this = x[:, size_x_linear:]
        # print(pca_sep_input.std(), 'haha pca_sep_input_this')
        x_to_use = pca_obj.transform(pca_sep_input_this)
        # print(x_to_use.std(), 'haha x_to_use')
        pca_sep_final_input = np.concatenate([x[:, :size_x_linear],
                                              x_to_use], axis=1)
        return pca_sep_final_input

    return transformer, pca_obj.explained_variance_ratio_.copy()


def original_data_check(X_original):
    assert isinstance(X_original, np.ndarray)
    assert X_original.ndim == 4
    assert X_original.shape[1:] == (1, 20, 20)
    assert X_original.shape[0] > 0


class GLMDataPreprocesser:
    def __init__(self):
        self.transformer = None
        # fraction of var in transformed data (for X_train_no_val_full).
        self.per_dim_var = None

    def get_transformer(self, X_train_no_val_full):
        # always use the 100% train (without validation) version to get transformation.
        raise NotImplementedError

    def transform(self, X: np.ndarray):
        original_data_check(X)
        assert self.transformer is not None, "run get_transformer first"
        data_return = self.transformer(X)
        # return self.transform_post(x_intermediate)
        assert data_return.ndim == 2 and data_return.shape[1] > 0 and data_return.shape[0] == X.shape[0]
        return data_return


class VanillaGLMPreprocessor(GLMDataPreprocesser):
    def __init__(self):
        super().__init__()

    def get_transformer(self, X_train_no_val_full):
        # simply return the remixing one.
        original_data_check(X_train_no_val_full)
        (self.transformer,
         self.per_dim_var) = remixing_transformer(X_train_no_val_full.reshape(X_train_no_val_full.shape[0], -1))


class FPGLMPreprocessor(GLMDataPreprocesser):
    def __init__(self):
        super().__init__()

    def get_transformer(self, X_train_no_val_full):
        # first, perform FP transformation.
        # and then return the remixing one.

        x_flat_all = get_power_model(X_train_no_val_full).reshape(X_train_no_val_full.shape[0], -1)
        mix_trans, explained_var_ratio = remixing_transformer(x_flat_all)

        def transformer(X):
            x_flat_all_this = get_power_model(X).reshape(X.shape[0], -1)
            return mix_trans(x_flat_all_this)

        self.transformer = transformer
        self.per_dim_var = explained_var_ratio


class GQMPreprocessor(GLMDataPreprocesser):
    # let's first try 1032 for debugging purpose.
    def __init__(self, locality, max_total_dim):
        super().__init__()
        self.locality = locality
        self.max_total_dim = max_total_dim

    def get_transformer(self, X_train_no_val_full):
        # first, perform FP transformation.
        # and then return the remixing one.

        x_flat_all, size_x_linear = get_q_model(X_train_no_val_full,
                                                self.locality)
        assert size_x_linear == 400
        transformer_q_pca, explaind_var_ratio_q = get_q_model_pca_trans(x_flat_all, size_x_linear,
                                                                        self.max_total_dim)
        mix_trans, _ = remixing_transformer(transformer_q_pca(x_flat_all))

        def transformer(X):
            # t1 = time.time()
            x_flat_all_this, _ = get_q_model(X, self.locality)
            # t2 = time.time()
            assert _ == size_x_linear
            combined = transformer_q_pca(x_flat_all_this)
            # t3 = time.time()
            final = mix_trans(combined)
            # t4 = time.time()
            # basically, t2-t1 takes most of the time,
            # and it doesn't scale with data size.
            # a very large overhead with constant time.
            # print(t2-t1, t3-t2, t4-t3)

            return final

        self.transformer = transformer
        self.per_dim_var = explaind_var_ratio_q


max_total_dim_debug = 1032  # old values
max_total_dim = None


def generate_transformer_dict(max_total_dim_this):
    transformer_dict = {
        'linear': (VanillaGLMPreprocessor, {}),
        'fpower': (FPGLMPreprocessor, {}),
        'gqm.2': (GQMPreprocessor, {'locality': 2, 'max_total_dim': max_total_dim_this}),
        'gqm.4': (GQMPreprocessor, {'locality': 4, 'max_total_dim': max_total_dim_this}),
        'gqm.8': (GQMPreprocessor, {'locality': 8, 'max_total_dim': max_total_dim_this}),
    }

    return transformer_dict
