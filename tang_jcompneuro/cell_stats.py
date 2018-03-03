"""this module has functions to compute some stats about cells, such as ccmax"""

import os.path
import h5py
import numpy as np
from strflab.stats import cc_max
from tang_jcompneuro import dir_dictionary
from tang_jcompneuro.stimulus_classification import decompose_subset
from tang_jcompneuro.io import load_neural_dataset, neural_dataset_dict

_global_dict = {
    'cell_ccmax': os.path.join(dir_dictionary['analyses'], 'cell_ccmax.hdf5'),
}


def compute_ccmax(neural_dataset_key, subset=None, num_trial=None):
    subset_proper = decompose_subset(subset)[0]

    # get num trial
    if num_trial is None:
        num_trial = neural_dataset_dict[neural_dataset_key]['trial_count']

    return _compute_ccmax_fetch(neural_dataset_key, subset_proper, num_trial)


def _compute_ccmax_fetch(neural_dataset_key, subset_proper, num_trial):
    key_to_fetch = '/'.join([neural_dataset_key, str(subset_proper), str(num_trial)])

    # check if it's there.
    with h5py.File(_global_dict['cell_ccmax']) as f:
        if key_to_fetch not in f:
            # write.
            # get response

            neural_response_this_subset = load_neural_dataset(neural_dataset_key, use_mean=False,
                                                              subset=subset_proper)
            assert 0 < num_trial <= neural_response_this_subset.shape[1]
            neural_response_this_subset = neural_response_this_subset[:, :num_trial]
            neural_response_this_subset_mean = load_neural_dataset(neural_dataset_key, use_mean=True,
                                                                   subset=subset_proper)[:, np.newaxis]
            # then fill in mean value
            # notice that how you do this will affect ccmax.
            # filling in mean is just one way to do it.
            all_data = np.where(np.isnan(neural_response_this_subset),
                                neural_response_this_subset_mean,
                                neural_response_this_subset)
            ccmax_this = cc_max(all_data.T)
            assert np.all(ccmax_this > 0)
            f.create_dataset(key_to_fetch, data=ccmax_this)
            f.flush()

        return f[key_to_fetch][...]
