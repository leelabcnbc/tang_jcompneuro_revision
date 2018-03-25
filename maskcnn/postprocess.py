import h5py
import os.path
from .training_aux_wrapper import file_and_key_to_save


# load back data.
def load_model_performance(dataset, image_subset, neuron_subset,
                           seed, arch_name, opt_name):
    model_subtype = arch_name + '+' + opt_name
    # then check whether exists or not.
    dir_to_save, file_name_base, key_to_save = file_and_key_to_save('cnn_population', model_subtype,
                                                                    dataset, image_subset, neuron_subset, seed)
    result = dict()
    with h5py.File(os.path.join(dir_to_save, file_name_base), 'r') as f_out:
        grp = f_out[key_to_save]
        # result['y_test_hat'] = grp['y_test_hat'][...]
        result['corr'] = grp['corr'][...]
        # result['corr_val'] = grp['corr_val'][...]
    return result
