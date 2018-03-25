"""rewrite of
https://github.com/leelabcnbc/tang-paper-2017/blob/9fb284269aec2623a300328fef2f075831e3d44d/feature_extraction/feature_extraction.ipynb"""

from tang_jcompneuro.io import load_image_dataset
from tang_jcompneuro.cnn_pretrained import (get_pretrained_network, process_one_case_wrapper)
import numpy as np
from torch.backends import cudnn
import os.path
from tang_jcompneuro import dir_dictionary

# this can save memory error. See <https://github.com/pytorch/pytorch/issues/1230>
cudnn.benchmark = False
cudnn.enabled = True

networks_to_try = ('vgg16', 'vgg16_bn',
                   'vgg19', 'vgg19_bn')

settings_dict = {
    'legacy': {'scale': 2 / 3, 'ec_size': 22},
    # this one will be used
}

bg_dict_per_dataset = {
    'Shape_9500': np.array([1.0, 1.0, 1.0]),
    'Shape_4605': np.array([1.0, 1.0, 1.0]),
}

dataset_dict = {
    'Shape_9500': lambda: load_image_dataset('Shape_9500', patch_size=160, normalize_cnn_format=False),
    # so that they will not be bigger than CNN input.
    'Shape_4605': lambda: load_image_dataset('Shape_4605', patch_size=160, normalize_cnn_format=False),
}

file_to_save_input = os.path.join(dir_dictionary['datasets'], 'cnn_feature_extraction_input.hdf5')
file_to_save_feature = os.path.join(dir_dictionary['features'], 'cnn_feature_extraction.hdf5')


def do_all():
    for net_name in networks_to_try:
        net = get_pretrained_network(net_name)
        net.cuda()
        for dataset, dataset_fn in dataset_dict.items():
            dataset_np = dataset_fn()
            print(dataset, dataset_np.shape)
            for setting_name, setting in settings_dict.items():
                process_one_case_wrapper(net_name, net, dataset_np, f'{dataset}/{net_name}/{setting_name}', setting,
                                         bg_dict_per_dataset[dataset], 100,
                                         file_to_save_input, file_to_save_feature)
        # may save some memory.
        del net


if __name__ == '__main__':
    do_all()
