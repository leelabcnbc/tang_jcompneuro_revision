"""similar to misc.py in the previous tang-paper-2017 repo"""
import os.path
from collections import OrderedDict
# from itertools import product

import numpy as np

from . import dir_dictionary

range_constructor = np.arange


def _shape_mapping_add_sublevels(prefix, result_dict, global_start_idx, group_step, group_num, start=1):
    for subgroup_idx, start_idx in enumerate(
            range(global_start_idx, global_start_idx + group_step * group_num, group_step), start=start):
        result_dict[prefix + str(subgroup_idx)] = range_constructor(start_idx, start_idx + group_step)
    return result_dict


def shape_9500_mapping(get_sublevels=False):
    result = OrderedDict()

    # edge
    result['E'] = range_constructor(80)
    # full lines
    if get_sublevels:
        result = _shape_mapping_add_sublevels('FL', result, 80, 40, 4)
    else:
        # total
        result['FL'] = range_constructor(80, 240)
    # dense lines
    if get_sublevels:
        result['DL1'] = range_constructor(240, 280)
        result['DL2'] = range_constructor(280, 320)
    else:
        result['DL'] = range_constructor(240, 320)

    # center bars
    if get_sublevels:
        result = _shape_mapping_add_sublevels('CB', result, 320, 40, 6)
    else:
        result['CB'] = range_constructor(320, 560)

    # side bars
    if get_sublevels:
        result['SB1'] = range_constructor(560, 640)
        result['SB2'] = range_constructor(640, 720)
        # longer, and so many.
        result['SB3'] = range_constructor(720, 1600)
    else:
        result['SB'] = range_constructor(560, 1600)

    # solid corners, acute
    if get_sublevels:
        result = _shape_mapping_add_sublevels('SCA', result, 1600, 40, 50)
    else:
        result['SCA'] = range_constructor(1600, 3600)

    # solid corners, obtuse
    if get_sublevels:
        result = _shape_mapping_add_sublevels('SCO', result, 3600, 80, 4)
    else:
        result['SCO'] = range_constructor(3600, 3920)

    # outline corners, acute
    if get_sublevels:
        result = _shape_mapping_add_sublevels('OCA', result, 3920, 40, 50)
    else:
        result['OCA'] = range_constructor(3920, 5920)

    # outline corners, obtuse
    if get_sublevels:
        result = _shape_mapping_add_sublevels('OCO', result, 5920, 80, 4)
    else:
        result['OCO'] = range_constructor(5920, 6240)

    # ray
    if get_sublevels:
        result = _shape_mapping_add_sublevels('RAY', result, 6240, 40, 3)
        result = _shape_mapping_add_sublevels('RAY', result, 6360, 10, 1, start=4)
        result = _shape_mapping_add_sublevels('RAY', result, 6370, 40, 1, start=5)
        result = _shape_mapping_add_sublevels('RAY', result, 6410, 10, 1, start=6)
        result = _shape_mapping_add_sublevels('RAY', result, 6420, 40, 1, start=7)
        result = _shape_mapping_add_sublevels('RAY', result, 6460, 10, 1, start=8)
    else:
        result['RAY'] = range_constructor(6240, 6470)

    # Y
    result['Y'] = range_constructor(6470, 6550)

    # fan
    result['F'] = range_constructor(6550, 6630)

    # cross
    if get_sublevels:
        result = _shape_mapping_add_sublevels('CX', result, 6630, 40, 8)
    else:
        result['CX'] = range_constructor(6630, 6950)

    # spike short, short
    if get_sublevels:
        result = _shape_mapping_add_sublevels('SSS', result, 6950, 40, 4)
    else:
        result['SSS'] = range_constructor(6950, 7110)

    # spike long, short
    if get_sublevels:
        result = _shape_mapping_add_sublevels('SLS', result, 7110, 40, 4)
    else:
        result['SLS'] = range_constructor(7110, 7270)

    # spike short
    if get_sublevels:
        result = _shape_mapping_add_sublevels('SS', result, 7270, 40, 4)
    else:
        result['SS'] = range_constructor(7270, 7430)

    # spike long
    if get_sublevels:
        result = _shape_mapping_add_sublevels('SL', result, 7430, 40, 4)
    else:
        result['SL'] = range_constructor(7430, 7590)

    # grid
    if get_sublevels:
        result = _shape_mapping_add_sublevels('G', result, 7590, 20, 2)
    else:
        result['G'] = range_constructor(7590, 7630)

    # balls
    if get_sublevels:
        result = _shape_mapping_add_sublevels('B', result, 7630, 80, 3)
    else:
        result['B'] = range_constructor(7630, 7870)

    # rings
    if get_sublevels:
        result = _shape_mapping_add_sublevels('R', result, 7870, 80, 3)
    else:
        result['R'] = range_constructor(7870, 8110)

    # curves
    if get_sublevels:
        result = _shape_mapping_add_sublevels('CV', result, 8110, 80, 6)
    else:
        result['CV'] = range_constructor(8110, 8590)

    # concentric ring
    if get_sublevels:
        result = _shape_mapping_add_sublevels('CR', result, 8590, 5, 3)
    else:
        result['CR'] = range_constructor(8590, 8605)

    # edge with circles
    if get_sublevels:
        result = _shape_mapping_add_sublevels('EC', result, 8605, 40, 8)
    else:
        result['EC'] = range_constructor(8605, 8925)

    # ray center
    if get_sublevels:
        result = _shape_mapping_add_sublevels('RC', result, 8925, 40, 8)
    else:
        result['RC'] = range_constructor(8925, 9245)

    # ray no center
    if get_sublevels:
        result = _shape_mapping_add_sublevels('RNC', result, 9245, 40, 6)
    else:
        result['RNC'] = range_constructor(9245, 9485)

    # stars
    if get_sublevels:
        result = _shape_mapping_add_sublevels('S', result, 9485, 5, 3)
    else:
        result['S'] = range_constructor(9485, 9500)

    # make sure it's correct.
    result_all = np.sort(np.concatenate(list(result.values())))
    assert np.array_equal(result_all, np.arange(9500))

    return result


# based on his v3 classification, which should be the same as
# /private_data/shape_params_data/Stimuli_Name.xlsx
# I debugged this against v3 dict in
# thesis_plots/v1_fitting/pattern_stimulus.ipynb
shape_9500_more_broad_classification_dict_tang = OrderedDict()
shape_9500_more_broad_classification_dict_tang['bar'] = ('E', 'FL', 'CB', 'SB', 'DL')
shape_9500_more_broad_classification_dict_tang['curvature'] = ('B', 'R', 'CR', 'CV')
shape_9500_more_broad_classification_dict_tang['corner'] = ('SCA', 'SCO', 'OCA', 'OCO', 'F')
shape_9500_more_broad_classification_dict_tang['cross'] = ('RAY', 'CX', 'Y', 'SSS', 'SLS', 'SS', 'SL', 'G')
shape_9500_more_broad_classification_dict_tang['composition'] = ('EC', 'RC', 'RNC', 'S')
_all_cate = sum(shape_9500_more_broad_classification_dict_tang.values(), ())
assert len(set(_all_cate)) == len(_all_cate)
assert set(shape_9500_mapping(get_sublevels=False).keys()) == set(_all_cate)


def _mapping_to_str_9500(mapping_this):
    label_str_this = np.empty((9500,), dtype=np.object_)
    for key, range_this in mapping_this.items():
        label_str_this[range_this] = key
    return label_str_this


def _attach_sample_weight(int_label):
    assert int_label.shape == (int_label.size,)
    assert (int_label >= 0).all()
    bin_count = np.bincount(int_label)
    assert np.all(bin_count > 0)
    weight_label = np.empty((int_label.size,), dtype=np.float64)
    for label in np.arange(bin_count.size):
        weight_label[int_label == label] = 1 / bin_count[label]

    return weight_label


def _attach_weight_label_wrapper(label_dict):
    for level in ('low', 'middle', 'high'):
        label_dict[(level, 'weight')] = _attach_sample_weight(label_dict[(level, 'int')])

    return label_dict


def _load_9500_labels():
    # give shape_9500 label in all three levels, providing both int version and string version.

    # first handle low one.
    label_str_low = _mapping_to_str_9500(shape_9500_mapping(get_sublevels=True))
    label_str_mid = _mapping_to_str_9500(shape_9500_mapping(get_sublevels=False))
    label_str_high = np.empty((9500,), dtype=np.object_)

    high_order_classification_dict = shape_9500_more_broad_classification_dict_tang

    for key, subkeys in high_order_classification_dict.items():
        label_str_high[np.in1d(label_str_mid, subkeys)] = key

    _, label_int_low = np.unique(label_str_low, return_inverse=True)
    _, label_int_mid = np.unique(label_str_mid, return_inverse=True)
    _, label_int_high = np.unique(label_str_high, return_inverse=True)

    return _attach_weight_label_wrapper({
        ('low', 'int'): label_int_low,
        ('low', 'str'): label_str_low,
        ('middle', 'int'): label_int_mid,
        ('middle', 'str'): label_str_mid,
        ('high', 'int'): label_int_high,
        ('high', 'str'): label_str_high,
    })


def _load_shape_4605_correspondence():
    a = np.loadtxt(os.path.join(dir_dictionary['shape_params_data'], 'mapping_vec.csv'), dtype=np.int64)
    assert a.shape == (4605,)
    return a


def _load_4605_labels():
    mapping = _load_shape_4605_correspondence()
    labels_old = _load_9500_labels()
    # int label has to be regenerated, as there's some sublevel missing.
    labels_new = {}
    for key in ('low', 'middle', 'high'):
        labels_new[(key, 'str')] = labels_old[(key, 'str')][mapping]
        _, labels_new[(key, 'int')] = np.unique(labels_new[(key, 'str')], return_inverse=True)
    return _attach_weight_label_wrapper(labels_new)


# this will be used by other files.
stimulus_label_dict_tang = {
    'Shape_9500': _load_9500_labels(),
    'Shape_4605': _load_4605_labels(),
}


# # generate all subsets needed
# def get_reference_subset_list(*, percentage_list=(None, 25, 50, 75)):
#     base_subsets = ('OT', 'nonOT', 'all')
#     list_all = []
#     for base_subset, per_this in product(base_subsets, percentage_list):
#         # TODO if per_this is float, maybe need some work to convert it to string.
#         suffix = '' if per_this is None else f'+{per_this}_0'  # just do one shuffling.
#         list_all.append(base_subset + suffix)
#     return list_all


def decompose_subset(subset):
    if subset is None:
        return None, None
    else:
        assert isinstance(subset, str)
        subset_params_loc = subset.find('+')
        if subset_params_loc != -1:
            raise RuntimeError('should not be here in new repo!')
            # subset_proper, subset_param = subset[:subset_params_loc], subset[subset_params_loc + 1:]
        else:
            subset_proper, subset_param = subset, None
        return subset_proper, subset_param


num_ot_dict = {
    'Shape_9500': 1600,
    'Shape_4605': 800,
}


#
def get_subset_slice(dataset, subset):
    """when subset is complex (with `+` in it), the return value won't be a slice."""
    assert dataset in {'Shape_9500', 'Shape_4605'}

    if subset is None or subset == 'all':
        return slice(None)


    num_ot = num_ot_dict[dataset]

    # num_total = {
    #     'Shape_9500': 9500,
    #     'Shape_4605': 4605,
    # }[dataset]

    # ok. I allow subset to have '+' in it. if not, then it's legacy subset,
    # without partitioning and all that stuff.
    # subset_params_loc = subset.find('+')
    # if subset_params_loc != -1:
    #     subset_proper, subset_param = subset[:subset_params_loc], subset[subset_params_loc + 1:]
    # else:
    #     subset_proper, subset_param = subset, None
    subset_proper, subset_param = decompose_subset(subset)
    if subset_param is None:
        if subset_proper == 'OT':
            return slice(None, num_ot)
        elif subset_proper == 'nonOT':
            return slice(num_ot, None)
        # didn't include nonOT_nonCN, as that's only for debugging purpose.
        elif subset_proper == 'all':
            return slice(None)
        else:
            raise ValueError('wrong subset')
    else:
        raise RuntimeError('should not be here in new repo!')
