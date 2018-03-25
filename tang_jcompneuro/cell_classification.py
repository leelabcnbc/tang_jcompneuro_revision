"""helper functions for cell classification
simplified from https://github.com/leelabcnbc/tang-paper-2017/blob/master/tang_2017/cell_classification.py
"""
import os.path
from collections import defaultdict, OrderedDict

import h5py
import numpy as np

from tang_jcompneuro import dir_dictionary
from tang_jcompneuro.io import neural_dataset_dict, load_neural_dataset

_global_dict = {
    'cell_classification': os.path.join(dir_dictionary['analyses'], 'cell_classification.hdf5'),
}


def _load_cell_classification(grp: h5py.Group):
    assert isinstance(grp, h5py.Group)
    dict_all = dict()
    for x in grp['label']:
        dict_all[x] = grp[f'label/{x}'][...]

    return dict_all


def _save_cell_classification(grp: h5py.Group, dict_all):
    for x, y in dict_all.items():
        grp.create_dataset(f'label/{x}', data=y)
        grp.file.flush()


def _compute_cell_classification_inner(neural_dataset_key, version, additional_params):
    assert additional_params is None
    image_dataset = neural_dataset_dict[neural_dataset_key]['image_dataset_key']
    if version == 3:
        data_mean = load_neural_dataset(neural_dataset_key, use_mean=True)
        labels_raw, stats_raw, passed = get_label_and_stat(criterion_v3(image_dataset), data_mean)
    else:
        raise ValueError('invalid version')
        # TODO: save stats and passed in the future if needed.
    return labels_raw


# you should only use this SINGLE function.
def compute_cell_classification(neural_dataset_key, version, additional_params=None, readonly=False):
    assert additional_params is None  # for future extension.
    key_to_use = '/'.join([neural_dataset_key, str(version), str(additional_params)])
    with h5py.File(_global_dict['cell_classification'], 'r' if readonly else 'a') as f:
        if key_to_use not in f:
            # compute and save
            dict_all = _compute_cell_classification_inner(neural_dataset_key, version, additional_params)
            f.create_group(key_to_use)
            _save_cell_classification(f[key_to_use], dict_all)

        return _load_cell_classification(f[key_to_use])


subset_decompose_dict = OrderedDict()
subset_decompose_dict['OT'] = ('SS', 'EB')
subset_decompose_dict['HO'] = ('CV', 'CN', 'CRS', 'Other', 'Multi')
name_mapping_dict = {
    'CV': 'curvature',
    'CN': 'corner',
    'CRS': 'cross',
    'Other': 'composition',
    'Multi': 'mixed',
    'SS': 'classical',
    'EB': 'end-stopping',
}


def _check_parition(master_label, label_counter):
    assert master_label.shape == label_counter.shape
    assert master_label.dtype == np.bool_
    assert label_counter.dtype == np.int64
    # exactly cover same thing
    assert np.array_equal(master_label, label_counter.astype(np.bool_))
    # each one appears exactly once
    assert np.array_equal(np.unique(label_counter), np.array([0, 1]))


def _get_ready_to_use_classification_coarse(readonly):
    class_dict_final = dict()
    for dataset in ('MkA_Shape', 'MkE2_Shape'):
        classification_dict_all = compute_cell_classification(dataset, 3, readonly=readonly)
        class_dict_this = dict()
        # print(classification_dict_all.keys())
        label_counter_sub = None
        for subset in subset_decompose_dict.keys():
            label_this = classification_dict_all[subset]

            if label_counter_sub is None:
                label_counter_sub = np.zeros_like(label_this, dtype=np.int64)
            assert label_counter_sub.shape == label_this.shape
            label_counter_sub += label_this.astype(np.int64)
            class_dict_this[subset] = label_this
        _check_parition(label_counter_sub.astype(np.bool_), label_counter_sub)
        class_dict_final[dataset] = class_dict_this

    return class_dict_final


def _get_ready_to_use_classification_detailed(readonly):
    # https://github.com/leelabcnbc/tang_jcompneuro/blob/d8e3bb719df89765723a25607baf0d4189162cd6/thesis_plots/v1_fitting/comparison_among_cnn_glm_vgg_decomposed_by_fine_subsets.ipynb
    class_dict_final = dict()
    for dataset in ('MkA_Shape', 'MkE2_Shape'):
        classification_dict_all = compute_cell_classification(dataset, 3, readonly=readonly)
        class_dict_this = dict()
        # print(classification_dict_all.keys())
        for subset, detailed in subset_decompose_dict.items():
            # print(subset)
            class_dict_this_this_subset = OrderedDict()
            label_master = classification_dict_all[subset]

            # to count occurence of each type of label.
            label_counter_sub = np.zeros_like(label_master, dtype=np.int64)

            for label_fine in detailed:
                label_this = classification_dict_all[label_fine]
                assert label_counter_sub.shape == label_this.shape
                duplicate_neuron_this = np.flatnonzero(np.logical_and(label_counter_sub > 0, label_this))
                if duplicate_neuron_this.size > 0:
                    # print('duplicate', duplicate_neuron_this)  # match
                    # https://github.com/leelabcnbc/tang-paper-2017/blob/master/population_analysis/noise_correlation_analysis.ipynb
                    label_this[duplicate_neuron_this] = False  # remove them
                label_counter_sub += label_this.astype(np.int64)
                # print(label_this_TO_USE.shape)
                class_dict_this_this_subset[label_fine] = label_this

            _check_parition(label_master, label_counter_sub)

            class_dict_this[subset] = class_dict_this_this_subset
        class_dict_final[dataset] = class_dict_this
    return class_dict_final


def get_ready_to_use_classification(coarse=True, readonly=False):
    if coarse:
        return _get_ready_to_use_classification_coarse(readonly)
    else:
        return _get_ready_to_use_classification_detailed(readonly)


class CellTypeCriterionSingle:
    def __init__(self,
                 name,
                 criterion_type='self-contained',
                 statistic='ratio_of_max',
                 statistic_param=None,
                 pos_neg_index=(None, None),
                 complement_of=None,  # this class is complement to the union of this value (under subset_of)
                 subset_of=None,  # this class is subset the intersection of this value
                 not_subset_of=None,  # this is similar to complement, except that it's used all the time. not used yet.
                 priority=0,  # this is used to sort all cell criterion, so that we can compute the complement ones,
                 strict=True,
                 ):

        # assign all common ones
        self.name = name
        self.priority = priority
        self.criterion_type = criterion_type
        self.subset_of = () if subset_of is None else subset_of
        self.not_subset_of = () if not_subset_of is None else not_subset_of
        self.strict = strict

        if criterion_type == 'self-contained':
            self.statistic = statistic
            self.statistic_param = statistic_param
            if self.statistic == 'ratio_of_max':
                assert np.isscalar(self.statistic_param) and self.statistic_param > 0
            else:
                raise ValueError('unsupported statistic')
            self.pos_index, self.neg_index = pos_neg_index
            assert isinstance(self.pos_index, np.ndarray) and isinstance(self.neg_index, np.ndarray)
            # check that they have no overlap
            assert np.union1d(self.pos_index, self.neg_index).size == self.pos_index.size + self.neg_index.size
            assert np.intersect1d(self.pos_index, self.neg_index).size == 0, 'no intersection between pos and neg!'
        elif criterion_type == 'complement':
            self.complement_of = complement_of
            assert self.complement_of is not None
        else:
            raise ValueError('unsupported criterion type!')

    def classify(self, mean_resp=None, class_dict=None):
        assert class_dict is not None
        if self.criterion_type == 'self-contained':
            if self.statistic == 'ratio_of_max':
                max_pos = mean_resp[self.pos_index].max()
                max_neg = mean_resp[self.neg_index].max()
                # hack for CNN
                if max_pos > 0 and max_neg > 0:
                    stat = max_pos / max_neg
                    in_this_class = stat > self.statistic_param
                else:
                    assert not self.strict, 'you can only enter this branch in non strict mode'
                    stat = np.nan
                    in_this_class = False
            else:
                raise ValueError('unsupported statistic')
        elif self.criterion_type == 'complement':
            stat = np.nan
            # if this is empty, by Python doc, it's True.
            in_this_class = all(not class_dict[c] for c in self.complement_of)
        else:
            raise ValueError('unsupported criterion type!')
        assert in_this_class in {True, False}
        # finally, check subset
        in_this_class = in_this_class and all(class_dict[c] for c in self.subset_of) and all(
            not class_dict[c] for c in self.not_subset_of)
        # return boolean for whether in this class, as well as the raw statistic.
        assert np.isfinite(stat) or np.isnan(stat)
        return in_this_class, np.float64(stat)


def create_range(range_list):
    return np.concatenate([np.arange(*x) for x in range_list])


def class_pos_neg_v3(dataset):
    result = defaultdict(lambda: (None, None))
    if dataset == 'Shape_9500':
        result['HO'] = create_range([(1600, 9500)]), create_range([(1600,)])
        result['EB'] = create_range([(320, 1600)]), create_range([(320,)])
        result['CN'] = create_range([(1600, 6240), (6550, 6630)]), create_range([(1600,), (7630, 8605)])
        result['CV'] = create_range([(7630, 8605)]), create_range([(7630,)])
        result['CRS'] = create_range([(6240, 6550), (6630, 7630)]), create_range([(6240,), (6550, 6630), (7630, 9500)])
        result['Other'] = create_range([(8605, 9500)]), create_range([(8605,)])
    elif dataset == 'Shape_4605':
        result['HO'] = create_range([(800, 4605)]), create_range([(800,)])
        result['EB'] = create_range([(160, 800)]), create_range([(160,)])
        result['CN'] = create_range([(800, 2960), (3115, 3155)]), create_range([(800,), (3655, 4150)])
        result['CV'] = create_range([(3655, 4150)]), create_range([(3655,)])
        result['CRS'] = create_range([(2960, 3115), (3155, 3655)]), create_range([(2960,), (3115, 3155), (3655, 4605)])
        # fix Tang's code's bug
        result['Other'] = create_range([(4150, 4605)]), create_range([(4150,)])
    else:
        raise ValueError('invalid dataset!')

    return result


def class_subset():
    result = defaultdict(lambda: None)
    result['EB'] = ('OT',)
    result['SS'] = ('OT',)
    result['Multi'] = ('HO',)
    return result


def class_complement():
    result = defaultdict(lambda: None)
    result['OT'] = ('HO',)
    result['SS'] = ('EB',)
    result['Multi'] = ('CN', 'CV', 'CRS', 'Other')
    return result


def class_priority():
    # lower means higher priority
    result = defaultdict(lambda: 0)
    result['OT'] = 1
    result['EB'] = 2
    result['SS'] = 3
    result['Multi'] = 4
    return result


def class_criterion_type():
    result = defaultdict(lambda: 'self-contained')
    result['OT'] = 'complement'
    result['SS'] = 'complement'
    result['Multi'] = 'complement'
    return result


def criterion_v3(dataset, permissible_threshold_raw=0.5, ratio=2, statistic='ratio_of_max',
                 strict=True):
    """this encodes the cell classification criterion used in PtnFig4Pie20170608"""
    result = dict()
    result['permissible_threshold_raw'] = permissible_threshold_raw
    # this is the most comprehensive one.

    criterion_this = class_pos_neg_v3(dataset)

    # either only HO or a lot.
    keys_this_criterion = criterion_this.keys()
    if keys_this_criterion == {'HO'}:
        class_to_consider = ('HO', 'OT')  # this is actually enough for us.
    elif keys_this_criterion == {'HO', 'EB', 'CN', 'CV', 'CRS', 'Other'}:
        class_to_consider = ('HO', 'EB', 'CN', 'CV', 'CRS', 'Other', 'SS', 'Multi', 'OT')
        # a key for what they mean.
        # check
        # https://github.com/leelabcnbc/tang_jcompneuro/blob/master/thesis_plots/
        #         v1_fitting/comparison_among_cnn_glm_vgg_decomposed_by_fine_subsets.ipynb
        # HO is higher-order, and OT is orientation.
        # name_mapping_dict = {
        #     'CV': 'curvature',
        #     'CN': 'corner',
        #     'CRS': 'cross',
        #     'Other': 'composition',
        #     'Multi': 'mixed',
        #     'SS': 'classical',
        #     'EB': 'end-stopping',
        # }
    else:
        raise ValueError('invalid set of keys')

    result['criterion_list'] = [
        CellTypeCriterionSingle(c,
                                criterion_type=class_criterion_type()[c],
                                statistic=statistic,
                                statistic_param=ratio,
                                pos_neg_index=criterion_this[c],
                                complement_of=class_complement()[c],
                                subset_of=class_subset()[c],
                                priority=class_priority()[c],
                                strict=strict) for c in class_to_consider
    ]
    result['criterion_list'] = sorted(result['criterion_list'], key=lambda x: x.priority)

    return result


# ok. a big for loop to go over all cells
def get_label_and_stat(criterion, data_mean, verbose=True):
    result_label = []
    result_stat = []

    assert data_mean is not None
    data_mean = data_mean.T
    num_neuron = data_mean.shape[0]
    assert num_neuron > 0

    passed_array = []

    for n_index in range(num_neuron):
        if ((n_index + 1) % 200) == 0 and verbose:
            print(n_index)
        mean_this = data_mean[n_index]
        assert mean_this.ndim == 1
        mean_this_max = mean_this.max()
        # all_this = data_all[n_index].T if data_all is not None else None
        passed = mean_this_max > criterion['permissible_threshold_raw']
        passed_array.append(passed)
        result_label_this = dict()
        result_stat_this = dict()
        for criterion_single in criterion['criterion_list']:
            if passed:
                (result_label_this[criterion_single.name],
                 result_stat_this[criterion_single.name]) = criterion_single.classify(mean_this,
                                                                                      result_label_this)
            else:
                result_label_this[criterion_single.name] = False
                result_stat_this[criterion_single.name] = np.nan
        result_label.append(result_label_this)
        result_stat.append(result_stat_this)
    passed_array = np.asarray(passed_array)

    # convert result label and result stat into dict format.
    result_label_new = dict()
    result_stat_new = dict()
    for class_name in result_label[0]:
        result_label_new[class_name] = np.asarray([x[class_name] for x in result_label])
        result_stat_new[class_name] = np.asarray([x[class_name] for x in result_stat])
    # check their dtypes.
    assert result_label_new.keys() == result_stat_new.keys()
    for class_name in result_label_new:
        print(class_name, result_label_new[class_name].dtype, result_stat_new[class_name].dtype)
        assert result_label_new[class_name].shape == result_stat_new[class_name].shape == (num_neuron,)
        assert result_label_new[class_name].dtype == np.bool_ and result_stat_new[class_name].dtype == np.float64
    assert passed_array.dtype == np.bool_ and passed_array.shape == (num_neuron,)
    return result_label_new, result_stat_new, passed_array
