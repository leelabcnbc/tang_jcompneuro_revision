import os.path

import h5py
import numpy as np
from scipy.io import loadmat

from tang_jcompneuro import dir_dictionary

tang_data_root = dir_dictionary['tang_data_root']  # change this if needed


def read_mat_list(data_root, key_list, file_list):
    raw_data_dict = {key: loadmat(os.path.join(data_root, file_name)) for key, file_name in zip(key_list, file_list)}
    return raw_data_dict


def read_hdf5_list(data_root, key_list, dataset_list, file_list):
    raw_data_dict = dict()
    for key, dataset_name, file_name in zip(key_list, dataset_list, file_list):
        with h5py.File(os.path.join(data_root, file_name), 'r') as f_inner:
            dataset = f_inner[dataset_name][:]
            raw_data_dict[key] = dataset
    return raw_data_dict


def write_hdf5_group(grp_handle, notes, name_list, dataset_list):
    grp_handle.attrs['notes'] = np.string_(notes)
    for name, data in zip(name_list, dataset_list):
        grp_handle.create_dataset(name, data=data, compression='gzip', compression_opts=4)


def fill_nan_for_corrected_all(response_all, response_count):
    # set all non-response to nan.
    count_it = np.nditer(response_count, flags=['multi_index'])
    error_msg_count = 0
    while not count_it.finished:
        # check count is correct.
        idx_stimulus, idx_neuron = count_it.multi_index
        count_this = count_it[0]
        assert count_this > 0
        if not np.all(response_all[idx_stimulus, count_this:, idx_neuron] == 0):
            # find problematic place.
            raise AssertionError('problematic idx neuron {}, stimulus {}, count {}'.format(idx_neuron,
                                                                                           idx_stimulus, count_this))
        if not np.all(response_all[idx_stimulus, :count_this, idx_neuron] != 0):
            if error_msg_count < 100:  # otherwise seems that it will choke jupyter...
                print('suspicious zero resp. idx neuron {}, stimulus {}, count {}'.format(idx_neuron, idx_stimulus,
                                                                                          count_this))
            error_msg_count += 1
        # assign nan.
        response_all[idx_stimulus, count_this:, idx_neuron] = np.nan
        count_it.iternext()


def fill_nan_for_raw_all(response_raw_f0s, response_raw_fs, response_count, strict_check=True):
    assert response_raw_f0s.shape == response_raw_fs.shape
    assert np.all(np.isfinite(response_raw_f0s))
    assert np.all(np.isfinite(response_raw_fs))
    # response_all = np.empty_like(response_raw_f0s, dtype=np.float64)
    # set all non-response to nan.
    count_it = np.nditer(response_count, flags=['multi_index'])
    while not count_it.finished:
        # check count is correct.
        idx_stimulus, idx_neuron = count_it.multi_index
        count_this = count_it[0]
        assert count_this > 0
        select_slice = (idx_stimulus, slice(None, count_this), idx_neuron)
        complement_slice = (idx_stimulus, slice(count_this, None), idx_neuron)

        f0s_this = response_raw_f0s[select_slice]
        fs_this = response_raw_fs[select_slice]
        f0s_this_c = response_raw_f0s[complement_slice]
        fs_this_c = response_raw_fs[complement_slice]

        if strict_check:
            assert np.all(f0s_this != 0) and np.all(fs_this != 0)
        assert np.all(f0s_this_c == 0) and np.all(fs_this_c == 0)
        response_raw_f0s[complement_slice] = np.nan
        response_raw_fs[complement_slice] = np.nan
        count_it.iternext()

    mask = np.isnan(response_raw_f0s)
    response_all = (response_raw_fs - response_raw_f0s) / response_raw_f0s
    response_all[mask] = np.nan
    assert np.all(np.isfinite(response_all[~mask]))
    return response_all


def write_monkey_corrected_data(grp_handle_, tang_data_root_this, rawDataFile, countEvenFile, countOddFile,
                                rawDataFileField=None,
                                countEvenFileField=None,
                                countOddFileField=None, hack_data_func=None,
                                notes_additional=''):
    if rawDataFileField is None and countEvenFileField is None and countOddFileField is None:
        rawDataFileField = os.path.splitext(rawDataFile)[0]
        countEvenFileField = os.path.splitext(countEvenFile)[0]
        countOddFileField = os.path.splitext(countOddFile)[0]

    key_list = ('rawData', 'countEven', 'countOdd')
    file_list = (rawDataFile, countEvenFile, countOddFile)
    raw_data_dict = read_mat_list(tang_data_root_this, key_list, file_list)
    Rsp0412PicS = raw_data_dict['rawData'][rawDataFileField]  # (1225x6x2250)
    NumO0412Pic = raw_data_dict['countOdd'][countOddFileField]  # (1225x2250)
    NumE0412Pic = raw_data_dict['countEven'][countEvenFileField]  # (1225x2250)
    assert NumO0412Pic.shape == NumE0412Pic.shape

    # (num stimulus x num neuron)
    response_count = NumE0412Pic.astype(np.uint16).T + NumO0412Pic.astype(np.uint16).T
    # (num stimulus x max trial x num neuron)
    response_all = Rsp0412PicS.T.astype(np.float64)  # always copy

    if hack_data_func is not None:
        print('hack this response matrix!')
        hack_data_func(response_all, response_count)  # in place change

    num_stimulus, max_trial, num_neuron = response_all.shape
    assert NumO0412Pic.shape == NumE0412Pic.shape == (num_neuron, num_stimulus)
    assert response_count.shape == (num_stimulus, num_neuron)
    notes = 'originally from {}, {} neurons, {} stimulus, {} max trial, in fact {} to {}'.format(tang_data_root_this,
                                                                                                 num_neuron,
                                                                                                 num_stimulus,
                                                                                                 max_trial,
                                                                                                 response_count.min(),
                                                                                                 response_count.max())
    notes += ' rawDataFile {} (field {}), countEvenFile {} (field {}), countOddFile {} (field {})'.format(rawDataFile,
                                                                                                          rawDataFileField,
                                                                                                          countEvenFile,
                                                                                                          countEvenFileField,
                                                                                                          countOddFile,
                                                                                                          countOddFileField)
    notes += notes_additional
    fill_nan_for_corrected_all(response_all, response_count)
    notes += '; percentage of zero in response_all: {}'.format(np.mean(response_all[np.isfinite(response_all)] == 0))
    print(notes)
    response_mean = np.nanmean(response_all, axis=1)
    assert np.all(np.isfinite(response_mean))
    write_hdf5_group(grp_handle_, notes, ['all', 'mean', 'count'], [response_all, response_mean, response_count])


def write_monkey_all_data(grp_handle_, tang_data_root_this, rawDataFile,
                          rawDataFileField=None, notes_additional=''):
    if rawDataFileField is None:
        rawDataFileField = os.path.splitext(rawDataFile)[0]

    key_list = ('rawData',)
    file_list = (rawDataFile,)
    raw_data_dict = read_mat_list(tang_data_root_this, key_list, file_list)
    Rsp0412PicS = raw_data_dict['rawData'][rawDataFileField]  # (1225x6x2250)

    response_all = Rsp0412PicS.T.astype(np.float64)
    num_stimulus, max_trial, num_neuron = response_all.shape
    assert np.isfinite(response_all).all()
    response_mean = np.nanmean(response_all, axis=1)
    assert response_mean.shape == (num_stimulus, num_neuron)
    notes = 'originally from {}, {} neurons, {} stimulus, {} max trial'.format(tang_data_root_this,
                                                                               num_neuron,
                                                                               num_stimulus,
                                                                               max_trial)
    notes += '; percentage of zero in response_all: {}'.format(np.mean(response_all == 0))
    notes += notes_additional
    print(notes)

    assert np.all(np.isfinite(response_mean))
    write_hdf5_group(grp_handle_, notes, ['all', 'mean'], [response_all, response_mean])


def write_monkey_raw_data(grp_handle_, tang_data_root_this, f0sFile, fsFile,
                          f0sFileField=None, fsFileField=None, hdf5=True, notes_additional=''):
    key_list = ('F0S', 'FS')
    if f0sFileField is None and fsFileField is None:
        f0sFileField = os.path.splitext(f0sFile)[0]
        fsFileField = os.path.splitext(fsFile)[0]
    assert f0sFileField is not None and fsFileField is not None

    file_list = (f0sFile, fsFile)
    dataset_list = (f0sFileField, fsFileField)

    if hdf5:
        raw_data_dict = read_hdf5_list(tang_data_root_this, key_list, dataset_list, file_list)
    else:
        raw_data_dict_ = read_mat_list(tang_data_root_this, key_list, file_list)
        raw_data_dict = dict()
        for key, dataset_name in zip(key_list, dataset_list):
            raw_data_dict[key] = raw_data_dict_[key][dataset_name]

    PicRespF0S = raw_data_dict['F0S']
    PicRespFS = raw_data_dict['FS']  # 2250 x 1225 x 4

    if hdf5:
        transpose_arg = (0, 2, 1)
    else:
        transpose_arg = (1, 2, 0)

    response_raw_f0s = np.transpose(PicRespF0S, transpose_arg).astype(np.float64)
    response_raw_fs = np.transpose(PicRespFS, transpose_arg).astype(np.float64)

    mask1 = response_raw_f0s != 0
    mask2 = response_raw_fs != 0

    mask = np.logical_and(mask1, mask2)
    strict_check = True
    if not np.array_equal(mask1, mask2):
        assert np.array_equal(mask, mask1) or np.array_equal(mask, mask2)
        notes_additional += ' different zero counts... take the intersection'
        print('masks may not be reliable!')
        strict_check = False
    else:
        assert np.array_equal(mask, mask1) and np.array_equal(mask, mask2)

    response_count = np.sum(mask, axis=1, dtype=np.uint8)  # uint8 should suffice...
    num_stimulus, max_trial, num_neuron = response_raw_f0s.shape

    notes = 'originally from {}, {} neurons, {} stimulus, {} max trial, in fact {} to {}'.format(tang_data_root_this,
                                                                                                 num_neuron,
                                                                                                 num_stimulus,
                                                                                                 max_trial,
                                                                                                 response_count.min(),
                                                                                                 response_count.max())
    notes += ' F0SFile {} (field {}), FSFile {} (field {}) '.format(f0sFile, f0sFileField,
                                                                    fsFile, fsFileField)
    notes += notes_additional

    response_all = fill_nan_for_raw_all(response_raw_f0s, response_raw_fs, response_count, strict_check)
    response_mean = np.nanmean(response_all, axis=1)
    notes += '; percentage of zero: {}'.format(np.mean(response_all[np.isfinite(response_all)] == 0))
    print(notes)
    assert np.all(np.isfinite(response_mean))

    write_hdf5_group(grp_handle_, notes,
                     ['all', 'mean', 'count', 'raw_fs', 'raw_f0s'], [response_all, response_mean, response_count,
                                                                     response_raw_fs, response_raw_f0s])


def write_monkey_a_pt_20160313(grp_handle):
    tang_data_root_this = os.path.join(tang_data_root, 'Data update(20160313)', 'Monkey A', 'Shape')
    write_monkey_corrected_data(grp_handle, tang_data_root_this, 'Rsp8OT5S.mat', 'NumE.mat', 'NumO.mat',
                                hack_data_func=hack_data_func_monkey_a_pt_20160313)


def hack_data_func_monkey_a_pt_20160313(response_all, response_count):
    # make those with NO trial to have at least 1 trial. How can we get such data in the first place?
    zero_count_mask = (response_count == 0)
    print('zero count records {} out of {}'.format(zero_count_mask.sum(), zero_count_mask.size))
    response_count[zero_count_mask] = 1

    count_it = np.nditer(response_count, flags=['multi_index'])
    hacked_count = 0
    while not count_it.finished:
        # check count is correct.
        idx_stimulus, idx_neuron = count_it.multi_index
        count_this = count_it[0]
        if not np.all(response_all[idx_stimulus, count_this:, idx_neuron] == 0):
            #             print('{} after {} have non-zero values'.format(
            #                     np.sum(response_all[idx_stimulus, count_this:, idx_neuron]!=0), count_this))
            response_all[idx_stimulus, count_this:, idx_neuron] = 0
            hacked_count += 1
        count_it.iternext()
    print('Hacked {} records, out of {}'.format(hacked_count, response_all.size))


def write_monkey_e_pt4605_2017(grp_handle):
    tang_data_root_this = os.path.join(tang_data_root, 'Data_MkE_Ptn_2017')
    write_monkey_corrected_data(grp_handle, tang_data_root_this, 'Rsp4OT5S.mat', 'NumE4OT5.mat',
                                'NumO4OT5.mat')


result_file = os.path.join(dir_dictionary['datasets'], 'tang_neural_data.hdf5')

group_list = (
    'monkeyA/Shape_9500/corrected_20160313',
    # 'monkeyE/Shape_4620/20161207',
    # this is actually monkey E. but due to the structure of my code, I need a new monkey name.
    'monkeyE2/Shape_4605/2017',
)

dispatch_function_list = (
    write_monkey_a_pt_20160313,  # pt -> pattern, a term used by Tai Sing by Shape_9500.
    # write_monkey_e_pt4620_20161207,
    write_monkey_e_pt4605_2017,
)
assert len(group_list) == len(dispatch_function_list)

with h5py.File(result_file) as f:
    for grp, func in zip(group_list, dispatch_function_list):
        if grp not in f:
            print('processing {}'.format(grp))
            grp_handle_outer = f.create_group(grp)
            func(grp_handle_outer)
        else:
            print('{} already done'.format(grp))
