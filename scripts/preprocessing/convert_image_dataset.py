"""master file to convert all needed datasets to HDF5 format"""
import os.path
from collections import OrderedDict

import h5py
import numpy as np
import skimage
import skimage.color
import skimage.io

from tang_jcompneuro import dir_dictionary


def save_shape9500(grp):
    # load Shape9500 dataset.
    shape9500_img_root = os.path.join(dir_dictionary['tang_data_root'], 'stimuli', 'Shape')
    img_filelist = ['{}.png'.format(x + 1) for x in range(9500)]
    imagelist_raw_raw = [skimage.io.imread(os.path.join(shape9500_img_root, thisfile)) for thisfile in img_filelist]
    imagelist_raw_raw = np.array(imagelist_raw_raw)
    print("shape of dataset: gray: {}".format(imagelist_raw_raw.shape, imagelist_raw_raw.shape))
    print("type of dataset: gray: {}".format(imagelist_raw_raw.dtype, imagelist_raw_raw.dtype))
    grp.create_dataset('original', data=imagelist_raw_raw, compression='gzip')


def save_shape4605(grp):
    shape9500_img_root = os.path.join(dir_dictionary['tang_data_root'], 'Data_MkE_Ptn_2017', 'StimiMkE.mat')
    # this mat file is actually HDF5 (v7.3 mat)
    with h5py.File(shape9500_img_root, 'r') as f:
        image_mat = f['StimiMkE'][...]
    image_mat = image_mat.T.astype(np.uint8)
    assert np.array_equal(np.unique(image_mat), [0, 32])
    image_mat[image_mat == 32] = 255
    # print(image_mat.shape, image_mat.dtype)
    # print(images_original.shape, images_original.dtype)
    assert np.array_equal(np.unique(image_mat), [0, 255])
    assert image_mat.shape == (4605, 160, 160)
    grp.create_dataset('original', data=image_mat, compression='gzip')


def main():
    image_data_file = os.path.join(dir_dictionary['datasets'], 'tang_stimulus.hdf5')
    dataset_dispatch_dict = OrderedDict()
    dataset_dispatch_dict['Shape_9500'] = save_shape9500  # expand this as you need.
    dataset_dispatch_dict['Shape_4605'] = save_shape4605  # expand this as you need.
    with h5py.File(image_data_file) as f_img:
        for key, func in dataset_dispatch_dict.items():
            if key not in f_img:
                print('processing {}'.format(key))
                g_handle = f_img.create_group(key)
                func(g_handle)
            else:
                print('{} already done before'.format(key))


if __name__ == '__main__':
    main()
