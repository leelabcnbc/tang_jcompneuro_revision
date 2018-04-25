# make sure that pytorch is not imported.
# https://docs.python.org/3/library/sys.html#sys.modules

import sys, pickle, os
from tang_jcompneuro import dir_dictionary


def main():
    print('before', sys.modules.keys())
    vgg_info_filename = os.path.join(dir_dictionary['analyses'], 'vgg19_pytorch_dump.pkl')
    with open(vgg_info_filename, 'rb') as f:
        # pickle.dump({
        #     'slicing_dict': slicing_dict,
        #     'vgg19_weights_dict': vgg_weights_dict_final,
        # }, f, protocol=pickle.HIGHEST_PROTOCOL)
        vgg19_info = pickle.load(f)
    print(vgg19_info.keys())
    # visually examined, it does not contain 'torch' or anything like that.
    print('after', sys.modules.keys())


if __name__ == '__main__':
    main()
