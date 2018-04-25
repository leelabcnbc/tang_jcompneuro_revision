# save slice dict on disk, to avoid loading of pytorch, as this will be used by TensorFlow (Keras) later.
import os
import pickle
from tang_jcompneuro.cnn_pretrained import get_one_network_meta, blob_corresponding_info
from torchvision.models import vgg19
from collections import OrderedDict
from tang_jcompneuro import dir_dictionary


def main():
    (helper_this, slicing_dict,
     blobs_to_extract, correspondence_func) = get_one_network_meta('vgg19')
    print(slicing_dict)
    # print(correspondence_func)

    # save vgg weights
    vgg19_this = vgg19(pretrained=True)

    vgg_weights_dict = {}
    for x, y in vgg19_this.named_parameters():
        vgg_weights_dict[x] = y.data.numpy()

    vgg_blob_info = blob_corresponding_info['vgg19']
    vgg_weights_dict_final = OrderedDict()
    for layer_name, layer_name_pytorch in vgg_blob_info.items():
        if layer_name.startswith('conv'):
            # -1 for relu.
            _, layer_idx = layer_name_pytorch.split('.')
            assert _ == 'features'
            layer_idx = int(layer_idx) - 1
            layer_name_pytorch_new = f'{_}.{layer_idx}'
            vgg_weights_dict_final[layer_name] = {
                'weight': vgg_weights_dict[layer_name_pytorch_new + '.weight'],
                'bias': vgg_weights_dict[layer_name_pytorch_new + '.bias'],
            }
        elif layer_name.startswith('pool'):
            vgg_weights_dict_final[layer_name] = None
        else:
            assert layer_name.startswith('fc')
            pass
    # print(vgg_weights_dict_final)
    vgg_info_filename = os.path.join(dir_dictionary['analyses'], 'vgg19_pytorch_dump.pkl')
    with open(vgg_info_filename, 'wb') as f:
        pickle.dump({
            'slicing_dict': slicing_dict,
            'vgg19_weights_dict': vgg_weights_dict_final,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
