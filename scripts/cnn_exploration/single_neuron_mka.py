from tang_jcompneuro.cnn_exploration_pytorch import explore_one_neuron_1L

from sys import argv
from torch.backends import cudnn

# this config is fastest.
cudnn.benchmark = False
cudnn.enabled = True

if __name__ == '__main__':
    arch_name, subset, neuron = argv[1:]
    neuron = int(neuron)
    explore_one_neuron_1L(arch_name, neuron, subset)
