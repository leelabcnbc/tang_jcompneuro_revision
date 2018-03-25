from maskcnn import training_aux_wrapper

from sys import argv


def main():
    dataset, image_subset, neuron_subset, seed, arch_name, opt_name = argv[1:]
    training_aux_wrapper.train_one_wrapper(dataset, image_subset, neuron_subset, int(seed), arch_name, opt_name)


if __name__ == '__main__':
    main()
