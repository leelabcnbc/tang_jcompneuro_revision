"""master script for model fittting"""

from tang_jcompneuro.model_fitting import train_one_case_generic
from sys import argv


def main():
    model_type, model_subtype, dataset_spec, neuron_spec = argv[1:]
    train_one_case_generic(model_type, model_subtype, dataset_spec, neuron_spec)


if __name__ == '__main__':
    main()
