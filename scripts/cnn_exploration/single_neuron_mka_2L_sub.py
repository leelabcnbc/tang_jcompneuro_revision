from os import chdir, remove
import os.path
from subprocess import run
from itertools import product
from tempfile import NamedTemporaryFile
from tang_jcompneuro import dir_dictionary
from tang_jcompneuro.cnn_exploration import two_layer_models_to_explore, neuron_to_explore_idx
from sys import argv

template_function = """
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
# 1-11 is for other people.
#SBATCH --exclude=compute-1-11
# --exclude is to reserve that bad node

. activate tf15
# wait for a while. otherwise, it may not work... maybe some bug of conda.
cd {root}
. ./setup_env_variables.sh

PYTHONUNBUFFERED=1 python scripts/cnn_exploration/single_neuron_mka_2L.py {{arch_name}} {{subset}} {{neuron}} 2>&1 | tee {root}/trash/cnn_exp_2L_{{arch_name}}_{{subset}}_{{neuron}}_out
""".strip().format(root=dir_dictionary['root'])

# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

if __name__ == '__main__':

    use_slurm = len(argv) == 1

    # right now, the "all" results are more important.
    # get all neurons to check.
    neurons_to_check_dict = neuron_to_explore_idx(merged=True)

    # to be fast first.
    # additional_iter = range(150)
    # run 50 cases for each neuron and loc.

    trash_global = os.path.join(dir_dictionary['root'], 'trash')
    chdir(trash_global)

    for arch_name, subset in product(two_layer_models_to_explore(), neurons_to_check_dict):

        neurons_to_expore_this_subset = neurons_to_check_dict[subset]
        for neuron_idx in neurons_to_expore_this_subset:
            str_to_write_full = template_function.format(neuron=neuron_idx, subset=subset,
                                                         arch_name=arch_name)
            # if arch_name not in {'k9c3_nobn_k3s3max_dropout'}:
            #     continue

            # print(str_to_write_full)
            # input('hi')
            # https://stackoverflow.com/questions/19543139/bash-script-processing-commands-in-parallel
            # str_to_write_full += 'wait\n'
            # # print(str_to_write_full)
            # # input('hi')
            file_temp = NamedTemporaryFile(delete=False)
            file_temp.write(str_to_write_full.encode('utf-8'))
            file_temp.close()
            print(arch_name, subset, neuron_idx)
            if use_slurm:
                run(['sbatch', file_temp.name], check=True)
            else:
                os.chmod(file_temp.name, 0o755)
                run(file_temp.name, check=True)
            remove(file_temp.name)
