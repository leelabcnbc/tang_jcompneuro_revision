from os import chdir, remove
import os.path
from subprocess import run
from itertools import product
from tempfile import NamedTemporaryFile
from tang_jcompneuro import dir_dictionary
from maskcnn.training_aux_wrapper import all_opt_configs, gen_all_arch_config
from sys import argv

template_function = """
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=11:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
# 1-11 is for other people.
#SBATCH --exclude=compute-1-11
# --exclude is to reserve that bad node

. activate tf15
# wait for a while. otherwise, it may not work... maybe some bug of conda.
cd {root}
. ./setup_env_variables.sh

PYTHONUNBUFFERED=1 python scripts/model_fitting/cnn_population/run.py {{dataset}} {{image_subset}} {{neuron_subset}} {{seed}} {{arch_config}} {{opt_config}}  2>&1 | tee {root}/trash/cnn_pop_{{dataset}}_{{image_subset}}_{{neuron_subset}}_{{seed}}_{{arch_config}}_{{opt_config}}_out
""".strip().format(root=dir_dictionary['root'])

# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

if __name__ == '__main__':

    use_slurm = len(argv) == 1

    trash_global = os.path.join(dir_dictionary['root'], 'trash')
    chdir(trash_global)

    dataset_list = ('MkA_Shape',)
    image_subset_list = ('all',)
    neuron_subset_list = ('OT', 'HO')
    seed_list = range(2)

    for dataset, image_subset, neuron_subset, seed in product(dataset_list, image_subset_list, neuron_subset_list,
                                                              seed_list):
        arch_config_list = gen_all_arch_config(dataset, image_subset, neuron_subset)
        for arch_config_name, opt_config_name in product(arch_config_list.keys(), all_opt_configs.keys()):
            str_to_write_full = template_function.format(dataset=dataset, image_subset=image_subset,
                                                         neuron_subset=neuron_subset, seed=seed,
                                                         arch_config=arch_config_name,
                                                         opt_config=opt_config_name)
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
            print(dataset, image_subset, neuron_subset, seed, arch_config_name, opt_config_name)
            if use_slurm:
                run(['sbatch', file_temp.name], check=True)
            else:
                os.chmod(file_temp.name, 0o755)
                run(file_temp.name, check=True)
            remove(file_temp.name)
