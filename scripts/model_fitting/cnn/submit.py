from sys import argv

from tang_jcompneuro.model_fitting import run_all_scripts, generate_all_scripts
from tang_jcompneuro.model_fitting_cnn import models_to_train, models_to_train_detailed_keys

header = """
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --time=96:00:00
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-1-11
# --exclude is to reserve that bad node
""".strip()

if __name__ == '__main__':
    use_slurm = len(argv) == 1
    # script_dict = generate_all_scripts(header, 'cnn', models_to_train.keys())
    # script_dict = generate_all_scripts(header, 'cnn', models_to_train_detailed_keys)
    script_dict = generate_all_scripts(header, 'cnn', (
                                                      'b.9',
                                                       'b.1', 'b.2', 'b.3', 'b.4', 'b.5',
                                                       'b.6', 'b.7', 'b.8',
                                                       'b.10',
                                                       'b.11', 'b.12', 'b.15', 'b.18'
                                                       ))
    # print(script_dict.keys(), len(script_dict))
    run_all_scripts(script_dict, slurm=use_slurm)
