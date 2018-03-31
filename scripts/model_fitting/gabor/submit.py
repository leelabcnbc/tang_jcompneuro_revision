from sys import argv

from tang_jcompneuro.model_fitting import run_all_scripts, generate_all_scripts
from tang_jcompneuro.model_fitting_gabor import models_to_train

header = """
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-1-11
# --exclude is to reserve that bad node
""".strip()

if __name__ == '__main__':
    use_slurm = len(argv) == 1
    print(models_to_train)
    # input('haha')
    script_dict = generate_all_scripts(header, 'gabor',models_to_train,
                                       # ('multi,2,1',),
                                       # override={
                                       #     # 'seed_list': [1],
                                       #     'neural_dataset_to_process': ('MkE2_Shape',),
                                       #     'subset_list': ('all',)
                                       # },
                                       )
    run_all_scripts(script_dict, slurm=use_slurm)
