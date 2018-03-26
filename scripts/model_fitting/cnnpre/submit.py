from sys import argv

from tang_jcompneuro.model_fitting import run_all_scripts, generate_all_scripts
from tang_jcompneuro.model_fitting_cnnpre import subtype_to_train

header = """
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --exclude=compute-1-11
# --exclude is to reserve that bad node
""".strip()

if __name__ == '__main__':
    use_slurm = len(argv) == 1
    print(subtype_to_train)
    # input('haha')
    script_dict = generate_all_scripts(header, 'cnnpre', subtype_to_train,
                                       # override={
                                       #     # 'seed_list': range(1),
                                       #     'neural_dataset_to_process': ('MkA_Shape',),
                                       #     'subset_list': ('all',)
                                       # },
                                       )
    run_all_scripts(script_dict, slurm=use_slurm)
