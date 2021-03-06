# tang_jcompneuro_revision
revision of Tang JCNS paper

## setup

### install dependencies

***TBA***

check `environment.yml` for actual set of dependencies.

this file was produced by `conda env export > environment.yml`.

Check <https://conda.io/docs/user-guide/tasks/manage-environments.html#exporting-the-environment-file>

In addition, you need the following set of libraries not installed in the conda environment. Specific commit SHAs are given.

* <https://github.com/leelabcnbc/strflab-python/commit/34d6fbe1e79f07a9469ab86fb6a57a6a99fded79>
* <https://github.com/leelabcnbc/leelab-toolbox/commit/68b11e8143fff788d07bf71353dc3aa058604137>
* <https://github.com/raghakot/keras-vis/commit/40b27dfa3ecb84cdde5ec6b44251923c3266cc40>

Plus those R things below, you will be good.

~~~bash
# install R stuff.
conda install -c conda-forge r-base=3.4.1 rpy2=2.8.5
# inside R (run `R`), install glmnet by `install.packages("glmnet")`
~~~

### set up symbolic links for private data.

after cloning it, create a symbolic link `tang_data` under
`private_data` to link to `/data2/leelab/data/tang_data`, the raw data files from Professor Tang,
by `ln -s /data2/leelab/data/tang_data/ tang_data` Notice that `tang_data` should not be there before. Otherwise,
do `rm tang_data` (NOT `tang_data/`!!! That way, actual data will be removed).

### run these every time before using

~~~bash
. activate tang_jcompneuro_revision
cd ~/tang_jcompneuro_revision
. ./setup_env_variables.sh
~~~

## structure

* `results_ipynb` all the human-readable results, typically in the format of Jupyter notebooks (`.ipynb`).
* `results` all the raw results produced by scripts, notebooks, etc.
* `private_data` all private data, such as Tang's neural data and raw stimuli.
* `tang_jcompneuro` the package part of the project.
* `scripts` all the other scripts used to generate raw results.
* `thesis_plots` source file for figures in the thesis proposal. 
* `slides` some slides.

### `scripts`

run all scripts using `python`. For example, `python scripts/preprocessing/convert_image_dataset.py` for
`convert_image_dataset.py` under `preprocessing`.

#### `preprocessing`

scripts to convert Tang's raw data into HDF5 format ready for late processing.

* `convert_image_dataset.py` convert image data sets.
* `convert_tang_neural_dataset.py` convert neural data sets.
* `split_datasets.py` generate train, val, and test datasets for pairs of neural and image data.
