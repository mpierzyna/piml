# Using Π-ML to model optical turbulence in the atmospheric surface layer
## Preliminary
This is a high-level, user-focused tutorial about how to use $\Pi$-ML the physical problem of optical turbulence
in the atmospheric surface layer. It is beyond the scope of this tutorial to introduce optical turbulence or the 
physical details of $\Pi$-ML. For more information, please refer to our accompanying paper.

Pierzyna, M., Saathof, R., and Basu, S. "Π-ML: A dimensional analysis-based machine learning parameterization of
optical turbulence in the atmospheric surface layer". _Optics Letters_, vol. 48, no. 17, 2023, pp. 4484-4487.

DOI: https://doi.org/10.1364/OL.492652 (also on [arxiv](https://arxiv.org/abs/2304.12177))

### Dataset
The trained $C_n^2$ models are uploaded to Zenodo: https://zenodo.org/record/8316104.

## Setup
Please follow the instructions in the [README.md](../README.md) of the main repository to set up the required Python
packages in a new conda environment. Afterward, activate the conda environment:
```shell
conda activate piml
```

## Create `config.yml`
The `config.yml` file is used to specify the parameters of the $\Pi$-ML model. It has to be called `config.yml` and
needs to sit in the root of your workspace. Please refer to the [README.md](../README.md) of the main repository for
more information about how to set up a workspace.

The `config.yml` file is a YAML file that contains a dictionary with the following keys. The example [`config.yml`](config.yml)
is fully working, so you can use it as a template for your own problem.
- `dim_vars`: List the dimensional variables that describe your problem with dimensions and if the variable can be signed
  (i.e., can be negative) or not.
  - `inputs`: Variables used as inputs to the model (features). Each variable is a dictionary with the following keys:
    - `symbol`: Symbol of the variable (`string`).
    - `signed`: Boolean indicating if the variable can be signed (`bool`).
    - `dimensions`: Dimensions of the variable (`string`).
  - `output`: Variables used as outputs of the model (labels)
- `dataset`: CSV dataset used for training. This needs to be preprocessed already meaning that all variables specified
   under `dim_vars` need to be present in the CSV file. 
  - `path`: Path to the CSV file (`string`) but prefixed with `!path`.
  - `test_interval`: Dates used to split dataset into train and test sets.
  - `col_to_var`: If your column names differ from the symbols you specified under `dim_vars`, you can use this dictionary
    to map the column names to the symbols.
  - `target_transforms`: Transformations applied to target `y`. See section below for more details
    - `pre_pi`: Transformation applied before dimensional data are transformed to $\Pi$ variables.
    - `pre_train`: Transformations applied to $\Pi$ variables before training, e.g. `np.log10`.
- `flaml`: Settings for FLAML automl framework. For full overview, see 
  [FLAML documentation](https://microsoft.github.io/FLAML/docs/reference/automl/automl/#automl-objects). 
  Important parameters are:
  - `estimator_list`: List of estimators to use, e.g., `xgboost`, `lightgbm`, `rf`, `lgbm`, `catboost`, or `extra_tree`.
  - `time_budget`: Time budget in seconds for the automl run **per ensemble member**.
  - `metric`: Metric to minimize, e.g., root-mean-squared-error `rmse`
  - `n_jobs`: Number of parallel jobs to run.
- `n_members`: Number of members each ensemble should have.
 
_Developer note_: The `config.yml` file is parsed using the `pydantic` and `pyyaml` packages. You can find the 
Python models for each section under the [piml/config](../piml/config).

## Run pipeline
Make sure, you have activated the `piml` conda environment: `conda activate piml`. Also, set the path to the workspace
as environment variable: `export PIML_WORKSPACE=workspace/cn2_mlo`. 

### 1. Create $\Pi$ groups/sets
#### `step_1_make_pi_sets.py`
todo

