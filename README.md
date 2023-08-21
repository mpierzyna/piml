# Π-ML: A dimensional analysis-based machine learning framework for physical modeling


## Reference
Pierzyna, M., Saathof, R., and Basu, S. "Π-ML: A dimensional analysis-based machine learning parameterization of 
optical turbulence in the atmospheric surface layer". _Optics Letters_, vol. 48, no. 17, 2023, pp. 4484-4487. 

DOI: https://doi.org/10.1364/OL.492652 (also on [arxiv](https://arxiv.org/abs/2304.12177))

## Setup
1.  Clone or download this git repository and its submodules:
    ```shell
    git clone --recurse-submodules https://github.com/mpierzyna/piml
    ```
2.  Set up the required Python packages in a new conda environment:
    ```shell
    conda env create -f environment.yml
    ```
    **Note**: The exact package versions from the environment file have to be used to guarantee that trained
    models can be loaded later.

## Example: Optical turbulence in the atmospheric surface layer
An example workspace set up to model optical turbulence strength $C_n^2$ following Pierzyna et al. (2023) is given 
in [workspace/cn2_mlo](workspace/cn2_mlo).
This directory also contains the trained $C_n^2$ models (**coming soon**).

## Quick start
### Set up a new workspace
1. Create new a workspace for by copying the template folder:
    ```shell
    cp -r workspace/template workspace/my_workspace[config.yml](workspace%2Fcn2_mlo%2Fconfig.yml)
    ```
2. For convenience, set the path to your workspace as environment variable:
    ```shell
    export PIML_WORKSPACE=workspace/my_workspace
    ```
   Note, this needs to be repeated everytime you open a new terminal.
3. Follow the example/tutorial in [workspace/cn2_mlo](workspace/cn2_mlo) to set up your own `config.yml`.

### Run the model pipeline
Activate the conda environment:
```shell
conda activate piml
```

1. `step_1_make_pi_sets.py`: Generates all possible $\Pi$-sets based on variables in `config.yml` and saves them
   to `my_workspace/1_raw/pi_sets_full.joblib`.
2. `step_2_constrain_pi_sets.py`: Apply the following constraints to reduce the number of possible $\Pi$-sets. 
   Please refer to our paper for more details. If you require different or more constraints, you need to modify the code.
    - Each $\Pi$-set can only contain a single $\Pi$-group that is function of the model output/target.
    - Signed dim. variables, have to retain their sign, so, e.g., squared versions of that variable are not allowed.
3. `step_3_split_train_test.py`: Split dimensional dataset into training and testing portions and make sure it is valid for training.
4. `step_4_train_ensemble.py`: Train ensemble of models for each valid $\Pi$-set. By default training happens 
    sequentially, which might take a long time. To train models in parallel, supply the `--pi_set=...` flag to train
    only a specific $\Pi$-set and use the array functionality of your HPC scheduler to run multiple jobs with increasing
    integer values for `--pi_set=...`.
5. `step_5_eval_ensemble.py`: Evaluate the trained ensemble of models on the test dataset and plot diagnostic figures.