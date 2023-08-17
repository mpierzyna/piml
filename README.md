# Π-ML: A dimensional analysis-based machine learning framework for physical modeling


## Reference
Pierzyna, M., Saathof, R., and Basu, S. "Π-ML: A dimensional analysis-based machine learning parameterization of 
optical turbulence in the atmospheric surface layer". _Optics Letters_. 2023. pp. ???-???.

DOI: https://doi.org/10.1364/OL.492652 ([arxiv](https://arxiv.org/abs/2304.12177))

## Setup
1.  Clone or download this git repository and its submodule:
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

## Quick start
1. Create a workspace for your problem
2. Generate all possible $\Pi$ sets: