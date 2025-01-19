# Practice Pytorch


## Conda Setup
* source ~/anaconda3/etc/profile.d/conda.sh 
* conda env update -f environment.yml --prune
* conda activate pytorch_env

## Conda Lock (optional)
* conda-lock lock -p linux-64
* conda-lock install -n pytorch_env conda-lock.yml

## Optional Setup
* direnv / .envrc

## Simple Test
* python --version
* python -c "import torch; print(torch.__version__)"
