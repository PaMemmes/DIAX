#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node

source /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/etc/profile.d/conda.sh

conda create --name gan -y
conda activate gan 
conda install -c conda-forge keras-tuner -y
conda install pandas -y
conda install matplotlib -y
conda install seaborn -c conda-forge -y
conda install -c conda-forge scikit-plot -y
conda install -c conda-forge xgboost -y
conda install -c conda-forge shap -y

python main.py 4 4 4

conda deactivate
conda env remove -n gan