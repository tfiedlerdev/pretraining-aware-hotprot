#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user leon.hermann@student.hpi.de
#SBATCH --time=70:0:0

eval "$(conda shell.bash hook)"
conda activate hotshot
export PYTHONPATH="/hpi/fs00/home/hoangan.nguyen/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/hoangan.nguyen/anaconda3/envs/hotshot/lib"
srun wandb agent hotprot/hot-prot-applications/wmwgymf2
