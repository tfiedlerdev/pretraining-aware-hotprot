#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user leon.hermann@student.hpi.de
#SBATCH --partition=sorcery
#SBATCH --time=3-00:00:00
#SBATCH --exclude=ac922-[01-02]

eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/leon.hermann/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/leon.hermann/mambaforge/envs/hotprot/lib"
srun wandb agent hotprot/hot-prot-applications/eu165gzr
