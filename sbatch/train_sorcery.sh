#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --mail-user tobias.fiedler@student.hpi.de
#SBATCH --time=5-00:00:00
#SBATCH --partition=sorcery
#SBATCH --exclude=ac922-[01-02]

eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/tobias.fiedler/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/tobias.fiedler/miniconda3/envs/hotprot/lib"
srun wandb agent "hotprot/hot-prot-applications/$1"

