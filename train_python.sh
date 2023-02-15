#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --partition sorcery
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user leon.hermann@student.hpi.de


eval "$(conda shell.bash hook)"
conda activate hotshot
srun jupyter nbconvert --execute --to notebook --inplace uni_prot.ipynb