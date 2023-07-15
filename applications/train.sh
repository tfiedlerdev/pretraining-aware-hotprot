#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user leon.hermann@student.hpi.de
#SBATCH --partition=sorcery
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ac922-[01-02]

eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/leon.hermann/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/leon.hermann/mambaforge/envs/hotprot/lib"
srun python data_analysis_generation/generate_representations.py /hpi/fs00/scratch/tobias.fiedler/allSequences.txt /hpi/fs00/scratch/leon.hermann/data/esm_3B esm2_t36_3B_UR50D 

