#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user leon.hermann@student.hpi.de
#SBATCH --time=70:0:0

eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/hoangan.nguyen/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/hoangan.nguyen/anaconda3/envs/hotprot/lib"
srun python data_analysis_generation/generate_representations.py data/all_sequences.txt data/uni_prot/T5/ --model="protT5" --telegram --repr_key="prott5_avg"   

