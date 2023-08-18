#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user tobias.fiedler@student.hpi.de
#SBATCH --time=120:0:0 

eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/tobias.fiedler/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/tobias.fiedler/miniconda3/envs/hotprot/lib"
srun python3 applications/train.py --batch_size=32 --epochs=30 --learning_rate=0.001 --model=summarizer --model_dropoutrate=0.5 --model_first_hidden_units=1024 --model_hidden_layers=2 --optimizer=adam --representation_key=s_s --summarizer_activation=identity --summarizer_mode=per_repr_position --summarizer_num_layers=1 --summarizer_out_size=1 --summarizer_type=average --val_on_trainset=false --wandb --early_stopping --wandb_run_name=base_model --bin_width=5 

