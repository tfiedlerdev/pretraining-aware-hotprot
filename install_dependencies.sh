#!/bin/bash
conda_path="../miniconda3/etc/profile.d/conda.sh"
source ${conda_path}
conda env create -f environment.yml
conda activate hotprot
pip uninstall torch
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install fair-esm fair-esm[esmfold] pytorch-lightning==1.9.4
pip install dllogger@git+https://github.com/NVIDIA/dllogger.git
pip install openfold@git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307