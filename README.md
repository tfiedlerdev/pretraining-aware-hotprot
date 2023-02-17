# HotProt
## Resouces
- [ESM2 language model paper](https://www.biorxiv.org/content/10.1101/622803v4)
- [ESMFold paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2.full.pdf)
- [ESM Github Repo](https://github.com/facebookresearch/esm)
- [UniProt Protein Embeddings](https://www.uniprot.org/help/embeddings)
- [UniProt Protein Embeddings Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9477085&tag=1)
- [FLIP Dataset](https://benchmark.protein.properties/)(the download link on their website might not work. In that case you can download the dataset from their github repo)
- [FLIP Dataset Paper](https://www.nature.com/articles/s41592-020-0801-4)

## Setup
These steps can be taken to setup the project on a linux machine.
1. Clone the repo with 'git clone https://github.com/LeonHermann322/hot-prot.git --recurse --submodules'
2. `conda env create -f environment.yml`
3. `conda activate hotprot`
4. You then have to manually install openfold and fair-esm
```sh
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install fair-esm
pip install fair-esm[esmfold]
pip install dllogger@git+https://github.com/NVIDIA/dllogger.git
pip install openfold@git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307
```

### Imports
- If you are using vscode and want to work with jupyter notebooks, go into vscode setting and set Jupyter: Notebook File Root to '${workspaceFolder}'
- To make the relative imports work set the Env Variable to your current project dir, e.g 
'''sh
export PYTHONPATH="$PYTHONPATH:/path/to/your/project/"
'''

### Dataset

1. Download [FLIP dataset](https://benchmark.protein.properties/landscapes) for protein thermostabilities
2. Download the ids of the [evaluation proteins](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur50_valid_headers.txt.gz) excluded from training of ESM language model
3. Run the `data_filtering.ipynb` to extract proteins from the FLIP dataset that haven't been used in training of ESM language model for our evaluation dataset to avoid data leakage and others for our training dataset
4. The fasta files for our evaluation and training data are saved in the `FLIP` directory
5. Download the averaged s_s0 representations TODO

## Train with hyperparamers with wandb

1. Edit the configuration in the 'training/hyperparameter_esm.yaml' if you like to
2. Then run 'wandb sweep training/hyperparameter_esm.yaml' 
3. You'll be prompted to run the agent by wandb. You will also be provided with a link where you can see immediate results



