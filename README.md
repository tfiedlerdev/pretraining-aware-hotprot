# Data Leakage from Protein LLM Pretraining
This repository contains the code and results related to our paper Beware of Data Leakage from Protein LLM Pretraining.
We measure the effects of data leakage from Protein Language Model (PLM) pretraining on the downstream task of protein thermostability prediction.
We do this by comparing the performance of a simple fully connected neural network attached to different [ESM](https://github.com/facebookresearch/esm) variants on two different dataset split strategies: one that considers which data ESM was pretrained on (EPA - ours) and one which doesn't (FLIP, from [FLIP paper](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1)).

## Results

Take a look at the raw results including commands for reproducing them in tabular form [here](./results.md).

## Copyright declaration
We reuse and modify the the source code of [ESM](https://github.com/facebookresearch/esm) created by Meta under the [MIT License](https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/LICENSE). The same goes for code snippets for factorized sparse tuning (not included in the paper) from [HotProtein](https://github.com/VITA-Group/HotProtein). They also allow reuse via the [MIT License](https://github.com/VITA-Group/HotProtein/blob/9cf0cbaf4454d5b3b266e1eb9a7d1b5060e2bf15/LICENSE).

## Setup
These steps can be taken to setup the project on a linux machine.
1. Clone the repo with `git clone https://github.com/tfiedlerdev/pretraining-aware-hotprot.git --recurse-submodules`
2. Run `./install_dependencies.sh`. You need to add the path to your conda.sh file in the script. Or follow the different steps manually. The package `transformers` automatically installs a torch version that we don't want. That is why we have to uninstall it first, so we can install ours. That is also done in the script.
3. Activate conda enviroment with `conda activate hotprot`

### Imports
- If you are using vscode and want to work with jupyter notebooks, go into vscode setting and set Jupyter: Notebook File Root to `${workspaceFolder}`
- To make the relative imports work set the Env Variable to your current project dir, e.g 
```sh
export PYTHONPATH="$PYTHONPATH:/path/to/your/project/"
```

### Data
The train,validation and test sets for FLIP and EPA are pregenerated and can be found under [./data](./data). 
You can generate them yourself by following the following steps, otherwise you can skip these steps:

We have prepared a ZIP archive containing 
- all data required for generating our train, test and validation sets 
- (not relevant) a pretrained model for ESM and ProtT5 representations
- (not relevant) ESM per protein representation for all sequences in our train/validation/test set with a length < 700
- (not relevant) ProtT5 per protein representation for all sequences in our train/validation/test set with a length < 700

1. Run data setup script: `bash setup_data.sh`. If this does not work, manually download the ZIP via [this link](https://drive.google.com/uc?export=download&id=1Im3y2X6iwhZHFJLIOKbIGtZ0ZHBWaqD_) and unzip the contents to the working directory (`unzip data.zip -d .`)
2. Generate our train/test set by executing all cells in [`create_datasets.ipynb`](data_analysis_generation/create_datasets.ipynb) Jupyter Notebook. This will create the HotProt (EPA split but all measurements for each protein are included), EPA (called HotProt median in code) and FLIP splits.

## Resouces
- [ESM2 language model paper](https://www.biorxiv.org/content/10.1101/622803v4)
- [ESM Github Repo](https://github.com/facebookresearch/esm)
- [UniProt Protein Embeddings](https://www.uniprot.org/help/embeddings)
- [UniProt Protein Embeddings Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9477085&tag=1)
- [FLIP Dataset](https://benchmark.protein.properties/)(the download link on their website might not work. In that case you can download the dataset from their github repo)
- [FLIP Dataset Paper](https://www.nature.com/articles/s41592-020-0801-4)
- [HotProtein Paper](https://openreview.net/forum?id=YDJRFWBMNby)


## Known Errors 
1. `GLIBCXX_3.4.30' not found' - set your LD_LIBRARY_PATH to your conda environment, i.e 
```sh
EXPORT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/anaconda3/envs/hotprot/lib
```



