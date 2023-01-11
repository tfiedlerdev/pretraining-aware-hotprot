# HotProt
## Resouces
- [ESM2 language model paper](https://www.biorxiv.org/content/10.1101/622803v4)
- [ESMFold paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2.full.pdf)
- [ESM Github Repo](https://github.com/facebookresearch/esm)

## Setup

Create new virtual environment
Run `pip install -r requirements.txt`
OR
Run 
1. `conda env create -f environment.yml`
2. `conda activate hotprot`
## Steps Taken

### Dataset

1. Download [FLIP dataset](https://benchmark.protein.properties/landscapes) for protein thermostabilities
2. Download the ids of the [evaluation proteins](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur50_valid_headers.txt.gz) excluded from training of ESM language model
3. Run the `data_filtering.ipynb` to extract proteins from the FLIP dataset that haven't been used in training of ESM language model for our evaluation dataset to avoid data leakage and others for our training dataset
4. The fasta files for our evaluation and training data are saved in the `FLIP` directory

### Embeddings

- [ ] TODO: find out which layers we want the embeddings of

Run `python esm-main/scripts/extract.py esm2_t33_650M_UR50D FLIP/eval_sequences.fasta esm_embeddings/default_repr_layers/per_tok/eval --include per_tok` to create protein embeddings from ESM