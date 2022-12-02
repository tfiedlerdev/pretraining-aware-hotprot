# HotProt

## Setup

Create new virtual environment
Run `pip install -r requirements.txt`

## Steps Taken

### Dataset

1. Download [FLIP dataset](https://benchmark.protein.properties/landscapes) for protein thermostabilities
2. Download the ids of the [evaluation proteins](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur50_valid_headers.txt.gz) excluded from training of ESM language model
3. Run the `data_filtering.ipynb` to extract proteins from the FLIP dataset that haven't been used in training of ESM language model for our evaluation dataset to avoid data leakage and others for our training dataset
4. The fasta files for our evaluation and training data are saved in the `FLIP` directory

### Embeddings

- [ ] TODO: find out which layers we want the embeddings of

Run `python esm-main/scripts/extract.py esm2_t33_650M_UR50D FLIP/eval_sequences.fasta esm_embeddings/default_repr_layers/per_tok/eval --include per_tok` to create protein embeddings from ESM
