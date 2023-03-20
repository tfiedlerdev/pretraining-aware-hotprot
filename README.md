# HotProt

This project attempts to infer the thermostability (melting point) of a given protein sequence with an end-to-end approach, meaning no information other than the sequence is needed. For this we run a forward pass of the ESMFold model and infer the thermostability of the protein based on the ESMFold representations. 
With our pretrained model we have achieved a mean absolute difference (MAD) of 3.77°C between actual and predicted melting points over our test set. 
As there are multiple melting point measurements for many of the different proteins, a MAD of 0°C would not be possible. 
In our test set, the MAD of the melting point measurements difference to its proteins mean melting point is `1.262`, which would consequently also be the MAD of a perfect model.

## Results
These are the predictions of our pretrained model on the test set. Reproduce this via `python3 applications/train.py --batch_size=32 --epochs=30 --learning_rate=0.001 --model=fc --model_first_hidden_units=1024 --model_hidden_layers=2 --optimizer=adam --val_on_trainset=false --model_dropoutrate=0.7 --representation_key=s_s_avg --early_stopping` (the results might be slightly different due to different model initialization).

![image](https://user-images.githubusercontent.com/29177177/225330082-aa0a784a-e2b5-459b-b1f5-5213d84ed17e.png)
![image](https://user-images.githubusercontent.com/29177177/225330226-247779bf-5bd3-4079-b393-e487c8f91c7c.png)

Predictions on Validation set:

![val_predictions](https://user-images.githubusercontent.com/29177177/225331716-7c131750-1933-4abf-9b2d-fb1ae9e4af51.png)



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
1. Clone the repo with `git clone https://github.com/LeonHermann322/hot-prot.git --recurse-submodules`
2. `conda env create -f environment.yml`
3. `conda activate hotprot`
4. Because the package `transformers` automatically installs a torch version that we don't want, we have to uninstall it first, so we can install ours.
```sh
pip uninstall torch
```
5. You then have to manually install openfold,torch and fair-esm, otherwise conda crashes
```sh
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install fair-esm
pip install fair-esm[esmfold]
pip install dllogger@git+https://github.com/NVIDIA/dllogger.git
pip install openfold@git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307
```

### Imports
- If you are using vscode and want to work with jupyter notebooks, go into vscode setting and set Jupyter: Notebook File Root to `${workspaceFolder}`
- To make the relative imports work set the Env Variable to your current project dir, e.g 
```sh
export PYTHONPATH="$PYTHONPATH:/path/to/your/project/"
```

### Data
We have prepared a ZIP archive containing 
- all data required for generating our train and validation sets 
- a pretrained model
- ESM per protein representation for all sequences in our train/validation set with a length < 700

1. Run data setup script: `bash setup_data.sh`. If this does not work, manually download the ZIP via [this link](https://drive.google.com/file/d/1Og0z3jpjerZmHzdNXBohAt5JP9zFPM3r/view?usp=share_link) and unzip the contents to the working directory (`unzip data.zip -d .`)
2. Generate our train/test set by executing all cells in [`create_datasets.ipynb`](data_analysis_generation/create_datasets.ipynb) Jupyter Notebook

## Applications
For this all steps in **Setup** must have been executed successfuly.
### Inference
To run inference using our pretrained model, run [inference.ipynb](applications/inference.ipynb). You will be asked to input a protein sequence.
### Evaluation
To evaluate an existing model, you can use our [`eval` script](applications/eval.py).
E.g. `python applications/eval.py -m data/pretrained/model.pt` 
Note that the model file specified after `-m` must be a pytorch module that takes an input of size `(batch_size, 1024)` and provides an output of size `(batch_size, 1)`. 
The results will be logged under `/results/eval`.
### Train
#### Single training run
To train a model with a hyperparameter specification, you can use our [`train` script](applications/train.py).
E.g. `python3 applications/train.py --batch_size=32 --epochs=5 --learning_rate=0.001 --model=fc --model_first_hidden_units=1024 --model_hidden_layers=2 --optimizer=adam --val_on_trainset=false --model_dropoutrate=0.7`
The results will be logged under `/results/train`.
#### Hyperparameter search

1. Edit the configuration in the `training/hyperparameter_esm.yaml` if you like to
2. Then run `wandb sweep training/hyperparameter_esm.yaml`
3. You'll be prompted to run the agent by wandb. You will also be provided with a link where you can see immediate results

### Generating representations
For generating ESM representations which can then be used for training, inference and evaluation, you can use our [`generate_esm_representations` script](data_analysis_generation/generate_esm_representations.py). 
It takes a text file with protein sequences seperated by newline as input and write the per protein representation (for each protein a vector of length 1024) to the specified output directory. 
For a given text file `./sequences.txt` with contents
```
MVAFLELTSDVSQPFVIPSLSPVSQPSSRKNSDANVDDLNLAIANAALLDASASSRSHSRKNSLSLL
MHPQLEAERFHSCLDFINALDKCHQKEYYKRIFGLCNNEKDALNKCLKEASLNNKKRAVIESRIKRADVEKRWKKIEEEEYGEDAILKTILDRQYAKKKQESDNDANSK
MDNKTPVTLAKVIKVLGRTGSRGGVTQVRVEFLEDTSRTIVRNVKGPVRENDILVLMESEREARRLR
```
the command could be executed ike `python data_analysis_generation/generate_esm_representations.py ./sequences.txt data/s_s_avg` with `data/s_s_avg` being the directory the representations will be written to (we call it `s_s_avg` because the variable in the ESMFold code and paper is called `s_s` and we are storing its average along the sequence axis).

## Known Errors 
1. `GLIBCXX_3.4.30' not found' - set your LD_LIBRARY_PATH to your conda environment, i.e 
```sh
EXPORT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/anaconda3/envs/hotprot/lib
```



