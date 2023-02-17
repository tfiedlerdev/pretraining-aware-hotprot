import argparse
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from thermostability.thermo_dataset import ThermostabilityDataset
from util.telegram import TelegramBot
import csv
import pickle
from datetime import datetime
from esm_custom import esm

class SequencesDataset(Dataset):
    def __init__(self, sequences: "set[str]") -> None:
        super().__init__()
        self.sequences = list(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def generate_representations(dir_path, sequences):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    esmfold = esm.pretrained.esmfold_v1().to(device)

    ds = SequencesDataset(sequences)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    timeStart = time.time()
    labels_file = os.path.join(dir_path, "sequences.csv")
    if not os.path.exists(labels_file):
        with open(labels_file,"w") as csv:
            csv.write(f"sequence, filename\n") 

    maxFilePrefix = len(os.listdir(dir_path))
    print(f"Starting with maxFilePrefix {maxFilePrefix}")
    batchesPredicted = 0

    for index, (inputs) in enumerate(loader):
        batch_size = len(inputs)
        numBatches = int(len(sequences) / batch_size)
        print(f"At batch {index}/{numBatches}")
        with torch.no_grad():
            print(f"Predicting")
            esm_output = esmfold.infer(sequences=inputs)
            s_s = esm_output["s_s"]
            batchesPredicted +=1
            with open(labels_file,"a") as csv:
                for s, data in enumerate(s_s):
                    maxFilePrefix+=1
                    file = str(maxFilePrefix)+".pt"
                    if not os.path.exists(file):
                        with open(os.path.join("data/s_s", file), "wb") as f:
                            torch.save(data.mean(0).cpu(),f)
                        csv.write(f"{inputs[s]}, {file}\n") 
        if index %5 == 0:
            secsSpent = time.time()- timeStart  
            secsToGo = (secsSpent/(batchesPredicted+1))*(numBatches-index-1)
            hoursToGo = secsToGo/(60*60)
            now = datetime.now()
            print(f"Done with {index}/{numBatches} batches (hours to go: {int(hoursToGo)}) [last update: {now.hour}:{now.minute}]")

def get_already_created_sequences(dir_path):
    labels_file = os.path.join(dir_path, "sequences.csv")
    if not os.path.exists(labels_file):
        with open(labels_file,"r") as f:
            return [seq for i,(seq, _) in enumerate(csv.reader(f,delimiter=",", skipinitialspace=True)) if i!=0]
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, nargs=1, help="File containing amino ascid sequences split by new line for which to generate representations")
    parser.add_argument("output_dir", type=str, nargs=1, help="Directory in which to place the representations")
    parser.add_argument("--batch_size", type=int,default=2)

    args = vars(parser.parse_args())
    with open(args["input_file"],"r") as f:
        seqs = f.readlines()

    print(f"Representations of {len(seqs)} sequences to be created")
    already_created_seqs = get_already_created_sequences(args["output_dir"])
    print(f"Representations of {len(already_created_seqs)} sequences of those already created")
    remaining_seqs = set(seqs).difference_update(already_created_seqs)
    print(f"Creating remaining representations of {len(remaining_seqs)} sequences")
    generate_representations(remaining_seqs)


