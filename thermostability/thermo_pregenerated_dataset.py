from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
import sys
import csv
from typing import List, Union
from torch.nn.functional import pad

def zero_padding(s_s_list: "list[tuple[torch.Tensor, torch.Tensor]]", fixed_size: Union[int, None]=None):
    max_size = fixed_size if fixed_size else max([s_s.size(0) for s_s, _ in s_s_list])

    padded_s_s = []
    temps =[]
    for s_s, temp in s_s_list:
        dif = max_size - s_s.size(0) 
        padded = pad(s_s, (0,0,dif,0), "constant", 0)
        padded_s_s.append(padded)
        temps.append(temp)
    results= torch.stack(padded_s_s, 0).unsqueeze(1), torch.stack(temps)
    return results

def zero_padding700(s_s_list: "list[tuple[torch.Tensor, torch.Tensor]]"):
    return zero_padding(s_s_list, 700)

""" Loads pregenerated esmfold outputs (sequence representations s_s) """
class ThermostabilityPregeneratedDataset(Dataset):
    def __init__(self, dataset_filename: str = "train.csv", limit: int = 100000) -> None:
        super().__init__()

        dsFilePath = os.path.join("data/s_s/", dataset_filename)
        if not os.path.exists(dsFilePath):
            raise Exception(f"{dsFilePath} does not exist.")

        with open("data/s_s/sequences.csv", newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            self.sequenceToFilename = {sequence: filename for (i, (sequence, filename)) in enumerate(spamreader) if i!=0}

        self.limit=limit
        with open(dsFilePath, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            seq_thermos = [(seq,thermo) for (i,(seq, thermo)) in enumerate(spamreader) if i!=0]
        
            self.filename_thermo_seq = [(self.sequenceToFilename[seq], thermo, seq) for (seq, thermo) in seq_thermos if seq in self.sequenceToFilename]
            diff = len(seq_thermos)-len(self.filename_thermo_seq)  
            print(f"Omitted {diff} sequences of {dataset_filename} because they have not been pregenerated")
        self.sequences_dir = "data/s_s"

    def __len__(self):
        return min(len(self.filename_thermo_seq), self.limit)
    
    def __getitem__(self, index):
        filename, thermo, seq = self.filename_thermo_seq[index]
        
        with open(os.path.join(self.sequences_dir, filename), "rb") as f:
            s_s = torch.load(f) 

        return s_s, torch.tensor(float(thermo), dtype=torch.float32)

        