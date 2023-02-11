from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
import sys
import csv
from typing import List, Union
from torch.nn.functional import pad
import h5py


""" Loads pregenerated uniprot outputs (sequence representations s_s) """
class UniProtDataset(Dataset):
    def __init__(self, dataset_filename: str = "train.csv", limit: int = 100000) -> None:
        super().__init__()

        self.limit = limit

        with open(f'data/s_s/{dataset_filename}', "r") as csv_file:
            csv_seqs = csv.reader(csv_file, delimiter=',', skipinitialspace=True)
            self.seqs = [seq for (i,(seq, thermo)) in enumerate(csv_seqs) if i!=0]
            self.seqs = set(self.seqs)

        first = True
        full_temps = {}
        with open("data/full_dataset_sequences.fasta", "r") as fasta:
            for line in fasta:
                if line[0] == '>':
                    if first:
                        first = False
                    else:
                        if entry['sequence'] in self.seqs:
                            full_temps[entry['id']] = entry['temp']
                    entry = {}
                    header_tokens = line.split(' ')
                    entry['id'] = header_tokens[0].replace('>','').split('_')[0]
                    entry['header'] = line.replace('\n', '')
                    entry['temp'] = float(header_tokens[1].split('=')[1].replace('\n',''))
                    entry['sequence'] = ''
                else:
                    entry['sequence'] = entry['sequence'] + line.replace('\n','')

        dsFilePath = os.path.join("data/uni_prot/per-protein.h5")
        if not os.path.exists(dsFilePath):
            raise Exception(f"{dsFilePath} does not exist.")
        self.dataset = []
        h5 = h5py.File(dsFilePath,'r')
        for item in h5.items():
            try:
                id = item[0]
                temp = full_temps[id]
                repr = item[1]
                entry = {
                    'representation': repr,
                    'temp': temp
                }
                self.dataset.append(entry)
            except KeyError:
                continue        

    def __len__(self):
        return min(len(self.dataset), self.limit)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return torch.from_numpy(item['representation'][()]), torch.tensor(float(item['temp']), dtype=torch.float32)

        