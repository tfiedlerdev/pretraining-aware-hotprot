from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
import sys
import csv

""" Loads pregenerated esmfold outputs (sequence representations s_s) """
class ThermostabilityPregeneratedDataset(Dataset):
    def __init__(self, dir_path: str) -> None:
        super().__init__()
        self.dir_path = dir_path
        labelsFilePath = os.path.join(dir_path, "labels.csv")
        if not os.path.exists(labelsFilePath):
            raise Exception(f"{labelsFilePath} does not exist.")


        with open(labelsFilePath, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            self.filename_thermo_seq = [row for (i, row) in enumerate(spamreader) if i!=0]


    def __len__(self):
        return len(self.filename_thermo_seq)

    def __getitem__(self, index):
        filename, thermo, seq = self.filename_thermo_seq[index]
        
        with open(os.path.join(self.dir_path, filename), "rb") as f:
            s_s = pickle.load(f) 

        return s_s, torch.Tensor(float(thermo))

        