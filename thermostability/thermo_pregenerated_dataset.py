from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
from IPython.display import clear_output, display
import sys

""" Loads pregenerated esmfold outputs (sequence representations s_s) """
class ThermostabilityPregeneratedDataset(Dataset):
    def __init__(self, dir_path: str) -> None:
        super().__init__()
        self.dir_path = dir_path
        labelsFilePath = os.path.join(dir_path, "labels.csv")
        if not os.path.exists(labelsFilePath):
            raise Exception(f"{labelsFilePath} does not exist.")

        with open('labels.csv', newline='\n') as csvfile:
            spamreader = csvfile.reader(csvfile, delimiter=',')
            self.filename_thermo_seq = [row for row in spamreader]

        print(self.filename_thermo_seq[:10])


    def __len__(self):
        return len(self.filename_thermo_seq)

    def __getitem__(self, index):
        filename, thermo, seq = self.filename_thermo_seq[index]
        
        with open(os.path.join(self.dir_path, filename)) as f:
            s_s = pickle.load(f) 

        return s_s, thermo

        