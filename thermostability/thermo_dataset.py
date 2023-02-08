from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
from IPython.display import clear_output, display
import sys

class ThermostabilityDataset(Dataset):
    def __init__(self, file_path: str, use_cache: bool = True, max_seq_len=-1, max_ds_len=-1, once_occuring_seq_only=False) -> None:
        super().__init__()
        

        cache_file = file_path+"_v2_cache.p"

        if use_cache and os.path.exists(cache_file):
            print("Loading data from cache file: ", cache_file)
            with open( cache_file, "rb" ) as f:
                self.sequence_thermo = pickle.load(f)
        else:     
            sequences = []
            melting_points = []
            with open(file_path) as file:
                for index, line in enumerate(file):
                    if index%50 == 0:
                        clear_output(wait=True)
                        print(f"Reading line {index}", flush=True)
                    if line.startswith(">"):
                        line_tokens = line.split(' ')
                        id = line_tokens[0]
                        melting_point = float(line_tokens[1].split('=')[1])
                        melting_points.append(melting_point)
                    else: 
                        sequence = line.replace("\n", "")
                        sequences.append(sequence)
            
            self.sequence_thermo =[(sequences[i], melting_points[i]) for i in range(len(sequences))]
    
            if use_cache:
                with open( cache_file, "wb" ) as f:
                    pickle.dump(self.sequence_thermo, f)

        if max_seq_len > 0: 
            self.sequence_thermo = list(filter(lambda seq_t: len(seq_t[0])<=max_seq_len, self.sequence_thermo))

        
        if once_occuring_seq_only:
            uniqueSeqs = set([seq for seq, t in self.sequence_thermo])
            nums = dict.fromkeys(uniqueSeqs, 0)

            for seq,_ in self.sequence_thermo:
                nums[seq] +=1
            self.sequence_thermo = list(filter(lambda seq_t: nums[seq_t[0]] == 1, self.sequence_thermo))

        self.max_ds_len = max_ds_len if max_ds_len != -1 else sys.maxsize


    def __len__(self):
        return len(self.sequence_thermo) if self.max_ds_len > len(self.sequence_thermo) else self.max_ds_len

    def __getitem__(self, index):
        if(torch.is_tensor(index)):
            index = index.tolist()
        seq, temp = self.sequence_thermo[index]
    
        return seq, torch.tensor([temp], dtype=torch.float32)

        