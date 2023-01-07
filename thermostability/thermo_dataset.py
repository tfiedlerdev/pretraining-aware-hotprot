from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
from IPython.display import clear_output, display

class ThermostabilityDataset(Dataset):
    def __init__(self, file_path: str, use_cache: bool = True, max_seq_len=-1, max_ds_len=-1) -> None:
        super().__init__()
        sequences = []
        melting_points = []

        cache_file = file_path+"_cache.p"

        if use_cache and os.path.exists(cache_file):
            print("Loading data from cache file: ", cache_file)
            with open( cache_file, "rb" ) as f:
                self.thermo_dataframe = pickle.load(f)
        else:     
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
            
            data = {'x': sequences, 'y': melting_points}

            self.thermo_dataframe = pd.DataFrame(data)
            if use_cache:
                with open( cache_file, "wb" ) as f:
                    pickle.dump(self.thermo_dataframe, f)

        if max_seq_len > 0: 
            mask = self.thermo_dataframe.apply(lambda row: len(row['x']) <= max_seq_len, axis=1)
            self.thermo_dataframe = self.thermo_dataframe[mask]
        
        self.max_ds_len = max_ds_len


    def __len__(self):
        return len(self.thermo_dataframe) if self.max_ds_len > len(self.thermo_dataframe) else self.max_ds_len

    def __getitem__(self, index):
        if(torch.is_tensor(index)):
            index = index.tolist()

        items = self.thermo_dataframe.iloc[index, :]
        

        return items["x"], torch.tensor([items["y"]], dtype=torch.float32)

        