from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
from IPython.display import clear_output, display

class ThermostabilityDataset(Dataset):
    def __init__(self, file_path: str, use_cache: bool = True) -> None:
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

    def __len__(self):
        return 20#len(self.thermo_dataframe)

    def __getitem__(self, index):
        if(torch.is_tensor(index)):
            index = index.tolist()

        items = self.thermo_dataframe.iloc[index, :]
        

        return items["x"], torch.tensor([items["y"]], dtype=torch.float32)

        