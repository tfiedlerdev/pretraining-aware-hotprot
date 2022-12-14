from torch.utils.data import Dataset
import pandas as pd
import torch

class ThermostabilityDataset(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        sequences = []
        melting_points = []
        with open(file_path) as file:
            for line in file:
                line_tokens = line.split(' ')
                id = line_tokens[0]
                melting_point = float(line_tokens[1].split('=')[1])
                sequence = line_tokens[2]

                melting_points.append(melting_point)
                sequences.append(sequence)
            
        data = {'x': sequences, 'y': melting_points}

        self.thermo_dataframe = pd.DataFrame(data)

    def __len__(self):
        return len(self.thermo_dataframe)

    def __getitem__(self, index):
        if(torch.is_tensor(index)):
            index = index.tolist()

        items = self.thermo_dataframe.iloc[index, :]
        return items

        