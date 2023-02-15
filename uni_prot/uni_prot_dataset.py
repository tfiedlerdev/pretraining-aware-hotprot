from torch.utils.data import Dataset
import torch
import os
import pickle
import csv
import h5py


""" Loads pregenerated uniprot outputs (sequence representations s_s) """


class UniProtDataset(Dataset):
    def __init__(
        self, dataset_filename: str = "train.csv", limit: int = 10000000
    ) -> None:
        super().__init__()
        self.cacheFile = f"data/uni_prot/cache_{dataset_filename.split('.')[0]}"
        self.limit = limit

        if not os.path.exists(self.cacheFile):

            with open(f"data/s_s/{dataset_filename}", "r") as csv_file:
                csv_seqs = csv.reader(csv_file, delimiter=",", skipinitialspace=True)
                self.seqs = [
                    seq for (i, (seq, thermo)) in enumerate(csv_seqs) if i != 0
                ]
                print(f"Seqs in {dataset_filename}: {len(self.seqs)} ")
                self.seqs = set(self.seqs)

            first = True
            full_temps = {}
            with open("data/full_dataset_sequences.fasta", "r") as fasta:
                for line in fasta:
                    if line[0] == ">":
                        if first:
                            first = False
                        else:
                            if entry["sequence"] in self.seqs:
                                if entry["id"] not in full_temps.keys():
                                    full_temps[entry["id"]] = [entry["temp"]]
                                else:
                                    full_temps[entry["id"]].append(entry["temp"])
                        entry = {}
                        header_tokens = line.split(" ")
                        entry["id"] = header_tokens[0].replace(">", "").split("_")[0]
                        entry["header"] = line.replace("\n", "")
                        entry["temp"] = float(
                            header_tokens[1].split("=")[1].replace("\n", "")
                        )
                        entry["sequence"] = ""
                    else:
                        entry["sequence"] = entry["sequence"] + line.replace("\n", "")

            dsFilePath = os.path.join("data/uni_prot/per-protein.h5")
            if not os.path.exists(dsFilePath):
                raise Exception(f"{dsFilePath} does not exist.")
            self.dataset = []
            h5 = h5py.File(dsFilePath, "r")
            for item in h5.items():
                try:
                    id = item[0]
                    temps = full_temps[id]
                    repr = torch.from_numpy(item[1][()])
                    for temp in temps:
                        entry = ( repr,
                           torch.tensor(float(temp), dtype=torch.float32)
                        )
                        self.dataset.append(entry)
                except KeyError:
                    continue
            with open(self.cacheFile, "wb") as f:
                pickle.dump(self.dataset, f)
                print(len(self.dataset))
        else:
            with open(self.cacheFile, "rb") as f:
                self.dataset = pickle.load(f)
                print(len(self.dataset))

    def __len__(self):
        return min(len(self.dataset), self.limit)

    def __getitem__(self, index):
        return self.dataset[index]
