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
import torch
from torch.utils.data import DataLoader
import copy
from thermostability.thermo_dataset import ThermostabilityDataset
from util.telegram import TelegramBot
from tqdm.notebook import tqdm
import sys
from util.train import train_model
from torch import nn
from uni_prot.dense_model import DenseModel
from datetime import datetime


from uni_prot.uni_prot_dataset import UniProtDataset


def main():
    cudnn.benchmark = True  
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        
    torch.cuda.list_gpu_processes()
    telegramBot = TelegramBot()

    train_ds = UniProtDataset("train.csv")
    val_ds = UniProtDataset("val.csv")

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2),
        "val": DataLoader(val_ds, batch_size=2, shuffle=True, num_workers=2)
    }

    dataset_sizes = {"train": len(train_ds),"val": len(val_ds)}
        
    model = DenseModel()
    model.to("cuda:0")

    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model, _ = train_model(model, criterion, exp_lr_scheduler, dataloaders=dataloaders, dataset_sizes=dataset_sizes, use_wandb=False,
                        num_epochs=100, prepare_labels = lambda x: x.to("cuda:0"),prepare_inputs= lambda x: x.to("cuda:0") )

    if not telegramBot.enabled:
       raise e
    telegramBot.send_telegram(f"Training failed with error message: {str(e)}") 
    def predictDiffs(set="val"):
        with torch.no_grad():
            n = len(dataloaders[set])
            diffs = torch.tensor([])
            for index, (inputs, labels) in enumerate(dataloaders[set]):
                inputs = inputs.to("cuda:0")
                print(f"Infering thermostability for sample {index}/{n}...")
                labels = labels.to("cuda:0")
                outputs = model(inputs)

                _diffs = outputs.squeeze().sub(labels.squeeze()).cpu()
                diffs = torch.cat((diffs, _diffs))
                print("Diff: ", _diffs)
        return diffs

    diffs = predictDiffs()

    #diffs = np.array([0, 0.1, 0.2,-0.2, -0.8, 0.1])
    plt.title("Differences predicted <-> actual thermostability")
    plt.hist(diffs, 50,histtype="step")
    resultsDir = "results"
    now = datetime.now()
    time = now.strftime("%d_%m_%Y_%H:%M:%S")
    os.makedirs(resultsDir, exist_ok=True)
    histFile = f"results/{time}_diffs.png"
    plt.savefig(histFile)
    telegramBot.send_photo(histFile, f"Differences predicted <-> actual thermostability at {time}")

    try: 
        modelPath = os.path.join(resultsDir, f"{time}_model.pth")
        torch.save(model, modelPath)
        telegramBot.send_telegram(f"Model saved at {modelPath}")
    except Exception as e:
        telegramBot.send_telegram(f"Saving model failed for reason: {str(e)}")


if __name__ == "__main__":
    main()