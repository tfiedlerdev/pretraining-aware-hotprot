import torch
from torch.utils.data import DataLoader
import optuna
import time
import copy
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from pathlib import Path
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
)
from thermostability.hotinfer import HotInferModelParallel
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from thermostability.cnn_pregenerated import CNNPregeneratedFC, CNNPregenerated
from tqdm.notebook import tqdm
import sys
from thermostability.thermo_dataset import ThermostabilityDataset
from thermostability.thermo_pregenerated_dataset import zero_padding_collate, zero_padding700_collate
import wandb
import argparse
import pylab as pl

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.cuda.empty_cache()
torch.cuda.list_gpu_processes()
from util.train import train_model


def run_train_experiment(config: dict = None, use_wandb = True):
    train_ds = ThermostabilityDataset(
        "train.csv", limit=config["dataset_limit"]
    )
    eval_ds = ThermostabilityDataset(
        "train.csv" if config["val_on_trainset"] else "val.csv",
        limit=config["dataset_limit"],
    )
    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            #collate_fn=zero_padding700_collate,
        ),
        "val": DataLoader(
            eval_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            #collate_fn=zero_padding700_collate,
        ),
    }
    dataset_sizes = {"train": len(train_ds), "val": len(eval_ds)}
    input_sizes = {
        "esm_s_B_avg": 2560
    }
    representation_key = config["representation_key"]
    input_size = input_sizes[representation_key]
    thermo = (
        HotInferPregeneratedFC(
            input_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
        if config["model"] == "fc"
        else CNNPregeneratedFC(
            input_seq_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
    )
    
    model = HotInferModelParallel(representation_key, thermo_module=thermo)

    if use_wandb:
        wandb.watch(thermo)
    criterion = nn.MSELoss()
    optimizer_ft = (
        torch.optim.Adam(thermo.parameters(), lr=config["learning_rate"])
        if config["optimizer"] == "adam"
        else torch.optim.SGD(
            thermo.parameters(), lr=config["learning_rate"], momentum=0.9
        )
    )
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)
    thermo, score = train_model(
        model,
        criterion,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        use_wandb,
        num_epochs=config["epochs"],
        prepare_labels=lambda x: x.to("cuda:1")
    )
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--model_hidden_layers", type=int, required=True)
    parser.add_argument("--model_first_hidden_units", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--val_on_trainset", type=bool)
    parser.add_argument("--dataset_limit", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--representation_key", type=str)
    parser.add_argument("--no_wandb", action='store_true')
    args = parser.parse_args()

    argsDict = vars(args)
    use_wandb = not argsDict["no_wandb"]
    del argsDict["no_wandb"]
    if use_wandb:
        with wandb.init(config=argsDict):
            run_train_experiment(config=wandb.config, use_wandb=True)
    else: 
        run_train_experiment(config=argsDict, use_wandb=False)