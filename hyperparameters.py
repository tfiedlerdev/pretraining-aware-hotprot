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
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from tqdm.notebook import tqdm
import sys
from thermostability.thermo_pregenerated_dataset import zero_padding, zero_padding700
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


def train_model(
    model, optimizer, criterion, scheduler, dataloaders, dataset_sizes, num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_epoch_loss = sys.float_info.max
    losses = []
    batchEnumeration = []
    allPredictions = torch.tensor([])
    allLabels = torch.tensor([])
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, torch.unsqueeze(labels, 1))

                    if phase == "val":
                        allPredictions = torch.cat(
                            (allPredictions, torch.squeeze(outputs.cpu()))
                        )
                        allLabels = torch.cat((allLabels, torch.squeeze(labels.cpu())))
                    # backward + optimize only if in training phase
                    if phase == "train":

                        if not torch.isnan(loss):
                            loss.backward()
                            threshold = 10
                            for p in model.parameters():
                                if p.grad != None:
                                    if p.grad.norm() > threshold:
                                        torch.nn.utils.clip_grad_norm_(p, threshold)
                            optimizer.step()
                        if torch.isnan(loss).any():
                            print(
                                f"Nan loss: {torch.isnan(loss)}| Loss: {loss}| inputs: {inputs}"
                            )
                # statistics
                batch_size = inputs.size(0)
                batch_loss = loss.item() * batch_size
                losses.append(batch_loss)
                batchEnumeration.append(
                    batchEnumeration[-1] + 1 if len(batchEnumeration) > 0 else 0
                )

                running_loss += batch_loss
                mean_abs_diff = torch.abs(outputs.squeeze().sub(labels.squeeze())).squeeze().mean().item()
                wandb.log({"mean_abs_diff": mean_abs_diff})
                if idx % 1 == 0:
                    tqdm.write(
                        "Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}, batch abs diff mean {:.6f}".format(
                            epoch,
                            num_epochs,
                            idx + 1,
                            len(dataloaders[phase]),
                            batch_loss / float(batch_size),
                            mean_abs_diff,
                        ),
                        end="\r",
                    )

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f}")

            # deep copy the model
            if phase == "val" and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                wandb.log({"mse_loss": epoch_loss})
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_epoch_loss:4f}")
    # load best model weights
    model.load_state_dict(best_model_wts)
    artifact = wandb.Artifact("results", type="train")
    pl.scatter(
        allPredictions.squeeze().tolist()[-dataset_sizes["val"]:],
        allLabels.squeeze().tolist()[-dataset_sizes["val"]:],
    )
    plotPath = f"results/predictions.png"
    pl.xlabel("Predictions")
    pl.ylabel("Labels")
    pl.savefig(plotPath)
    artifact.add_file(plotPath)
    wandb.log_artifact(artifact)
    return model, best_epoch_loss


def run_train_experiment(config: dict = None):
    with wandb.init(config=config):
        train_ds = ThermostabilityPregeneratedDataset(
            "train.csv", limit=config["dataset_limit"]
        )
        eval_ds = ThermostabilityPregeneratedDataset(
            "train.csv" if config["val_on_trainset"] else "val.csv",
            limit=config["dataset_limit"],
        )

        dataloaders = {
            "train": DataLoader(
                train_ds,
                batch_size=2,
                shuffle=True,
                num_workers=4,
                collate_fn=zero_padding700,
            ),
            "val": DataLoader(
                eval_ds,
                batch_size=2,
                shuffle=True,
                num_workers=4,
                collate_fn=zero_padding700,
            ),
        }

        dataset_sizes = {"train": len(train_ds), "val": len(eval_ds)}
        config = wandb.config
        model = HotInferPregeneratedFC(
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
        model.to(device)
        wandb.watch(model)
        criterion = nn.MSELoss()

        optimizer_ft = (
            torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            if config["optimizer"] == "adam"
            else torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum = 0.9)
        )

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.9)

        model, score = train_model(
            model,
            optimizer_ft,
            criterion,
            exp_lr_scheduler,
            dataloaders,
            dataset_sizes,
            num_epochs=config["epochs"],
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
    args = parser.parse_args()

    run_train_experiment(config=vars(args))
