import torch
import time
import copy
from torch import nn as nn
from tqdm.notebook import tqdm
import sys
import wandb
import pylab as pl
from typing import Callable
from util.telegram import TelegramBot
import seaborn as sns
import matplotlib.pyplot as plt
from util.plotting import plot_predictions


def execute_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader: torch.utils.data.DataLoader,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    on_batch_done: Callable[
        [int, torch.Tensor, float, float], None
    ] = lambda idx, outputs, loss, running_mad: None,
    optimizer: torch.optim.Optimizer=None,
):

    epoch_predictions = torch.tensor([])
    epoch_actuals = torch.tensor([])
    running_loss = 0.0
    epoch_mad = 0.0
    # Iterate over data.
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = prepare_inputs(inputs)
        labels = prepare_labels(labels)
        # zero the parameter gradients
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.unsqueeze(labels, 1))
        epoch_predictions = torch.cat((epoch_predictions, outputs.cpu()))
        epoch_actuals = torch.cat((epoch_actuals, labels.cpu()))
        # statistics
        batch_size = len(inputs)
        batch_loss = loss.item() * batch_size

        running_loss += batch_loss
        mean_abs_diff = (
            torch.abs(outputs.squeeze().sub(labels.squeeze())).squeeze().mean().item()
        )
        epoch_mad += mean_abs_diff
        running_mad = epoch_mad / (idx + 1)
        on_batch_done(idx, outputs, loss / float(batch_size), running_mad)

    epoch_mad = epoch_mad / len(dataloader)
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, epoch_mad, epoch_actuals, epoch_predictions


def train_model(
    model,
    criterion: nn.modules.loss._Loss,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloaders,
    dataset_sizes,
    use_wandb,
    num_epochs=25,
    return_best_model=True,
    max_gradient_clip: float = 10,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    label="",
):
    optimizer = scheduler.optimizer
    since = time.time()
    best_model_path = f"results/{label}_best_model.pt"
    if return_best_model:
        torch.save(model, best_model_path)
    best_epoch_mad = sys.float_info.max
    best_epoch_loss = sys.float_info.max
    bestEpochPredictions = torch.tensor([])
    bestEpochLabels = torch.tensor([])

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            def on_batch_done(idx, outputs, loss, running_mad):
                if phase == "train":
                    if not torch.isnan(loss):
                        loss.backward()
                        if max_gradient_clip:
                            threshold = max_gradient_clip
                            for p in model.parameters():
                                if p.grad != None:
                                    if p.grad.norm() > threshold:
                                        torch.nn.utils.clip_grad_norm_(p, threshold)
                        optimizer.step()
                    if torch.isnan(loss).any():
                        print(
                            f"Nan loss: {torch.isnan(loss)}| Loss: {loss}| inputs: {inputs}"
                        )
                tqdm.write(
                    "Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}, epoch abs diff mean {:.6f}".format(
                        epoch,
                        num_epochs,
                        idx + 1,
                        len(dataloaders[phase]),
                        loss,
                        running_mad,
                    ),
                    end="\r",
                )

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            with torch.set_grad_enabled(phase == "train"):
                epoch_loss, epoch_mad, epoch_actuals, epoch_predictions = execute_epoch(
                    model,
                    criterion,
                    dataloaders[phase],
                    prepare_inputs,
                    prepare_labels,
                    on_batch_done=on_batch_done,
                    optimizer=optimizer
                )

            if epoch_mad < best_epoch_mad:
                best_epoch_mad = epoch_mad
            if use_wandb:
                wandb.log(
                    {
                        f"epoch_mad_{phase}": epoch_mad,
                    }
                )
            if phase == "train":
                scheduler.step()

            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "val":
                if use_wandb:
                    wandb.log({"mse_loss": epoch_loss})
                if epoch_loss < best_epoch_loss:
                    best_epoch_loss = epoch_loss
                    if return_best_model:
                        torch.save(model, best_model_path)

                    bestEpochLabels = epoch_actuals
                    bestEpochPredictions = epoch_predictions
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_epoch_loss:4f}")

    # load best model weights
    if return_best_model:
        model = torch.load(best_model_path)
    preds = bestEpochPredictions.squeeze().tolist()
    actuals = bestEpochLabels.squeeze().tolist()
    train_size = dataset_sizes["train"]
    val_size = dataset_sizes["val"]
    plot_predictions(
        f"predictions_{label}_epochs{num_epochs}_gradClip{max_gradient_clip}_trainSize{train_size}_valSize{val_size}",
        f"Loss: {best_epoch_loss: .2f}, {label}, mad {best_epoch_mad: .2f}",
        preds,
        actuals,
        use_wandb,
    )
    return model, best_epoch_loss
