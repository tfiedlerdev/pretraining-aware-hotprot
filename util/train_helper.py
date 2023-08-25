import torch
import time
from torch import nn as nn
from tqdm.notebook import tqdm
import numpy as np
import sys
import wandb
from typing import Callable, get_args, Tuple, List
from scipy.stats import spearmanr
import pandas as pd
from typing_extensions import TypedDict, Literal
from torch.utils.data import Dataset
from pynvml.smi import nvidia_smi
from util.yaml_config import YamlConfig
from thermostability.fst_dataset import FSTDataset, zero_padding_fst
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
    k_max_sum_collate,
    k_max_var_collate,
    zero_padding700_collate,
)
from thermostability.thermo_dataset import ThermostabilityDataset
import psutil
import os


def log_gpu_memory(device: int):
    memory_stats = nvidia_smi.getInstance().DeviceQuery(
        "memory.free, memory.total, memory.used"
    )
    total_memory = memory_stats["gpu"][device]["fb_memory_usage"]["total"]
    free_memory = memory_stats["gpu"][device]["fb_memory_usage"]["free"]
    used_memory = memory_stats["gpu"][device]["fb_memory_usage"]["used"]
    unit = memory_stats["gpu"][device]["fb_memory_usage"]["unit"]
    return total_memory, free_memory, unit


def log_memory():
    process = psutil.Process(os.getpid())
    mem = dict(process.memory_info()._asdict())
    rss = mem["rss"] / 1000**3
    vms = mem["vms"] / 1000**3
    # returns used memory of the current process in GB
    return rss, vms, "GB"


def metrics_per_temp_range(min_temp, max_temp, epoch_predictions, epoch_actuals):
    subset_predictions = []
    subset_actuals = []

    for pred, actual in zip(epoch_predictions, epoch_actuals):
        if min_temp <= actual and actual < max_temp:
            subset_predictions.append(pred)
            subset_actuals.append(actual)

    diffs = np.array(
        [abs(pred - actual) for pred, actual in zip(subset_predictions, subset_actuals)]
    )
    return f"{min_temp}-{max_temp}", diffs, subset_predictions, subset_actuals


DatasetNamesLiteral = Literal["pregenerated", "end_to_end", "fst"]
DatasetNames: List[DatasetNamesLiteral] = get_args(DatasetNamesLiteral)
DatasetSplitsLiteral = Literal["ours", "ours_median", "flip"]
DatasetSplits: List[DatasetNamesLiteral] = get_args(DatasetNamesLiteral)


def get_dataset(
    dataset: DatasetNamesLiteral,
    split: DatasetSplitsLiteral,
    yaml_config: YamlConfig,
    phase: Literal["train", "val", "test"],
    limit: int,
    representation_key: str,
    max_seq_len: int = 700,
) -> Dataset:
    splits_dir_path = yaml_config["DatasetSplitsPath"]
    split_files = {
        "ours": {
            "train": os.path.join(splits_dir_path, "train.csv"),
            "val": os.path.join(splits_dir_path, "val.csv"),
            "test": os.path.join(splits_dir_path, "test.csv"),
        },
        "ours_median": {
            "train": os.path.join(splits_dir_path, "train_median.csv"),
            "val": os.path.join(splits_dir_path, "val_median.csv"),
            "test": os.path.join(splits_dir_path, "test_median.csv"),
        },
        "flip": {
            "train": os.path.join(splits_dir_path, "train_FLIP.csv"),
            "val": os.path.join(splits_dir_path, "val_FLIP.csv"),
            "test": os.path.join(splits_dir_path, "test_FLIP.csv"),
        },
    }
    file_name = split_files[split][phase]

    if dataset == "fst":
        return FSTDataset(
            file_name,
            limit,
            max_seq_len,
            yaml_config["DatasetPath"],
            representation_key,
        )
    elif dataset == "pregenerated":
        return ThermostabilityPregeneratedDataset(
            file_name,
            limit,
            max_seq_len,
            yaml_config["DatasetPath"],
            representation_key,
        )
    elif dataset == "end_to_end":
        return ThermostabilityDataset(file_name, limit, max_seq_len)
    raise Exception(f"Unknown dataset {dataset}. Must be one of {DatasetNames}")


def get_collate_fn(config: dict, representation_key: str):
    collate_fn = None
    if representation_key == "s_s":
        collate_fn_key = config["collate_fn"]
        if collate_fn_key == "pad700":
            collate_fn = zero_padding700_collate
        else:
            k = config["collate_k"]
            if k == None:
                raise Exception(
                    f"For the selected collate function ({collate_fn_key}), you need to defined collate_k"
                )
            if collate_fn_key == "k_max_sum_pooling":
                collate_fn = k_max_sum_collate(k)
            elif collate_fn_key == "k_max_var_pooling":
                collate_fn = k_max_var_collate(k)
    elif config["dataset"] == "fst":
        collate_fn = zero_padding_fst

    return collate_fn


def evaluate_temp_bins(predictions, labels, bin_width, key: str):
    np_preds = np.array(predictions)
    np_actuals = np.array(labels)

    metrics_per_class = {}

    i = 0
    while i * bin_width < np_actuals.max():
        if (i + 1) * bin_width >= np_actuals.min():
            class_label, diffs, subset_pred, subset_target = metrics_per_temp_range(
                i * bin_width,
                (i + 1) * bin_width,
                np_preds,
                np_actuals,
            )
            metrics_per_class[
                f"{class_label}_best_epoch_spearman_r_s_{key}"
            ] = spearmanr(subset_pred, subset_target).correlation
            metrics_per_class[
                f"{class_label}_best_epoch_max_abs_diff_{key}"
            ] = diffs.max()
            metrics_per_class[
                f"{class_label}_best_epoch_median_abs_diff_{key}"
            ] = np.median(diffs)
            metrics_per_class[
                f"{class_label}_best_epoch_mean_abs_diff_{key}"
            ] = diffs.mean()

        i = i + 1

    return metrics_per_class


def calculate_metrics(predictions, labels, key: str, temp_bin_width: int = 20):
    diffs = pd.Series([abs(pred - labels[i]) for (i, pred) in enumerate(predictions)])
    metrics = evaluate_temp_bins(predictions, labels, temp_bin_width, key)
    metrics[f"best_epoch_spearman_r_s_{key}"] = spearmanr(
        predictions, labels, nan_policy="raise"
    ).correlation
    metrics[f"best_epoch_max_abs_diff_{key}"] = diffs.max()
    metrics[f"best_epoch_median_abs_diff_{key}"] = diffs.median()
    metrics[f"best_epoch_mean_abs_diff_{key}"] = diffs.mean()
    return metrics


def execute_epoch_fst(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader: torch.utils.data.DataLoader,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    on_batch_done: Callable[
        [int, torch.Tensor, float, float], None
    ] = lambda idx, outputs, loss, running_mad: None,
    optimizer: torch.optim.Optimizer = None,
):
    epoch_predictions = torch.tensor([])
    epoch_actuals = torch.tensor([])
    running_loss = 0.0
    epoch_mad = 0.0
    # Iterate over data.
    for idx, (seqs, (inputs, labels)) in enumerate(dataloader):
        inputs = prepare_inputs(inputs)
        labels = prepare_labels(labels)
        # zero the parameter gradients
        if optimizer:
            optimizer.zero_grad()
        outputs = model((seqs, inputs))
        loss = criterion(outputs, torch.unsqueeze(labels, 1))
        epoch_predictions = torch.cat((epoch_predictions, outputs.cpu()))
        epoch_actuals = torch.cat((epoch_actuals, labels.cpu()))
        # statistics
        batch_loss = loss.item()

        running_loss += batch_loss
        mean_abs_diff = (
            torch.abs(outputs.squeeze().sub(labels.squeeze())).squeeze().mean().item()
        )
        epoch_mad += mean_abs_diff
        running_mad = epoch_mad / (idx + 1)
        on_batch_done(idx, outputs, loss, running_mad)

    epoch_mad = epoch_mad / len(dataloader)
    epoch_loss = running_loss / len(dataloader)
    return (
        epoch_loss,
        epoch_mad,
        epoch_actuals.squeeze().tolist(),
        epoch_predictions.squeeze().tolist(),
    )


def execute_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader: torch.utils.data.DataLoader,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    on_batch_done: Callable[
        [int, torch.Tensor, float, float], None
    ] = lambda idx, outputs, loss, running_mad: None,
    optimizer: torch.optim.Optimizer = None,
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

        loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))

        epoch_predictions = torch.cat((epoch_predictions, outputs.cpu()))
        epoch_actuals = torch.cat((epoch_actuals, labels.cpu()))
        # statistics
        batch_loss = loss.item()
        running_loss += batch_loss
        mean_abs_diff = (
            torch.abs(outputs.squeeze().sub(labels.squeeze())).squeeze().mean().item()
        )
        epoch_mad += mean_abs_diff
        running_mad = epoch_mad / (idx + 1)
        on_batch_done(idx, outputs, loss, running_mad)

    epoch_mad = epoch_mad / len(dataloader)
    epoch_loss = running_loss / len(dataloader)
    return (
        epoch_loss,
        epoch_mad,
        epoch_actuals.squeeze().tolist(),
        epoch_predictions.squeeze().tolist(),
    )


class TrainResponse(TypedDict):
    model: nn.Module
    best_epoch_loss: float
    best_val_mad: float
    epoch_mads: 'dict[Literal["train", "val"], "list[float]"]'
    best_epoch_actuals: "list[float]"
    best_epoch_predictions: "list[float]"
    test_loss: float
    test_mad: float
    test_actuals: "list[float]"
    test_predictions: "list[float]"


def train_model(
    model,
    criterions,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloaders,
    use_wandb,
    num_epochs=25,
    best_model_path: str = None,
    max_gradient_clip: float = None,
    epoch_function: Callable = execute_epoch,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    should_stop: Callable[["list[float]"], bool] = lambda epoch_val_losses: False,
) -> TrainResponse:
    optimizer = scheduler.optimizer
    since = time.time()

    if best_model_path:
        torch.save(model, best_model_path)
    best_val_mad = sys.float_info.max
    epoch_losses = {"train": [], "val": []}
    best_epoch_loss = sys.float_info.max
    best_epoch_predictions = torch.tensor([])
    best_epoch_actuals = torch.tensor([])
    epoch_mads = {"train": [], "val": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            def on_batch_done(idx, outputs, loss, running_mad):
                if phase == "train":
                    if not torch.isnan(loss):
                        loss.backward()
                        if not max_gradient_clip is None:
                            threshold = max_gradient_clip
                            for p in model.parameters():
                                if p.grad is not None:
                                    if p.grad.norm() > threshold:
                                        torch.nn.utils.clip_grad_norm_(p, threshold)
                        optimizer.step()
                    if torch.isnan(loss).any():
                        print(f"Nan loss: {torch.isnan(loss)}| Loss: {loss}")
                if idx % 10 == 0:
                    total, free, unit = log_gpu_memory(0)
                    rms, vms, mem_unit = log_memory()
                    tqdm.write(
                        "Epoch: [{}/{}], Batch: [{}/{}], RAM: {:.2f} {}, {:.2f} {}, GPU: {:.2f} / {:.2f} {}, batch loss: {:.6f}, epoch abs diff mean {:.6f}".format(
                            epoch,
                            num_epochs,
                            idx + 1,
                            len(dataloaders[phase]),
                            rms,
                            mem_unit,
                            vms,
                            mem_unit,
                            total - free,
                            total,
                            unit,
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
                (
                    epoch_loss,
                    epoch_mad,
                    epoch_actuals,
                    epoch_predictions,
                ) = epoch_function(
                    model,
                    criterions[phase],
                    dataloaders[phase],
                    prepare_inputs,
                    prepare_labels,
                    on_batch_done=on_batch_done,
                    optimizer=optimizer,
                )
            epoch_mads[phase].append(epoch_mad)
            epoch_losses[phase].append(epoch_loss)

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
                    wandb.log({"loss": epoch_loss})
                if epoch_loss < best_epoch_loss:
                    best_val_mad = epoch_mad
                    best_epoch_loss = epoch_loss
                    if best_model_path:
                        torch.save(model, best_model_path)

                    best_epoch_actuals = epoch_actuals
                    best_epoch_predictions = epoch_predictions
        print()
        if phase == "val" and should_stop(epoch_losses["val"]):
            print("Stopping early...")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val epoch loss: {best_epoch_loss:4f}")

    # load best model weights
    if best_model_path:
        model = torch.load(best_model_path, map_location={"cpu": "cuda:0"})

    if dataloaders["test"]:
        print("Executing validation on test set...")
        model.eval()
        with torch.set_grad_enabled(False):
            test_loss, test_mad, test_actuals, test_predictions = epoch_function(
                model,
                criterions["test"],
                dataloaders["test"],
                prepare_inputs,
                prepare_labels,
                on_batch_done=on_batch_done,
                optimizer=optimizer,
            )
        print()

    return {
        "model": model,
        "best_epoch_loss": best_epoch_loss,
        "best_val_mad": best_val_mad,
        "epoch_mads": epoch_mads,
        "best_epoch_actuals": best_epoch_actuals,
        "best_epoch_predictions": best_epoch_predictions,
        "test_loss": test_loss,
        "test_mad": test_mad,
        "test_actuals": test_actuals,
        "test_predictions": test_predictions,
    }
