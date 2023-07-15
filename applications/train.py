import torch
from torch.utils.data import DataLoader
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
    zero_padding700_collate,
    zero_padding_700,
)
import torch.distributed as dist
from thermostability.hotinfer import HotInferModel

from thermostability.hotinfer_pregenerated import (
    HotInferPregeneratedFC,
    HotInferPregeneratedSummarizerFC,
)
from thermostability.cnn_pregenerated import CNNPregeneratedFC
from thermostability.thermo_dataset import ThermostabilityDataset
import wandb
import argparse
import torch.multiprocessing as mp
import os
from thermostability.repr_summarizer import (
    RepresentationSummarizerSingleInstance,
    RepresentationSummarizerMultiInstance,
    RepresentationSummarizerAverage,
)
from thermostability.fst_hotinfer import FSTHotProt
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from util.weighted_mse import WeightedMSELossMax, WeightedMSELossScaled
from util.train_helper import (
    train_model,
    calculate_metrics,
    get_dataset,
    get_collate_fn,
    execute_epoch,
    execute_epoch_fst,
)
from datetime import datetime as dt
from util.experiments import store_experiment
from esm_custom.esm.esmfold.v1.pretrained import esmfold_v1
from esm_custom.esm.esmfold.v1.esmfold import ESMFold
from esm_custom.esm.sparse_multihead_attention import SparseMultiheadAttention
from tqdm import tqdm

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.cuda.empty_cache()
torch.cuda.list_gpu_processes()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    setup(rank, world_size)
    argsDict = vars(args)
    representation_key = argsDict["representation_key"]
    currentTime = dt.now().strftime("%d-%m-%y_%H:%M:%S")
    results_path = f"results/train/{representation_key}/{currentTime}"
    if rank == 0:
        with wandb.init(config=argsDict, group="DDP", id=f"{rank}_y"):
            run_train_experiment(
                config=wandb.config,
                use_wandb=True,
                results_path=results_path,
                should_log=False,
                rank=rank
            )
def run_train_experiment(
    results_path: str,
    config: dict = None,
    use_wandb: bool = True,
    should_log: bool = True,
    rank: int = 0,
    
    
):
    representation_key = config["representation_key"]
    model_parallel = config["model_parallel"] == "true"
    val_on_trainset = config["val_on_trainset"] == "true"
    limit = config["dataset_limit"]

    valFileName = "data/train.csv" if val_on_trainset else "data/val.csv"
    train_ds = get_dataset(
        config["dataset"],
        "data/train.csv",
        limit,
        representation_key,
        config["seq_len"],
    )
    eval_ds = get_dataset(
        config["dataset"], valFileName, limit, representation_key, config["seq_len"]
    )
    test_ds = get_dataset(
        config["dataset"], "data/test.csv", limit, representation_key, config["seq_len"]
    )

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=get_collate_fn(config["dataset"], representation_key),
        ),
        "val": DataLoader(
            eval_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=get_collate_fn(config["dataset"], representation_key),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=get_collate_fn(config["dataset"], representation_key),
        ),
    }

    train_mean, train_var = train_ds.norm_distr()
    val_mean, val_var = eval_ds.norm_distr()
    test_mean, test_var = test_ds.norm_distr()

    train_losses = {
        "weighted_mse": WeightedMSELossMax(train_mean, train_var),
        "scaled_mse": WeightedMSELossScaled(train_mean, train_var),
        "mse": nn.MSELoss(),
    }
    eval_losses = {
        "weighted_mse": WeightedMSELossMax(val_mean, val_var),
        "scaled_mse": WeightedMSELossScaled(val_mean, val_var),
        "mse": nn.MSELoss(),
    }
    test_losses = {
        "weighted_mse": WeightedMSELossMax(test_mean, test_var),
        "scaled_mse": WeightedMSELossScaled(test_mean, test_var),
        "mse": nn.MSELoss(),
    }

    criterions = {
        "train": train_losses[config["loss"]],
        "val": eval_losses[config["loss"]],
        "test": test_losses[config["loss"]],
    }

    summarizer = (
        RepresentationSummarizerSingleInstance(
            per_residue_output_size=config["summarizer_out_size"],
            num_hidden_layers=config["summarizer_num_layers"],
            activation=nn.ReLU
            if config["summarizer_activation"] == "relu"
            else nn.Identity,
            per_residue_summary=config["summarizer_mode"] == "per_residue",
            p_dropout=config["model_dropoutrate"],
        )
        if config["summarizer_type"] == "single_instance"
        else RepresentationSummarizerMultiInstance(
            per_residue_output_size=config["summarizer_out_size"],
            num_hidden_layers=config["summarizer_num_layers"],
            activation=nn.ReLU
            if config["summarizer_activation"] == "relu"
            else nn.Identity,
            per_residue_summary=config["summarizer_mode"] == "per_residue",
            p_dropout=config["model_dropoutrate"],
        )
        if config["summarizer_type"] in ["700_instance", "multi_instance"]
        else RepresentationSummarizerAverage(
            per_residue_summary=config["summarizer_mode"] == "per_residue"
        )
        if config["summarizer_type"] == "average"
        else None
    )

    input_sizes = {
        "esm_3B": 2560 * 700,
        "esm_s_B_avg": 2560,
        "prott5_avg": 1024,
        "s_s_0_A": 148 * 1024,
        "s_s_0_avg": 1024,
        "s_s_avg": 1024,
        "s_s": 1024 * 700,
    }

    input_size = input_sizes[representation_key]

    thermo = (
        HotInferPregeneratedFC(
            input_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
            p_dropout=config["model_dropoutrate"],
        )
        if config["model"] == "fc"
        else CNNPregeneratedFC(
            input_seq_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
        if config["model"] == "cnn"
        else HotInferPregeneratedSummarizerFC(
            p_dropout=config["model_dropoutrate"],
            summarizer=summarizer,
            thermo_module=HotInferPregeneratedFC(
                input_len=summarizer.per_sample_output_size,
                num_hidden_layers=config["model_hidden_layers"],
                first_hidden_size=config["model_first_hidden_units"],
                p_dropout=config["model_dropoutrate"],
            ),
        )
    )

    model = (
        thermo
        if not model_parallel
        else HotInferModel(representation_key, thermo_module=thermo)
    )

    if config["factorized_rank"] != 0:
        model = FSTHotProt(
            model, factorized_sparse_tuning_rank=config["factorized_rank"]
        )
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )

    if not model_parallel and config["factorized_rank"] == 0:
        model = model.to("cuda:0")

    if use_wandb:
        wandb.watch(thermo)

    weight_decay = 1e-5 if config["weight_regularizer"] else 0
    optimizer_ft = (
        torch.optim.Adam(
            thermo.parameters(), lr=config["learning_rate"], weight_decay=weight_decay
        )
        if config["optimizer"] == "adam"
        else torch.optim.SGD(
            thermo.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=weight_decay,
        )
    )
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)
    if should_log:
        os.makedirs(results_path, exist_ok=True)

    def should_stop(val_epoch_losses: "list[float]"):
        if not config["early_stopping"]:
            return False
        if len(val_epoch_losses) < 3:
            return False

        has_improved = (
            val_epoch_losses[-2] < val_epoch_losses[-3]
            or val_epoch_losses[-1] < val_epoch_losses[-3]
        )
        return not has_improved

    train_result = train_model(
        model,
        criterions,
        exp_lr_scheduler,
        dataloaders,
        use_wandb,
        num_epochs=config["epochs"],
        epoch_function=execute_epoch_fst
        if config["dataset"] == "fst"
        else execute_epoch,
        prepare_inputs=lambda x: x,
        prepare_labels=lambda x: x if not model_parallel else x,
        best_model_path=os.path.join(results_path, "model.pt") if should_log else None,
        should_stop=should_stop,
    )
    best_epoch_predictions = train_result["best_epoch_predictions"]
    best_epoch_actuals = train_result["best_epoch_actuals"]
    best_epoch_loss = train_result["best_epoch_loss"]
    best_epoch_mad = train_result["best_val_mad"]
    epoch_mads = train_result["epoch_mads"]
    test_predictions = train_result["test_predictions"]
    test_actuals = train_result["test_actuals"]
    test_epoch_loss = train_result["test_loss"]
    test_mad = train_result["test_mad"]

    if use_wandb:

        def log_scatter(predictions, actuals, key: str):
            data = [
                [x, y]
                for (x, y) in zip(
                    predictions,
                    actuals,
                )
            ]
            table = wandb.Table(data=data, columns=["predictions", "labels"])
            wandb.log(
                {
                    f"predictions_{key}": wandb.plot.scatter(
                        table, "predictions", "labels", title=key
                    )
                }
            )

        log_scatter(best_epoch_predictions, best_epoch_actuals, "val")
        log_scatter(test_predictions, test_actuals, "test")
        metrics = calculate_metrics(best_epoch_predictions, best_epoch_actuals, "val")
        wandb.log(metrics)
        test_metrics = calculate_metrics(test_predictions, test_actuals, "test")
        wandb.log(test_metrics)

    print("storing experiment")

    if should_log:
        store_experiment(
            results_path,
            "val",
            best_epoch_loss,
            best_epoch_mad,
            best_epoch_predictions,
            best_epoch_actuals,
            config,
            epoch_mads,
        )
        store_experiment(
            results_path,
            "test",
            test_epoch_loss,
            test_mad,
            test_predictions,
            test_actuals,
            args=config,
        )

    return best_epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--model_hidden_layers", type=int, default=1)
    parser.add_argument("--model_first_hidden_units", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_on_trainset", type=str, choices=["true", "false"])
    parser.add_argument("--dataset_limit", type=int, default=1000000)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--model", type=str, default="fc", choices=["fc", "cnn", "summarizer"]
    )
    parser.add_argument("--model_parallel", type=str, choices=["true", "false"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--representation_key", type=str, default="s_s_avg")
    parser.add_argument("--model_dropoutrate", type=float, default=0.3)
    parser.add_argument("--weight_regularizer", type=bool, default=True)
    parser.add_argument("--seq_length", type=int, default=700)
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--summarizer_out_size", type=int, default=1)
    parser.add_argument(
        "--summarizer_activation", default="identity", choices=["relu", "identity"]
    )
    parser.add_argument(
        "--summarizer_type",
        default=None,
        choices=[None, "single_instance", "700_instance", "multi_instance", "average"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pregenerated", "end_to_end", "uni_prot", "fst"],
        default="pregenerated",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["weighted_mse", "mse", "scaled_mse"],
        default="mse",
    )
    parser.add_argument("--summarizer_num_layers", type=int, default=1)
    parser.add_argument(
        "--summarizer_mode",
        type=str,
        choices=["per_residue", "per_repr_position"],
        default="per_residue",
    )
    parser.add_argument("--bin_width", type=int, default=20)
    parser.add_argument("--factorized_rank", type=int, default=4)
    args = parser.parse_args()

    argsDict = vars(args)

    use_wandb = argsDict["wandb"]
    del argsDict["wandb"]
    should_log = not argsDict["nolog"]
    del argsDict["nolog"]
    representation_key = argsDict["representation_key"]
    currentTime = dt.now().strftime("%d-%m-%y_%H:%M:%S")
    results_path = f"results/train/{representation_key}/{currentTime}"

    if use_wandb:
        WORLD_SIZE = torch.cuda.device_count()
        mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)

    else:
        run_train_experiment(
            config=argsDict,
            use_wandb=False,
            results_path=results_path,
            should_log=should_log,
        )
