import torch
from torch.utils.data import DataLoader
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from thermostability.hotinfer import HotInferModel

from thermostability.hotinfer_pregenerated import (
    HotInferPregeneratedFC,
    HotInferPregeneratedSummarizerFC,
)
from thermostability.cnn_pregenerated import (
    CNNPregeneratedFC,
    CNNPregeneratedFullHeightFC,
)
import wandb
import argparse
import os
from thermostability.repr_summarizer import (
    RepresentationSummarizerSingleInstance,
    RepresentationSummarizerMultiInstance,
    RepresentationSummarizerAverage,
)
from thermostability.fst_hotinfer import FSTHotProt
from util.weighted_mse import WeightedMSELossMax, WeightedMSELossScaled
from util.train_helper import (
    train_model,
    calculate_metrics,
    get_dataset,
    get_collate_fn,
    execute_epoch,
    execute_epoch_fst,
    log_gpu_memory,
    log_memory,
    DatasetNames,
)
from datetime import datetime as dt
from util.experiments import store_experiment
from pynvml import nvmlInit
from thermostability.huggingface_esm import (
    ESMForThermostability,
    model_names as esm2_model_names,
)
from util.yaml_config import YamlConfig

nvmlInit()
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.cuda.empty_cache()
# torch.cuda.list_gpu_processes()
yamlConfig = YamlConfig()


def str_to_bool(value):
    if value.lower() in ["true", "t"]:
        return True
    elif value.lower() in ["false", "f"]:
        return False
    elif value.lower() in ["none", "n"]:
        return None
    else:
        raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def run_train_experiment(
    results_path: str,
    config: dict = None,
    use_wandb: bool = True,
    should_log: bool = True,
):
    representation_key = config["representation_key"]
    model_parallel = config["model_parallel"] == "true"
    val_on_trainset = config["val_on_trainset"] == "true"
    limit = config["dataset_limit"]

    log_gpu_memory(0)

    train_ds = get_dataset(
        config["dataset"],
        config["dataset_split"],
        yamlConfig,
        "train",
        limit,
        representation_key,
        config["seq_length"],
    )
    eval_ds = get_dataset(
        config["dataset"],
        config["dataset_split"],
        yamlConfig,
        "train" if val_on_trainset else "val",
        limit,
        representation_key,
        config["seq_length"],
    )
    test_ds = get_dataset(
        config["dataset"],
        config["dataset_split"],
        yamlConfig,
        "test",
        limit,
        representation_key,
        config["seq_length"],
    )

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=get_collate_fn(config, representation_key),
        ),
        "val": DataLoader(
            eval_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=get_collate_fn(config, representation_key),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=get_collate_fn(config, representation_key),
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

    input_sizes = {
        "esm_3B": 2560 * config["seq_length"],
        "esm_650M": 1280 * config["seq_length"],
        "esm_150M": 640 * config["seq_length"],
        "esm_35M": 480 * config["seq_length"],
        "esm_8M": 320 * config["seq_length"],
        "esm_s_B_avg": 2560,
        "prott5_avg": 1024,
        "s_s_0_A": 148 * 1024,
        "s_s_0_avg": 1024,
        "s_s_avg": 1024,
        "s_s": 1024 * config["collate_k"],
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
            per_residue_summary=config["summarizer_mode"] == "per_residue",
            per_sample_output_size=int(
                input_sizes[representation_key] / config["seq_length"]
            )
            if config["summarizer_mode"] == "per_repr_position"
            else config["seq_length"],
        )
        if config["summarizer_type"] == "average"
        else None
    )

    thermo = (
        HotInferPregeneratedFC(
            input_len=input_sizes[representation_key],
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
            p_dropout=config["model_dropoutrate"],
        )
        if config["model"] == "fc"
        else CNNPregeneratedFC(
            input_seq_len=input_sizes[representation_key],
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
        if config["model"] == "cnn"
        else CNNPregeneratedFullHeightFC()
        if config["model"] == "cnn_full_height"
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
        if config["model"] == "summarizer"
        else None
    )

    model = (
        ESMForThermostability.from_config(config, yamlConfig)
        if config["model"] == "hugg_esm"
        else (
            thermo
            if not model_parallel
            else HotInferModel(representation_key, thermo_module=thermo)
        )
    )

    if config["factorized_rank"] != 0 and config["dataset"] == "fst":
        model = FSTHotProt(
            model,
            esm_model=config["esm_version"],
            factorized_sparse_tuning_rank=config["factorized_rank"],
        )

    model = model.to("cuda:0")

    if not model_parallel and config["factorized_rank"] == 0:
        model = model.to("cuda:0")

    if use_wandb:
        wandb.watch(model)

    optimizer_ft = (
        torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_regularizer"],
        )
        if config["optimizer"] == "adam"
        else torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=config["weight_regularizer"],
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

    total, free, unit = log_gpu_memory(0)
    rms, vms, mem_unit = log_memory()

    print(f"GPU Mem before train start: {total - free} / {total} {unit}")
    print(f"RAM before training: RSS {rms} {mem_unit}, VMS {vms} {mem_unit}")

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
        prepare_inputs=lambda x: x.to("cuda:0") if torch.is_tensor(x) else x,
        prepare_labels=lambda x: x.to("cuda:0") if torch.is_tensor(x) else x,
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
        val_cm = store_experiment(
            results_path,
            "val",
            best_epoch_loss,
            best_epoch_mad,
            best_epoch_predictions,
            best_epoch_actuals,
            config,
            epoch_mads,
        )
        test_cm = store_experiment(
            results_path,
            "test",
            test_epoch_loss,
            test_mad,
            test_predictions,
            test_actuals,
            args=config,
        )
        if use_wandb:
            wandb.log({"val_cm": wandb.Image(val_cm), "test_cm": wandb.Image(test_cm)})

    return best_epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--model_hidden_layers", type=int, default=1)
    parser.add_argument("--model_first_hidden_units", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--val_on_trainset", type=str, choices=["true", "false"], default="false"
    )
    parser.add_argument("--dataset_limit", type=int, default=1000000)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--model",
        type=str,
        default="fc",
        choices=["fc", "cnn", "summarizer", "cnn_full_height", "hugg_esm"],
    )
    parser.add_argument("--model_parallel", type=str, choices=["true", "false"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--representation_key", type=str, default="s_s_avg")
    parser.add_argument("--model_dropoutrate", type=float, default=0.3)
    parser.add_argument("--weight_regularizer", type=float, default=0.0)
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
        choices=DatasetNames,
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
    parser.add_argument(
        "--collate_fn",
        type=str,
        choices=["pad700", "k_max_sum_pooling", "k_max_var_pooling"],
        default=None,
    )
    parser.add_argument(
        "--collate_k",
        type=int,
        default=700,
    )
    parser.add_argument("--bin_width", type=int, default=20)
    parser.add_argument("--factorized_rank", type=int, default=None)
    parser.add_argument(
        "--esm_version",
        type=str,
        choices=[
            "esm2_t48_15B_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t33_650M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t6_8M_UR50D",
        ],
        default="esm2_t6_8M_UR50D",
    )
    parser.add_argument(
        "--hugg_esm_size",
        type=str,
        choices=esm2_model_names.keys(),
        default=None,
    )
    parser.add_argument(
        "--hugg_esm_freeze",
        type=str_to_bool,
        default="None",
    )
    parser.add_argument(
        "--hugg_esm_layer_norm",
        type=str_to_bool,
        default="None",
    )
    parser.add_argument(
        "--hugg_esm_batch_norm",
        type=str_to_bool,
        default="None",
    )
    parser.add_argument(
        "--hugg_esm_pooling",
        choices=["bos_token", "mean"],
        default=None,
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="NAME of the wandb run. If not specified, a name is generated automatically",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        choices=["ours", "ours_median", "flip"],
        default="ours",
        help="Name of the dataset split to use. 'ours' for our split which has multiple measurements per protein, 'ours_median' for our split with median values per protein (train, val and test don't have overlapping protein clusters and val and test weren't seen during ESM2 training), flip for the split from the flip paper (single measurement per protein, first measurement per protein taken. Val and test not ensured to be unseen during ESM2 training)",
    )
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
        wandb.login(key=yamlConfig["WandBApiKey"])
        with wandb.init(
            config=argsDict,
            name=argsDict["wandb_run_name"],
            project="hot-prot-applications",
            entity="hotprot",
        ):
            run_train_experiment(
                config=argsDict,
                use_wandb=True,
                results_path=results_path,
                should_log=should_log,
            )
    else:
        run_train_experiment(
            config=argsDict,
            use_wandb=False,
            results_path=results_path,
            should_log=should_log,
        )
