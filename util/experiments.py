import json
from util.plotting import plot_predictions
from util.train_helper import calculate_metrics
import os
from typing import Optional
from typing_extensions import Literal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def store_experiment(
    output_dir_path: str,
    key: str,
    epoch_loss: float,
    epoch_mad: float,
    epoch_predictions: "list[float]",
    epoch_actuals: "list[float]",
    args: Optional[dict] = None,
    epoch_mads: "dict[Literal['train', 'val'], list[float]]" = None,
):
    plot_predictions(
        f"{key}_predictions",
        f"Loss: {epoch_loss:.2f}, mad {epoch_mad:.2f}",
        epoch_predictions,
        epoch_actuals,
        output_dir=output_dir_path,
    )
    if args:
        args = dict(args)
        with open(os.path.join(output_dir_path, "metadata.json"), "w") as f:
            json.dump(args, f, indent=5)

    if epoch_mads:
        plt.clf()

        for values in epoch_mads.values():
            plt.plot(range(len(values)), values)
        plt.xlabel("Epoch")
        plt.ylabel("MAD ")
        plt.title(f"MAD over epochs")
        plt.legend(epoch_mads.keys())
        plt.savefig(os.path.join(output_dir_path, f"{key}_mads.png"))

    metrics = calculate_metrics(epoch_predictions, epoch_actuals, key)
    with open(os.path.join(output_dir_path, f"{key}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=5)

    bin_width = args["bin_width"]
    pred_classes = [
        f"{math.floor(pred / bin_width)*bin_width}-{(math.floor(pred / bin_width)+1)*bin_width}"
        for pred in epoch_predictions
    ]
    actual_classes = [
        f"{math.floor(actual / bin_width)*bin_width}-{(math.floor(actual / bin_width)+1)*bin_width}"
        for actual in epoch_actuals
    ]

    cm = confusion_matrix(actual_classes, pred_classes)
    f, ax = plt.subplots(1, 1, figsize=(15, 15))
    heatmap = sns.heatmap(
        cm,
        norm=LogNorm(),
        cmap="Reds",
        annot=True,
        fmt="n",
        xticklabels=sorted(set(actual_classes + pred_classes)),
        yticklabels=sorted(set(actual_classes + pred_classes)),
        ax=ax,
    )
    heatmap.set_xlabel("Predictions")
    heatmap.set_ylabel("Targets")
    plt.savefig(os.path.join(output_dir_path, f"{key}_sns_heatmap.png"))

    # disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(actual_classes)))
    disp = ConfusionMatrixDisplay.from_predictions(
        actual_classes, pred_classes, cmap="hsv"
    )
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax)

    corrects = 0
    for i in range(len(actual_classes)):
        if actual_classes[i] == pred_classes[i]:
            corrects += 1
    acc = corrects / len(actual_classes)
    disp.figure_.suptitle(f"CM {key} (acc. {acc:.2f})")
    cm_path = os.path.join(output_dir_path, f"{key}_conf_mat.png")
    plt.savefig(cm_path)
    print(f"Results stored in {output_dir_path}")
    return cm_path
