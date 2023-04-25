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
    return f"{min_temp}-{max_temp}", diffs.mean(), np.median(diffs), diffs.max()


def evaluate_temp_bins(epoch_predictions, epoch_actuals, bin_width):
    np_preds = np.array(epoch_predictions)
    np_actuals = np.array(epoch_actuals)

    metrics_per_class = {}

    i = 0
    while i * bin_width < np_actuals.max():
        if (i + 1) * bin_width >= np_actuals.min():
            class_label, mean_diff, median_diff, max_diff = metrics_per_temp_range(
                i * bin_width,
                (i + 1) * bin_width,
                np_preds,
                np_actuals,
            )
            metrics_per_class[class_label] = {
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "max_diff": max_diff,
            }
        i = i + 1

    return metrics_per_class


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

    metrics = evaluate_temp_bins(epoch_predictions, epoch_actuals, args["bin_width"])
    with open(
        os.path.join(output_dir_path, f"{key}_metrics_temp_ranges.json"), "w"
    ) as f:
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
    sns.heatmap(
        cm,
        square=True,
        norm=LogNorm(),
        cmap="Reds",
        annot=True,
        xticklabels=sorted(set(actual_classes)),
        yticklabels=sorted(set(actual_classes)),
    )
    plt.savefig(os.path.join(output_dir_path, "sns_heatmap.png"))

    # disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(actual_classes)))
    disp = ConfusionMatrixDisplay.from_predictions(
        actual_classes, pred_classes, cmap="hsv"
    )
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(output_dir_path, "conf_mat.png"))
    print(f"Results stored in {output_dir_path}")
