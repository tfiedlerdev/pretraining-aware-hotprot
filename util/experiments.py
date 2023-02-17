import json
from util.plotting import plot_predictions
import os
from typing_extensions import Literal
import matplotlib.pyplot as plt

def store_experiment(
    output_dir_path: str,
    epoch_loss: float,
    epoch_mad: float,
    epoch_predictions: "list[float]",
    epoch_actuals: "list[float]",
    args: dict,
    epoch_mads: "dict[Literal['train', 'val'], list[float]]" = None,
):
    plot_predictions(
        f"predictions",
        f"Loss: {epoch_loss: .2f}, mad {epoch_mad: .2f}",
        epoch_predictions,
        epoch_actuals,
        output_dir=output_dir_path,
    )
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
        plt.savefig(os.path.join(output_dir_path, f"mads.png"))

    print(f"Results stored in {output_dir_path}")
