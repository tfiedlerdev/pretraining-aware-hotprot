import yaml
import os

required_keys = ["WandBApiKey"]
default_value = "<your value here>"
help_text = {
    "WandBApiKey": "Your Weights and Biases API key. You can find it in your W&B account settings.",
    "HuggESMCacheDir": "Directory to store cached representations with format /<cache_dir>/start_token_{model_size}/<seq>.pt and /<cache_dir>/start_token_{model_size}/sequences.csv. For hugg_esm model, representation key is inferred from model size",
    "DatasetSplitsPath": "Path to directory containing the dataset split CSV files (train.csv, val.csv, test.csv, train_median.csv, val_median.csv, test_median.csv, train_FLIP.csv, val_FLIP.csv, test_FLIP.csv).",
    "DatasetPath": "Path to directory containing the pregenerated dataset files with format /<dataset_path>/<representation_key>/<seq_index>.pt and /<dataset_path>/<representation_key>/sequences.csv.",
}


class YamlConfig:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = None

    def _load_config(self):
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                print(
                    "Created a config.yaml file in project root. Please specify your WandBApiKey in it if you intend to use wandb."
                )
                for key in required_keys:
                    f.write(f"{key}: {default_value}\n")
        with open(self.config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, key):
        if self.config is None:
            self.config = self._load_config()
        value = self.config.get(key, default_value)
        if value == default_value:
            explanation = help_text.get(key, "")
            explanation = f"({explanation})" if explanation != "" else explanation
            raise ValueError(f"Please specify {key} in config.yaml {explanation}")
        return value
