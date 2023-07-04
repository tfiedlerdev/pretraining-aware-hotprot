import yaml
import os

required_keys = ["WandBApiKey"]
default_value = "<your value here>"


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
        value = self.config[key]
        if value == default_value:
            raise ValueError(f"Please specify your {key} in config.yaml")
        return value
