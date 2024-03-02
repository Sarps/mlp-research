import os
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from archive.lib.language_index import LanguageIndex


@dataclass_json
@dataclass
class TransformerModelConfigs:
    num_layers: int = 4
    d_model: int = 128
    num_heads: int = 8
    dff: int = 512
    dropout_rate: int = 0.1
    input_vocabs: str = 'en.vocabs.json'
    target_vocabs: str = 'tw.vocabs.json'
    # Training parameters
    input_dataset: str = 'en.txt'
    input_max_timesteps: int = 128
    target_dataset: str = 'tw.txt'
    target_max_timesteps: int = 128
    batch_size: int = 32
    train_epochs: int = 10
    # CustomSchedule parameters
    init_lr: float = 0.00001
    lr_after_warmup: float = 0.0005
    final_lr: float = 0.0001
    warmup_epochs: int = 2
    decay_epochs: int = 18

    def input_language_index(self, asset_path: str) -> LanguageIndex:
        return LanguageIndex(
            'en',
            os.path.join(asset_path, self.input_vocabs),
            self.input_max_timesteps,
            os.path.join(asset_path, self.input_dataset),
        )

    def target_language_index(self, asset_path: str) -> LanguageIndex:
        return LanguageIndex(
            'tw',
            os.path.join(asset_path, self.target_vocabs),
            self.input_max_timesteps,
            os.path.join(asset_path, self.target_dataset),
        )

    def save(self, config_path: str):
        with open(config_path, 'w') as file:
            file.write(self.to_json())

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            return cls.from_json(f.read())
