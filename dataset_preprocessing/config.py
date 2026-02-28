#
# file: dataset_preprocessing/config.py
# desc: Contains dataset configuration definitions
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#

import yaml
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class DataModelConfig:
    """
    Defines the configuration parameters for the data model used in preprocessing the dataset for KlangScribe AI.
    """
    pass


@dataclass
class PreprocessorConfig:
    """
    Defines the configuration parameters for the dataset preprocessor, including input/output directories and data model configuration.
    """
    input_dir: str = MISSING
    output_dir: str = MISSING
    data_model_config: DataModelConfig = field(default_factory=DataModelConfig)


# Register the configuration with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="preprocessor_config", node=PreprocessorConfig)