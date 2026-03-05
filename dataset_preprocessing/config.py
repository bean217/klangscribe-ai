#
# file: dataset_preprocessing/config.py
# desc: Contains dataset configuration definitions
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#

from dataclasses import dataclass, field
import omegaconf as oc
from typing import Optional
from hydra.core.config_store import ConfigStore


@dataclass
class DataModelConfig:
    """
    Defines the configuration parameters for the data model used in preprocessing the dataset for KlangScribe AI.
    """
    window_size: int = 100              # number of fixed-grid ticks per chart window (e.g., 100 ticks = 2 seconds at 50 ticks per second) 
    time_step_size: float = 0.02        # 20ms time steps; 50 fixed-grid ticks per second
    resolution: int = 480               # standard Clone Hero resolution of 480 ticks per quarter note (PPQN)
    target_sample_rate: int = 24000     # target sample rate for audio processing (e.g., resampling, spectrogram generation, etc.)

@dataclass
class PreprocessorConfig:
    """
    Defines the configuration parameters for the dataset preprocessor, including input/output directories and data model configuration.
    """
    input_dir: str = oc.MISSING
    output_dir: str = oc.MISSING        
    num_workers: Optional[int] = None   # number of worker threads to use for parallel preprocessing; -1 to use all available CPU cores
    data_model_config: DataModelConfig = field(default_factory=DataModelConfig)


# Register the configuration with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="preprocessor_config", node=PreprocessorConfig)