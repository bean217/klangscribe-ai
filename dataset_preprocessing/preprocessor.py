#
# file: dataset_preprocessing/preprocessor.py
# desc: Provides functionality for preprocessing canonical chart data into a format suitable for training KlangScribe models.
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#

import yaml
import hydra
import logging
import argparse
import polars as pl
from enum import Enum
from omegaconf import OmegaConf
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from chart.processor import ChartProcessor
from dataset_preprocessing.config import PreprocessorConfig, DataModelConfig
from utils.logging import format_exception


#####################
#                   #
#   Logging Setup   #
#                   #
#####################


log = logging.getLogger(__name__)


###########################
#                         #
#   Preprocessing Logic   #
#                         #
###########################


class SongPreprocessStatus(Enum):
    SUCCESS = "success"         # indicates that a song was successfully preprocessed
    DISCARD = "discard"         # indicates that a song was discarded during preprocessing (e.g., due to validation failure)
    FAILURE = "failure"         # indicates that an error occurred during preprocessing of a song (e.g., file read/write error, processing error, etc.), which should be investigated and resolved


@dataclass
class SongPreprocessResult:
    """
    Defines the result of preprocessing a single song, including the song ID and any relevant metadata or error information.
    """
    song_id: str
    success: SongPreprocessStatus
    error_message: str = None


def process_song(cfg: PreprocessorConfig, split_path: str, song_id: str):
    """
    Preprocesses a single song
    """
    try:
        dm_cfg = cfg.data_model_config

        # (1) convert the .npz file for this song to its vectorized chart data
        chart_path = f"{split_path}/charts/sid_{song_id}.npz"
        chart_data = ChartProcessor.process_chart(chart_path)

        # (2) convert vectorized chart into its absolute-time representation
        abstime_chart_data = ChartProcessor.convert_to_abstime(chart_data)

        # (3) validate chart by checking minimum delta time between notes (if any notes are closer than min_delta, log a warning and skip this song)
        min_delta = dm_cfg.min_delta_time
        if not ChartProcessor.validate_chart(abstime_chart_data, min_delta):
            log.warning(f"Skipping song {song_id} due to minimum delta time violation.")
            return SongPreprocessResult(song_id=song_id, success=SongPreprocessStatus.DISCARD, error_message="Minimum delta time violation")
    
        # (4) convert the absolute-time chart data into its fixed-grid representation
        fixed_grid_chart_data = ChartProcessor.convert_to_fixed_grid(abstime_chart_data, dm_cfg.grid_size)
    
        # (5) convert the fixed-grid chart data into event-based format, where each event corresponds to a change in the state of the chart (e.g., note on, note off, etc.)
        event_based_chart_data = ChartProcessor.convert_to_event_based(fixed_grid_chart_data)

        # (6) chunk the fixed-grid chart data into overlapping windows based on data model context length (padding as necessary)
        #     resulting shape is (num_chunks, context_length, num_features)
        chunked_chart_data = ChartProcessor.chunk_chart_data(event_based_chart_data, dm_cfg.context_length)

        # (7) save the chunked chart data to

    except Exception as e:
        log.error(f"Error processing song {song_id}: {format_exception(e)}")
        return SongPreprocessResult(song_id=song_id, success=SongPreprocessStatus.FAILURE, error_message=format_exception(e))


def preprocess_split(cfg: PreprocessorConfig, split: str):
    """
    Preprocesses the specified data split (train/val/test) by processing the raw .chart files in the input directory and saving the preprocessed files to the output directory.

    Args:
        - cfg (PreprocessorConfig): The configuration parameters for the preprocessor, including input and output directories and data model configuration.
        - split (str): The data split to preprocess (e.g., "train", "val", "test").
    """
    log.info(f"Preprocessing {split} split...")
    split_path: str = f"{cfg.input_dir}/{split}"

    # read the local `song_metadata.parquet` file to get the list of songs in this split
    metadata_df = pl.read_parquet(f"{split_path}/song_metadata.parquet")
    song_ids = metadata_df["dir_id"].to_list()

    with ThreadPoolExecutor() as exec:
        futures = []
        for song_id in song_ids:
            futures.append(exec.submit(process_song, cfg, split_path, song_id))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error processing song: {format_exception(e)}")


def preprocess_dataset(cfg: PreprocessorConfig):
    """
    Preprocess the canonical dataset files located in the input directory and save the preprocessed files to the output directory.
    
    Args:
        - cfg (PreprocessorConfig): The configuration parameters for the preprocessor, including input and output directories and data model configuration.
    """
    log.info(f"Preprocessing dataset with input directory: {cfg.input_dir} and output directory: {cfg.output_dir}")

    # Preprocess each data split (train/val/test)
    preprocess_split(cfg,split="train")
    preprocess_split(cfg, split="val")
    preprocess_split(cfg,split="test")


@hydra.main(version_base=None, config_name="preprocessor_config")
def main(cfg: PreprocessorConfig):
    log.info(OmegaConf.to_yaml(cfg))

    # Add your preprocessing logic here
    log.info("Starting dataset preprocessing...")
    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
