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
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, as_completed

from chart.processor import process_chart
from dataset_preprocessing.config import PreprocessorConfig, DataModelConfig


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


def process_song(cfg: PreprocessorConfig, song_id: str):
    """
    Preprocesses a single song
    """
    pass


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
            futures.append(exec.submit(process_song, cfg, song_id))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error processing song: {e}")


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
