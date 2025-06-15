import os
import argparse
from pathlib import Path

from logzero import setup_logger
from omegaconf import OmegaConf
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.transformer_decoder.train_transformerdecoder import train_transformer_decoder_from_config
from lab_common.transformer_decoder.transformerdecoderconfig import load_config

logger = setup_logger(name=Path(__file__).stem, level="INFO")

DEFAULT_TRAINING_CONFIG_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH, "labs", "lab8", "lab_8_3", "cpu_test_function_caption_transformer_decoder_config.yaml"
)

DEFAULT_TRAINING_DATASET_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab8", "function_embeddings_train_mini.jsonl"
)


def main():
    parser = argparse.ArgumentParser(description="Transformer decoder training for function captioning.")

    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help="Path to YAML configuration file for training (default: predefined project config path)."
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Display the loaded configuration.
    logger.info(f"***Loaded configuration****:\n{OmegaConf.to_yaml(config)}\n**********")

    if config.train_data_folder_path is None:
        config.train_data_folder_path = DEFAULT_TRAINING_DATASET_PATH
        logger.info(f"Using default training dataset path: {DEFAULT_TRAINING_DATASET_PATH}")

    # Start training
    train_transformer_decoder_from_config(config)


if __name__ == "__main__":
    main()
