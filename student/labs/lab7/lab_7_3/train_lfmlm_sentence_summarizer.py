import os
import argparse
from pathlib import Path

from logzero import setup_logger
from omegaconf import OmegaConf

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.longformermlm.longformermlmconfig import load_config
from lab_common.longformermlm.train_longformermlm import train_longformer_mlm_from_config

logger = setup_logger(name=Path(__file__).stem, logfile=f"{Path(__file__).stem}.log", level="INFO")




DEFAULT_TRAINING_CONFIG_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH, "lab_datasets","lab7", "sentence_longformer_mlm", "mini" , "cfg", "default_lfmlm_sentence_summarizer_train_config.yaml"
)


def main():
    parser = argparse.ArgumentParser(description="Train Longformer for Masked Language Modeling (MLM).")

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

    config.train_data_folder_path = os.path.join(
        ROOT_PROJECT_FOLDER_PATH, config.train_data_folder_path
    )

    # Start training
    train_longformer_mlm_from_config(config)


if __name__ == "__main__":
    main()
