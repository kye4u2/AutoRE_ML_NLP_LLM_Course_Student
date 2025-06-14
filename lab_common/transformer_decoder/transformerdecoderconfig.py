import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, MISSING
from logzero import setup_logger

logger = setup_logger(name=Path(__file__).stem, level="INFO")


@dataclass
class TransformerDecoderConfig:
    # Path to the training dataset file. If None, dummy data is used.
    train_data_folder_path: Optional[str] = None

    # Batch size for training.
    batch_size: int = 32

    # Number of training epochs.
    epochs: int = 20

    # Learning rate.
    lr: float = 0.001

    # Vocabulary size.
    max_vocab_size: int = 0

    # Number of attention heads.
    nhead: int = 8

    # Number of transformer decoder layers.
    num_layers: int = 6

    # Dropout rate.
    dropout: float = 0.1

    # Maximum caption length (including <SOS> and <EOS> tokens).
    seq_len: int = 20

    # Directory where the best model will be saved.
    save_dir: str = "./models"

    # Seed for reproducibility.
    seed: int = 42

    # Max dataset size. If 0, the entire dataset is used.
    max_dataset_size: int = 0

    # Minimum token length. If 0, no minimum length is enforced.
    min_token_length: int = 3

    # Min tokens in caption
    min_tokens_in_caption: int = 3


DEFAULT_CONFIG = OmegaConf.structured(TransformerDecoderConfig)


def load_config(config_path=None):
    """Loads configuration from a YAML file if provided; otherwise, defaults are used."""
    if config_path:
        if os.path.exists(config_path):
            config = OmegaConf.merge(DEFAULT_CONFIG, OmegaConf.load(config_path))
            logger.info(f"Loaded configuration from {config_path}")
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config = DEFAULT_CONFIG
        logger.info("Using default configuration values")

    return config
