import os
from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf, MISSING
from logzero import logger

@dataclass
class LongformerMLMConfig:
    """
    Configuration class for training Longformer on Masked Language Modeling (MLM).

    Parameters:
    - train_data_folder_path (str): Path to the folder containing tokenized training sequences.
    - validation_data_folder_path (str): Path to the folder containing tokenized validation sequences.
    - token_map_path (str): Path to save or load the token map (default: "token_map.jsonl").

    Model Hyperparameters:
    - d_model (int): Hidden dimension size of the transformer model (default: 1024).
    - attention_window (int): Attention window size for Longformer (default: 512).
    - max_seq_len (int): Maximum sequence length for input sequences (default: 256).
    - num_attention_heads (int): Number of attention heads (default: 16).
    - num_hidden_layers (int): Number of transformer layers (default: 8).

    Training Parameters:
    - num_epochs (int): Number of training epochs (default: 10).
    - learning_rate (float): Learning rate for the optimizer (default: 5e-5).
    - batch_size (int): Batch size for training (default: 16).
    - mask_ratio (float): Ratio of tokens to mask during training (default: 0.10).
    - max_train_num_sequences (int): Maximum number of training sequences (0 for no limit, default: 100).
    - max_val_num_sequences (int): Maximum number of validation sequences (0 for no limit, default: 100).
    - val_mask_ratio (float): Ratio of tokens to mask during validation (default: 0.15).

    Dropout Settings:
    - attention_probs_dropout_prob (float): Dropout probability for self-attention layers (default: 0.1).
    - hidden_dropout_prob (float): Dropout probability for feed-forward layers (default: 0.1).

    Checkpoint & Output:
    - checkpoint_dir (str): Directory to save training checkpoints (default: "./checkpoints").
    """

    # Data paths
    train_data_folder_path: str = MISSING  # Path to training data
    validation_data_folder_path: Optional[str] = None  # Path to validation data
    token_map_path: Optional[str] = None  # Path for saving token map

    # Model hyperparameters
    d_model: int = 1024  # Hidden layer dimension
    attention_window: int = 512  # Attention window size
    max_seq_len: int = 256  # Maximum sequence length
    num_attention_heads: int = 16  # Number of attention heads
    num_hidden_layers: int = 8  # Number of transformer layers
    min_seq_len: int = 5  # Minimum sequence length (0 for no limit)
    num_global_tokens: int = 5  # Number of global tokens

    # Training parameters
    num_epochs: int = 10  # Number of training epochs
    learning_rate: float = 5e-5  # Learning rate
    batch_size: int = 16  # Batch size
    mask_ratio: float = 0.10  # Ratio of masked tokens in training
    max_train_num_sequences: int = 100  # Max training sequences (0 for unlimited)
    max_val_num_sequences: int = 100  # Max validation sequences (0 for unlimited)
    max_unk_ratio: float = 0.20  # Maximum allowed ratio of [UNK] tokens to valid (i.e. non-special) tokens.
    min_token_occurrence: int = 7  # Minimum number of occurrences for a token to be included in the vocabulary

    # Dropout probabilities
    attention_probs_dropout_prob: float = 0.1  # Dropout for self-attention
    hidden_dropout_prob: float = 0.1  # Dropout for feed-forward layers

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"  # Dire

    # Max vocab size
    max_vocab_size: int = 0  # Maximum vocabulary size (0 for no limit)




DEFAULT_CONFIG = OmegaConf.structured(LongformerMLMConfig)


def load_config(config_path=None)-> LongformerMLMConfig:
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
