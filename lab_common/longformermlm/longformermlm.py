import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
from tqdm import tqdm

import torch
from accelerate import Accelerator
from torch import nn as nn
from transformers import LongformerConfig, LongformerModel


from logzero import setup_logger

from lab_common.longformermlm.longformermlmconfig import load_config, LongformerMLMConfig

logger = setup_logger(name=Path(__file__).stem, level="INFO")

logging.getLogger("gensim").setLevel(logging.WARN)



class LongformerMLM(nn.Module):
    def __init__(self,
                 token_map,
                 d_model,
                 attention_window,
                 max_seq_len,
                 num_attention_heads,
                 num_hidden_layers,
                 num_global_tokens,
                 attention_probs_dropout_prob=0.1,  # Dropout in self-attention
                 hidden_dropout_prob=0.1,  # Dropout in feed-forward layers
                 output_attentions=True,
                 accelerator=None

                 ):
        super(LongformerMLM, self).__init__()

        if num_global_tokens > max_seq_len:
            raise ValueError("Number of global tokens cannot exceed the maximum sequence length.")

        if num_global_tokens <=0:
            raise ValueError("Number of global tokens must be greater than 0.")


        # Determine vocab size from token_map
        vocab_size = len(token_map)

        config = LongformerConfig(
            attention_window=attention_window,
            max_position_embeddings=max_seq_len + 2,  # Add 1 for CLS token, and ensure safe limit
            hidden_size=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            vocab_size=len(token_map),
            intermediate_size=4 * d_model,  # Default value in Longformer
            output_attentions=output_attentions,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob
        )

        # Longformer model
        self.longformer = LongformerModel(config)
        self.longformer.pooler = None  # Remove the pooler layer
        self.output_attentions = output_attentions
        self._accelerator = accelerator
        self._d_model = d_model
        self._num_global_tokens = num_global_tokens

        # Linear layer for MLM predictions
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Fetch special token IDs from token_map
        self.cls_token_id = token_map["[CLS]"]["id"]
        self.mask_token_id = token_map["[MASK]"]["id"]
        self.unk_token_id = token_map["[UNK]"]["id"]
        self.pad_token_id = token_map["[PAD]"]["id"]

        # Store token map and max sequence length
        self._token_map = token_map
        self._max_seq_len = max_seq_len

    def forward(self, x, attention_mask, global_attention_mask):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            global_attention_mask (torch.Tensor): Global attention mask of shape (batch_size, seq_len).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'logits' (torch.Tensor): Predictions for masked language modeling of shape (batch_size, seq_len, vocab_size).
                - 'last_hidden_state' (torch.Tensor): Last hidden states from the transformer of shape (batch_size, seq_len, hidden_dim).
                - 'global_attentions' (torch.Tensor, optional): Global attentions from the model if self.output_attentions is True.
        """
        # Pass inputs through the Longformer model
        outputs = self.longformer(
            input_ids=x,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )

        # Extract the last hidden states
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

        # Compute MLM predictions
        logits = self.fc_out(last_hidden_state)  # Shape: (batch_size, seq_len, vocab_size)

        return logits, outputs


    def tokens_to_tensor(self,
                         tokens: List[str],
                         ) -> torch.Tensor:
        """
        Convert a list of tokens into tensors using the provided token map.
        If a token is not found, the `[UNK]` token is used. Pads the sequence to `max_sequence` length if necessary.

        Args:
            tokens (List[str]): List of tokens to convert to tensors.
            token_map (Dict[str, Dict[str, int]]): A dictionary mapping tokens to their IDs and counts.
            max_sequence (int): Maximum length of the sequence, including special tokens.

        Returns:
            torch.Tensor: Tensor of token IDs, padded to `max_sequence` length.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")



    @classmethod
    def get_config_from_checkpoint(cls, checkpoint_dir: str) -> LongformerMLMConfig:
        """
        Load the model configuration from a checkpoint directory.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.

        Returns:
            dict: Model configuration loaded from the checkpoint.
        """
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        config = load_config(config_path)
        return config

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_dir: str,
            model_filename: str = "model_best_val_accuracy.pt",
            accelerator: Optional[Accelerator] = None,
            load_optimizer: bool = False,
            output_attentions: Optional[bool] = None
    ) -> Union[Tuple["LongformerMLM", dict], Tuple["LongformerMLM", dict, torch.optim.Optimizer]]:
        """
        Load a model and its token_map from a checkpoint directory, using `Accelerator` for device management.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            accelerator (Accelerator, optional): `Accelerator` instance for device placement. If None, a new one is created.
            load_optimizer (bool): Whether to load the optimizer state. Default is False.

        Returns:
            tuple:
                - (LongformerMLM, token_map, optimizer) if load_optimizer is True.
                - (LongformerMLM, token_map) if load_optimizer is False.
        """
        # Create an Accelerator instance if not provided
        if accelerator is None:
            accelerator = Accelerator()

        # Load the token_map
        token_map_path = os.path.join(checkpoint_dir, "token_map.json")
        with open(token_map_path, "r") as f:
            token_map = json.load(f)

        # Load model state and configuration from checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, model_filename)
        #checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # Load to CPU initially

        logger.info(f"Accelerator device: {accelerator.device}")

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=accelerator.device)

        # Extract model hyperparameters from the checkpoint
        config = checkpoint["config"]

        if output_attentions is not None:
            logger.debug("Overriding 'output_attentions' from checkpoint with provided value.")

        if "num_global_tokens" not in config:
            config["num_global_tokens"] = 1
            logger.warning("num_global_tokens not found in the checkpoint. Setting to default value of 1.")

        # Instantiate the model
        model = cls(
            token_map=token_map,
            d_model=config["d_model"],
            attention_window=config["attention_window"],
            max_seq_len=config["max_seq_len"],
            num_attention_heads=config["num_attention_heads"],
            num_hidden_layers=config["num_hidden_layers"],
            num_global_tokens=config["num_global_tokens"],
            attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            output_attentions=output_attentions if output_attentions is not None else config["output_attentions"],
            accelerator=accelerator
        )

        # Handle "module." prefix in state_dict keys
        state_dict = checkpoint["model_state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            logger.info("Detected 'module.' prefix in state_dict. Removing it.")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        # Move the model to the appropriate device using Accelerator
        model = accelerator.prepare(model)

        if load_optimizer:
            # Ensure optimizer state exists in the checkpoint
            if "optimizer_state_dict" in checkpoint:
                # Reconstruct the optimizer with the model's parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Default LR; adjust as needed
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # Use Accelerator to prepare the optimizer
                optimizer = accelerator.prepare(optimizer)
                return model, token_map, optimizer
            else:
                raise ValueError("Optimizer state not found in the checkpoint.")

        return model, token_map

    @property
    def d_model(self) -> int:
        """
        Get the model's hidden dimension size. Also known as the embedding dimension.
        """
        return self._d_model

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator


def load_longformer_mlm(checkpoint_dir: str,
                        load_optimizer: bool = False,
                        output_attentions: bool = False) -> LongformerMLM:
    """
    Load a Longformer MLM model from a checkpoint directory.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
        load_optimizer (bool): Flag to load the optimizer state from the checkpoint.

    Returns:
        LongformerMLM: The loaded Longformer MLM model.
    """
    try:

        model, _ = LongformerMLM.from_checkpoint(checkpoint_dir=checkpoint_dir,
                                                 load_optimizer=load_optimizer,
                                                 output_attentions=output_attentions)

    except FileNotFoundError as e:
        logger.error(e)
        return
    except ValueError as e:
        logger.error(e)
        return

    return model


def main():
    parser = argparse.ArgumentParser(description="Load Longformer MLM model from a checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the checkpoint directory.")
    parser.add_argument("--load_optimizer", action="store_true",
                        help="Flag to load the optimizer state from the checkpoint.")
    parser.add_argument("--output_attentions", action="store_true",
                        help="Flag to override the checkpoint configuration to output attentions.")
    parser.add_argument("-g","--generate_embedding", action="store_true",
                        help="Flag to generate embeddings for a sequence of tokens.")
    parser.add_argument("--input_tokens",
                        type=str,
                        nargs='+',
                        default=["Get", "Sub", "Load", "Store", "Branch", "Bit_Extend", "Ret"],
                        required=False,
                        help="List of tokens to convert to tensors and generate embeddings.")

    args = parser.parse_args()

    model = load_longformer_mlm(args.checkpoint_dir,
                                load_optimizer=args.load_optimizer,
                                output_attentions=args.output_attentions)



if __name__ == "__main__":
    main()