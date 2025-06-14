#!/usr/bin/env python
import gc
import datetime
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from accelerate import Accelerator
from omegaconf import OmegaConf, DictConfig
from logzero import setup_logger

from lab_common.common import collate_fn
from lab_common.labs.lab8.embedding_caption_context import EmbeddingCaptionContext, EmbeddingCaptionDataset
from lab_common.longformermlm.dataset_utils import split_dataset, save_token_map
from labs.lab8.transformerdecoder import TransformerDecoder
from lab_common.transformer_decoder.transformerdecoderconfig import load_config

logger = setup_logger(name=Path(__file__).stem, level="INFO")


logging.getLogger("gensim").setLevel(logging.CRITICAL)
logging.getLogger("word2vec").setLevel(logging.CRITICAL)
logging.getLogger("tokenizer").setLevel(logging.CRITICAL)

#######################################
# Training and Evaluation Functions
#######################################
def train(model, dataloader, criterion, optimizer, accelerator, vocab_size):
    model.train()
    total_loss = 0.0
    for batch_idx, (embeddings, captions) in enumerate(dataloader):
        embeddings = embeddings.to(accelerator.device)  # (batch_size, d_model)
        captions = captions.to(accelerator.device)  # (batch_size, seq_len)
        embeddings = embeddings.unsqueeze(0)  # shape: (S=1, batch_size, d_model)

        # Teacher forcing: input is all tokens except the last; target is all tokens except the first.
        tgt_input = captions[:, :-1]
        tgt_output = captions[:, 1:]

        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(accelerator.device)

        optimizer.zero_grad()
        logits = model(tgt_input, embeddings, tgt_mask=tgt_mask)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)



def evaluate(model, dataloader, criterion, accelerator, vocab_size):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for embeddings, captions in dataloader:
            embeddings = embeddings.to(accelerator.device)
            captions = captions.to(accelerator.device)
            embeddings = embeddings.unsqueeze(0)

            tgt_input = captions[:, :-1]
            tgt_output = captions[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(accelerator.device)

            logits = model(tgt_input, embeddings, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


#######################################
# Run Training Function with Explicit Parameters
#######################################

def train_transformer_decoder_from_config(config: DictConfig):
    train_transformer_decoder(**config, config=config)


def train_transformer_decoder(train_data_folder_path, batch_size, epochs, lr, max_vocab_size,
                              nhead, num_layers, dropout, seq_len, save_dir, seed, max_dataset_size, min_token_length,
                              min_tokens_in_caption, config: DictConfig):
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Using device: {device}")

    token_map = None
    vocab_size = None
    d_model = None


    logger.info(f"Loading training dataset from {train_data_folder_path}")
    train_dataset, token_map = EmbeddingCaptionContext.load_dataset(
        train_data_folder_path,
        max_vocab_size=max_vocab_size,
        seq_len=seq_len,
        min_token_length=min_token_length,
        min_tokens_in_caption=min_tokens_in_caption
    )

    if max_dataset_size > 0:
        train_dataset = train_dataset[:max_dataset_size]

    d_model = train_dataset[0][0].shape[0]
    logger.info("Model embedding dimension: {}".format(d_model))
    vocab_size = len(token_map)
    logger.info("Vocabulary size: {}".format(vocab_size))

    # Split dataset into training, validation, and test sets.
    train_dataset, val_dataset, test_dataset = split_dataset(train_dataset, seed=seed)
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")


    # DataLoader settings.
    if device.type == "cuda":
        pin_memory = True
        pin_device = f"cuda:{device.index}" if device.index is not None else "cuda:0"
    else:
        pin_memory = False
        pin_device = "cpu"

    num_workers = max(2, os.cpu_count() // 4)
    logger.info(f"Using {num_workers} workers for DataLoader.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_device
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_device
    )
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=pin_device
        )

    # Initialize model, loss, optimizer.
    model = TransformerDecoder(
        token_map=token_map, d_model=d_model,
        nhead=nhead, num_hidden_layers=num_layers,
        dropout=dropout, seq_len=seq_len,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    unwrapped_model = accelerator.unwrap_model(model)

    # Initialize best metrics.
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_similarity = 0.0  # Higher is better (max=1)
    config_saved = False
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(save_dir, f"transformer_decoder_{timestamp}")

    # Helper function to save configuration, token_map, and test dataset only once.
    def save_config_if_needed():
        nonlocal config_saved
        if not config_saved:
            os.makedirs(save_dir, exist_ok=True)
            config_path = os.path.join(save_dir, "config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config, f)
            save_token_map_path = os.path.join(save_dir, "token_map.jsonl")
            save_token_map(token_map, save_token_map_path)
            if test_dataset:
                test_dataset_path = os.path.join(save_dir, "test_dataset.jsonl")
                test_embedding_caption_dataset = EmbeddingCaptionDataset(test_dataset)
                test_embedding_caption_dataset.save_to_jsonl(test_dataset_path)
            config_saved = True

    # Helper function to save a checkpoint.
    def save_checkpoint(file_path, epoch, train_loss, val_loss, extra_metrics=None):
        if extra_metrics is None:
            extra_metrics = {}
        if os.path.exists(file_path):
            os.rename(file_path, file_path + ".bak")
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "seed": seed,
            "config": {
                "d_model": d_model,
                "seq_len": seq_len,
                "num_attention_heads": nhead,
                "num_hidden_layers": num_layers,
                "dropout": dropout,
            },
        }
        checkpoint.update(extra_metrics)
        torch.save(checkpoint, file_path)

    # Training loop.
    for epoch in range(1, epochs + 1):
        if accelerator.is_main_process:
            logger.info(f"\nEpoch {epoch}/{epochs}")

        train_loss = train(unwrapped_model, train_loader, criterion, optimizer, accelerator, vocab_size)
        val_loss = evaluate(unwrapped_model, val_loader, criterion, accelerator, vocab_size)


        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save best validation loss model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_config_if_needed()
            best_val_model_path = os.path.join(save_dir, "model_best_val.pt")
            save_checkpoint(best_val_model_path, epoch, train_loss, val_loss)
            logger.info(f"Saved best validation model to {best_val_model_path}")

        # Save best training loss model.
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_config_if_needed()
            best_train_model_path = os.path.join(save_dir, "model_best_train.pt")
            save_checkpoint(best_train_model_path, epoch, train_loss, val_loss)
            logger.info(f"Saved best training model to {best_train_model_path}")



    # Optionally evaluate on test dataset.
    if test_dataset:
        logger.info("Evaluating on test dataset.")
        test_loss = evaluate(unwrapped_model, test_loader, criterion, accelerator, vocab_size)
        logger.info(f"Test Loss: {test_loss:.4f}")
    else:
        logger.info("No test dataset provided. Skipping evaluation.")

    logger.info(f"Best validation loss : {best_val_loss}")




    # Save final model.
    final_model_path = os.path.join(save_dir, "model_final.pt")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save({
        "epoch": epoch,
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "vocab_size": vocab_size,
        "seed": seed,
        "config": {
            "d_model": d_model,
            "seq_len": seq_len,
            "num_attention_heads": nhead,
            "num_hidden_layers": num_layers,
            "dropout": dropout,
        },
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Clean up.
    del model, optimizer, train_loader, val_loader, criterion
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Completed Training.")



#######################################
# Main Function: Parsing Config Using OmegaConf
#######################################
def main():
    parser = argparse.ArgumentParser(description="Train Transformer Decoder with OmegaConf")
    parser.add_argument(
        "--config_path",
        type=str,
        default="default_transformer_decoder_config.yaml",
        help="Path to the YAML configuration file (default: default_config.yaml)"
    )
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config_path)

    # Display the loaded configuration.
    logger.info(f"***Loaded configuration****:\n{OmegaConf.to_yaml(config)}\n**********")

    # Call run_training with explicit parameters from the config.
    train_transformer_decoder(**config, config=config)


if __name__ == "__main__":
    main()
