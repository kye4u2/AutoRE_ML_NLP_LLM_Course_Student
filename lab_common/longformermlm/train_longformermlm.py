import argparse
import datetime
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
import numpy as np
from accelerate import Accelerator
from typing import Dict, Optional

from logzero import setup_logger
from tqdm import tqdm  # Import tqdm for progress bar
from omegaconf import OmegaConf, DictConfig

from lab_common.longformermlm.longformermlm import LongformerMLM
from labs.lab7.lab_7_2.mask_tokens import _mask_tokens
from lab_common.longformermlm.dataset_utils import load_tokens_from_jsonl, SequenceDataset
from lab_common.longformermlm.longformermlmconfig import load_config
from lab_common.longformermlm.sequence_processor import load_data_sequences_from_folder

logger = setup_logger(name=Path(__file__).stem, logfile=f"{Path(__file__).stem}.log", level="INFO")

# NOTE: This training explicitly requires OUTPUT_ATTENTIONS to be True in LongformerMLM
OUTPUT_ATTENTIONS = True  # Output attentions for Longformer model


def mask_tokens(inputs,
                pad_token_id,
                cls_token_id,
                unk_token_id,  # Include UNK token ID
                mask_token_id,
                mask_ratio=0.15):
    """
    Masks random tokens in the input sequence for MLM training.
    Excludes PAD, CLS, and UNK tokens from being masked.
    """
    return _mask_tokens(inputs,
                        pad_token_id,
                        cls_token_id,
                        unk_token_id,
                        mask_token_id,
                        mask_ratio)


def validate(model,
             dataloader,
             token_map,
             num_global_tokens,
             token_weights=None,
             mask_ratio=0.15,
             attention_window=512,
             mean_alignment_loss_weight=0.3,
             mlm_loss_weight=1.0,
             weighted_alignment_loss_weight=0.3,
             accelerator=None):
    """
    Perform validation on the given dataloader with global attention consistent with training.
    Incorporates MLM loss, mean alignment loss, and weighted alignment loss.

    Args:
        model: The trained model.
        dataloader: DataLoader for validation data.
        token_map: Token map with special tokens.
        token_weights: Optional weights for CrossEntropyLoss.
        mask_ratio: Ratio of tokens to mask for validation (default: 0.15).
        attention_window: Attention window size for global attention (default: 512).
        mean_alignment_loss_weight (float): Weight for the mean alignment loss.
        mlm_loss_weight (float): Weight for the MLM loss.
        weighted_alignment_loss_weight (float): Weight for the weighted alignment loss.
        accelerator: For handling multi-GPU/TPU environments (can be None if not in distributed mode).
    """

    model.eval()

    # 1. Initialize loss functions
    criterion = (
        nn.CrossEntropyLoss(weight=token_weights, ignore_index=-100)
        if token_weights
        else nn.CrossEntropyLoss(ignore_index=-100)
    )

    mse_loss_fn = nn.MSELoss()  # MSE Loss for alignment

    # 2. Initialize accumulators
    total_loss = 0
    total_mlm_loss = 0
    total_mean_alignment_loss = 0
    total_weighted_alignment_loss = 0
    correct_predictions = 0
    total_masked_tokens = 0

    # Track how many batches we skip because they contain only -100 labels
    skipped_batches = 0

    with torch.no_grad():

        # Create tqdm progress bar only for the main process
        pbar = None
        if accelerator is not None and accelerator.is_main_process:
            pbar = tqdm(total=len(dataloader), desc=f"Validation", unit="batch")

        for batch_idx, batch in enumerate(dataloader):
            # 2.1 Mask tokens for validation
            input_data, labels, mask = mask_tokens(
                batch,
                pad_token_id=token_map["[PAD]"]["id"],
                cls_token_id=token_map["[CLS]"]["id"],
                unk_token_id=token_map["[UNK]"]["id"],
                mask_token_id=token_map["[MASK]"]["id"],
                mask_ratio=mask_ratio,
            )

            # Clamp input IDs to valid range
            #input_data = torch.clamp(input_data, min=0, max=len(token_map) - 1)

            # 2.2 Create attention masks
            attention_mask = input_data != token_map["[PAD]"]["id"]
            global_attention_mask = torch.zeros_like(input_data)
            global_attention_mask[:, :num_global_tokens] = 1  # CLS + global tokens


            # 2.3 Forward pass
            logits, outputs = model(input_data, attention_mask, global_attention_mask)

            # Extract embeddings
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS embedding
            token_embeddings = outputs.last_hidden_state[:, 1:, :]  # Token embeddings (skip CLS at index 0)

            # Remove CLS logits to match the labels
            logits = logits[:, 1:, :]  # shape: (batch_size, seq_len-1, vocab_size)
            masked_logits = logits.reshape(-1, len(token_map))  # flatten to (batch_size*(seq_len-1), vocab_size)
            masked_labels = labels[:, 1:].reshape(-1)  # flatten to (batch_size*(seq_len-1))

            # Check if the entire batch is only -100 (i.e., no valid tokens)
            valid_mask_count = (masked_labels != -100).sum().item()
            if valid_mask_count == 0:
                # All labels are -100 => skip this batch for MLM loss
                logger.warning(f"Batch {batch_idx}: All labels are -100. Skipping this batch.")
                skipped_batches += 1

                # Update progress bar if present
                if accelerator is not None and accelerator.is_main_process and pbar is not None:
                    pbar.update(1)
                continue  # Move to next batch

            # -----------------------------
            # Optional: Additional Checks
            # -----------------------------
            vocab_size = len(token_map)
            valid_label_range_mask = ((masked_labels >= 0) & (masked_labels < vocab_size)) | (masked_labels == -100)
            invalid_label_count = (~valid_label_range_mask).sum().item()
            if invalid_label_count > 0:
                logger.warning(f"Batch {batch_idx}: Found {invalid_label_count} label(s) out of range!")

            if torch.isnan(masked_labels).any() or torch.isinf(masked_labels).any():
                logger.warning(f"Batch {batch_idx}: masked_labels contain NaN or Inf!")

            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                logger.warning(f"Batch {batch_idx}: masked_logits contain NaN or Inf!")

            for name, embedding in [("cls_embedding", cls_embedding),
                                    ("token_embeddings", token_embeddings)]:
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    logger.warning(f"Batch {batch_idx}: {name} has NaN or Inf!")

            # 3. Compute losses
            # 3.1 Masked Language Modeling Loss (MLM)
            mlm_loss = criterion(masked_logits, masked_labels)

            if torch.isnan(mlm_loss).any() or torch.isinf(mlm_loss).any():
                logger.warning(f"Batch {batch_idx}: MLM Loss is NaN or Inf! masked_labels: {masked_labels}")

            # 3.2 Mean Alignment Loss
            token_mean_embedding = token_embeddings.mean(dim=1)  # shape: (batch_size, hidden_dim)
            mean_alignment_loss = mse_loss_fn(cls_embedding, token_mean_embedding)

            # 3.3 Weighted Alignment Loss
            cls_attention_scores = outputs.global_attentions[-1][:, :, :, 0]  # shape depends on model output
            cls_attention_weights, _ = cls_attention_scores.max(dim=1)  # max-pool over heads
            cls_attention_weights = F.softmax(cls_attention_weights, dim=-1)

            seq_len = min(token_embeddings.size(1), cls_attention_weights.size(1))
            aligned_token_embeddings = token_embeddings[:, :seq_len, :]
            aligned_cls_attention_weights = cls_attention_weights[:, :seq_len]

            weighted_token_embeddings = torch.sum(
                aligned_cls_attention_weights.unsqueeze(-1) * aligned_token_embeddings,
                dim=1
            )
            weighted_alignment_loss = mse_loss_fn(cls_embedding, weighted_token_embeddings)

            # 4. Combine losses with weights
            weighted_mlm_loss = mlm_loss_weight * mlm_loss
            weighted_mean_alignment_loss = mean_alignment_loss_weight * mean_alignment_loss
            weighted_weighted_alignment_loss = weighted_alignment_loss_weight * weighted_alignment_loss

            loss = weighted_mlm_loss + weighted_mean_alignment_loss + weighted_weighted_alignment_loss

            # 5. Accumulate losses
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_mean_alignment_loss += mean_alignment_loss.item()
            total_weighted_alignment_loss += weighted_alignment_loss.item()

            # 6. Accuracy for masked tokens
            predictions = masked_logits.argmax(dim=-1)
            mask_indices = (masked_labels != -100)
            correct_predictions += (predictions[mask_indices] == masked_labels[mask_indices]).sum().item()
            total_masked_tokens += mask_indices.sum().item()

            # Update progress bar on main process
            if accelerator is not None and accelerator.is_main_process and pbar is not None:
                avg_loss = total_loss / (
                        batch_idx + 1 - skipped_batches)  # subtract skipped for a more accurate average
                pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f} (skipped: {skipped_batches})"})
                pbar.update(1)

        # Close the progress bar on the main process
        if accelerator is not None and accelerator.is_main_process and pbar is not None:
            pbar.close()

    # 7. Compute averages
    num_processed_batches = len(dataloader) - skipped_batches

    # Convert losses to tensors (necessary for accelerator.gather)
    average_loss = torch.tensor(
        total_loss / num_processed_batches if num_processed_batches > 0 else float("inf"),
        device=accelerator.device
    )
    avg_mlm_loss = torch.tensor(
        total_mlm_loss / num_processed_batches if num_processed_batches > 0 else float("inf"),
        device=accelerator.device
    )
    avg_mean_alignment_loss = torch.tensor(
        total_mean_alignment_loss / num_processed_batches if num_processed_batches > 0 else float("inf"),
        device=accelerator.device
    )
    avg_weighted_alignment_loss = torch.tensor(
        total_weighted_alignment_loss / num_processed_batches if num_processed_batches > 0 else float("inf"),
        device=accelerator.device
    )

    # Gather loss values across all GPUs
    all_avg_losses = accelerator.gather(average_loss)
    all_avg_mlm_losses = accelerator.gather(avg_mlm_loss)
    all_avg_mean_alignment_losses = accelerator.gather(avg_mean_alignment_loss)
    all_avg_weighted_alignment_losses = accelerator.gather(avg_weighted_alignment_loss)

    # 8. Compute accuracy across GPUs by gathering correct and total counts
    local_correct = torch.tensor(correct_predictions, device=accelerator.device)
    local_total = torch.tensor(total_masked_tokens, device=accelerator.device)
    gathered_correct = accelerator.gather(local_correct)
    gathered_total = accelerator.gather(local_total)



    # Compute mean loss values only on the main process
    if accelerator.is_main_process:

        overall_correct = gathered_correct.sum().item()
        overall_total = gathered_total.sum().item()
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

        final_avg_loss = all_avg_losses.mean().item()
        final_avg_mlm_loss = all_avg_mlm_losses.mean().item()
        final_avg_mean_alignment_loss = all_avg_mean_alignment_losses.mean().item()
        final_avg_weighted_alignment_loss = all_avg_weighted_alignment_losses.mean().item()

        # 8. Log final stats
        logger.info(f"Skipped {skipped_batches} batch(es) with only -100 labels.")
        logger.info(f"Validation Loss Breakdown:")
        logger.info(f"  MLM Loss: {final_avg_mlm_loss:.4f}")
        logger.info(f"  Mean Alignment Loss: {final_avg_mean_alignment_loss:.4f}")
        logger.info(f"  Weighted Alignment Loss: {final_avg_weighted_alignment_loss:.4f}")
        logger.info(f"  Total Loss: {final_avg_loss:.4f}")
        logger.info(f"  Accuracy: {overall_accuracy:.4%}")
    else:
        overall_accuracy = None


    return average_loss, overall_accuracy


def compute_token_weights(token_map: Dict[str, Dict[str, int]]) -> torch.Tensor:
    """
    Computes weights for tokens based on their frequencies in token_map.
    Infrequent tokens are given higher weights, and frequent tokens lower weights.

    Args:
        token_map (Dict[str, Dict[str, int]]): A dictionary where each key is a token
                                               and its value is a dictionary containing metadata,
                                               including the "count" key for frequency
                                               and the "id" key for alignment.

    Returns:
        torch.Tensor: A tensor containing the computed weights for each token, ordered by token IDs.
    """
    # Sort token_map by token IDs to ensure alignment with CrossEntropyLoss weight parameter
    sorted_tokens = sorted(token_map.items(), key=lambda item: item[1]["id"])  # Sort by 'id'

    # Extract token frequencies in the correct order
    token_frequencies = np.array([token["count"] for _, token in sorted_tokens])
    total_count = token_frequencies.sum()

    # Compute inverse frequency weights and normalize
    weights = total_count / (token_frequencies + 1e-8)  # Prevent division by zero
    weights = weights / weights.sum()  # Normalize weights to sum to 1

    # Return weights as a PyTorch tensor
    return torch.tensor(weights, dtype=torch.float32)


def train_longformer_mlm_from_config(config: DictConfig):
    """
    Train longformer mlm from a configuration file.
    """

    train_longformer_mlm(**config, config=config)


def train_longformer_mlm(train_data_folder_path: str, validation_data_folder_path: Optional[str],
                         token_map_path: str,
                         d_model, attention_window, max_seq_len,
                         num_attention_heads, num_hidden_layers, num_epochs, learning_rate, batch_size, mask_ratio, num_global_tokens,
                         max_train_num_sequences=0, max_val_num_sequences=0, max_unk_ratio=0.20,
                         mean_alignment_loss_weight=0, mlm_loss_weight=1, weighted_alignment_loss_weight=0,
                         checkpoint_dir="checkpoints", attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,
                         min_seq_len: int = 5, max_vocab_size=0, min_token_occurrence:int = 7 , config: DictConfig = None,
                         test_dataset=None):
    """
    Train Longformer for Masked Language Modeling with additional losses and save checkpoints:
      - Best training loss: model_best_train.pt
      - Best validation loss: model_best_val.pt
      - Best similarity (if computed): model_best_similarity.pt
      - Final model: model_final.pt

    All files are saved into a dedicated save directory (created using a timestamp) within the given checkpoint_dir.
    """
    accelerator = Accelerator()
    device = accelerator.device

    # Log device info on the main process only.
    if accelerator.is_main_process:
        logger.info(f"Using device: {device}")

    # --- CPU-Bound Work: Data Loading (run only on main process) ---
    if accelerator.is_main_process:
        # Load token map (if provided)
        if token_map_path is not None:
            data_token_dict = load_tokens_from_jsonl(token_map_path)
        else:
            data_token_dict = None

        logger.info("Loading training data sequences...")
        train_data_sequences, data_token_dict = load_data_sequences_from_folder(
            train_data_folder_path,
            data_token_dict,
            allow_token_addition=True,
            min_sequence_length=min_seq_len,
            max_vocab_size=max_vocab_size,
            mask_ratio=mask_ratio,
            max_unk_ratio=max_unk_ratio,
            min_token_occurrence=min_token_occurrence
        )
        train_token_count = sum(token.count for token in data_token_dict.values())
        train_unk_count = data_token_dict["[UNK]"].count

        if validation_data_folder_path is not None:
            logger.info("Loading validation data sequences...")
            validation_data_sequences, val_token_dict = load_data_sequences_from_folder(
                validation_data_folder_path,
                data_token_dict,
                min_sequence_length=min_seq_len,
                max_vocab_size=max_vocab_size,
                mask_ratio=mask_ratio,
                max_unk_ratio=max_unk_ratio,
                min_token_occurrence=min_token_occurrence
            )
        else:
            logger.info("Validation path not provided. Splitting training data into 80/20 train/validation split...")
            np.random.shuffle(train_data_sequences)
            split_index = int(len(train_data_sequences) * 0.8)
            new_train_data_sequences = train_data_sequences[:split_index]
            validation_data_sequences = train_data_sequences[split_index:]
            train_data_sequences = new_train_data_sequences
            val_token_dict = data_token_dict

        unk_count = val_token_dict["[UNK]"].count - train_unk_count
        total_count = sum(token.count for token in val_token_dict.values()) - train_token_count
        if total_count > 0:
            unk_percent = (unk_count / total_count) * 100
            logger.info(f"Total Validation tokens: {total_count}, UNK tokens: {unk_count} ({unk_percent:.2f}%)")

        training_data = [seq.tokenized_sequence for seq in train_data_sequences]
        validation_data = [seq.tokenized_sequence for seq in validation_data_sequences]

        if max_train_num_sequences > 0:
            np.random.shuffle(training_data)
            training_data = training_data[:max_train_num_sequences]
            logger.debug(f"Using a maximum of {len(training_data)} sequences for training.")
        if max_val_num_sequences > 0:
            np.random.shuffle(validation_data)
            validation_data = validation_data[:max_val_num_sequences]
            logger.debug(f"Using a maximum of {len(validation_data)} sequences for validation.")

        logger.info(f"Loaded {len(training_data)} sequences for training.")
        logger.info(f"Loaded {len(validation_data)} sequences for validation.")
    else:
        # For non-main processes, set variables to None.
        data_token_dict = None
        train_data_sequences = None
        validation_data_sequences = None
        training_data = None
        validation_data = None
        val_token_dict = None

    # Wait for the main process to finish loading the data.
    accelerator.wait_for_everyone()

    # --- Broadcast the Loaded Data from the Main Process ---
    data_token_dict = broadcast_object_list([data_token_dict], from_process=0)[0]
    training_data = broadcast_object_list([training_data], from_process=0)[0]
    validation_data = broadcast_object_list([validation_data], from_process=0)[0]

    # --- Dataset and DataLoader Setup (runs on all processes) ---
    train_dataset = SequenceDataset(training_data, max_seq_len)
    val_dataset = SequenceDataset(validation_data, max_seq_len)

    if device.type == "cuda":
        pin_memory = True
        pin_device = f"cuda:{device.index}" if device.index is not None else "cuda:0"
    else:
        pin_memory = False
        pin_device = "cpu"

    num_workers = max(2, os.cpu_count() // 4)
    if accelerator.is_main_process:
        logger.info(f"Using {num_workers} workers for DataLoader.")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  pin_memory_device=pin_device)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                pin_memory_device=pin_device)

    # --- Model, Optimizer, Loss, and Accelerator Preparation ---
    # Convert the token map into a plain dict for the model.
    token_map = {token: data_token_dict[token].__dict__ for token in data_token_dict}

    model = LongformerMLM(token_map,
                          d_model,
                          attention_window,
                          max_seq_len,
                          num_attention_heads,
                          num_hidden_layers,
                          num_global_tokens,
                          attention_probs_dropout_prob,
                          hidden_dropout_prob,
                          output_attentions=OUTPUT_ATTENTIONS)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss()

    # Prepare the model, optimizer, and dataloaders for distributed training.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # ----------------- Checkpoint saving setup -----------------
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_accuracy = float('-inf')

    config_saved = False
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(checkpoint_dir, f"longformer_mlm_{timestamp}")


    def save_config_if_needed():
        nonlocal config_saved
        if not config_saved:
            os.makedirs(save_dir, exist_ok=True)
            if config:
                config_path = os.path.join(save_dir, "config.yaml")
                with open(config_path, "w") as f:
                    OmegaConf.save(config, f)
                logger.info(f"Config saved at {config_path}")
            token_map_path_file = os.path.join(save_dir, "token_map.json")
            with open(token_map_path_file, 'w') as f:
                json.dump(token_map, f, indent=4)
            logger.info(f"Token map saved at {token_map_path_file}")
            config_saved = True


    def save_checkpoint(file_path, epoch, train_loss, val_loss, extra_metrics=None):
        if extra_metrics is None:
            extra_metrics = {}
        if os.path.exists(file_path):
            os.rename(file_path, file_path + ".bak")
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                "d_model": d_model,
                "attention_window": attention_window,
                "max_seq_len": max_seq_len,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "num_global_tokens": num_global_tokens,
                "attention_probs_dropout_prob": attention_probs_dropout_prob,
                "hidden_dropout_prob": hidden_dropout_prob,
                "output_attentions": OUTPUT_ATTENTIONS,
            },
        }
        checkpoint.update(extra_metrics)
        torch.save(checkpoint, file_path)
    # -----------------------------------------------------------

    # --------------------- Training Loop -----------------------
    for epoch in range(1, num_epochs + 1):
        if accelerator.is_main_process:
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_masked_tokens = 0
        skipped_batches = 0
        total_batches = len(train_dataloader)
        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") \
            if accelerator.is_main_process else None

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_data, labels, mask = mask_tokens(
                batch,
                pad_token_id=token_map["[PAD]"]["id"],
                cls_token_id=token_map["[CLS]"]["id"],
                unk_token_id=token_map["[UNK]"]["id"],
                mask_token_id=token_map["[MASK]"]["id"],
                mask_ratio=mask_ratio,

            )
            input_data = torch.clamp(input_data, min=0, max=len(token_map) - 1)
            attention_mask = input_data != token_map["[PAD]"]["id"]
            global_attention_mask = torch.zeros_like(input_data)
            global_attention_mask[:, :num_global_tokens] = 1  # CLS + first few tokens get global attention

            logits, outputs = model(input_data, attention_mask, global_attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            token_embeddings = outputs.last_hidden_state[:, 1:, :]
            logits = logits[:, 1:, :]
            masked_logits = logits.reshape(-1, len(token_map))
            masked_labels = labels[:, 1:].reshape(-1)

            train_mask_count = (masked_labels != -100).sum().item()
            if train_mask_count == 0:
                logger.warning(f"Batch {batch_idx}: All labels are -100. Skipping this batch.")
                skipped_batches += 1
                if pbar is not None:
                    pbar.update(1)
                continue

            mlm_loss = criterion(masked_logits, masked_labels)
            token_mean_embedding = token_embeddings.mean(dim=1)
            mean_alignment_loss = mse_loss_fn(cls_embedding, token_mean_embedding)

            cls_attention_scores = outputs.global_attentions[-1][:, :, :, 0]
            cls_attention_weights, _ = cls_attention_scores.max(dim=1)
            cls_attention_weights = F.softmax(cls_attention_weights, dim=-1)

            seq_len = min(token_embeddings.size(1), cls_attention_weights.size(1))
            aligned_token_embeddings = token_embeddings[:, :seq_len, :]
            aligned_cls_attention_weights = cls_attention_weights[:, :seq_len]
            weighted_token_embeddings = torch.sum(
                aligned_cls_attention_weights.unsqueeze(-1) * aligned_token_embeddings, dim=1
            )
            weighted_alignment_loss = mse_loss_fn(cls_embedding, weighted_token_embeddings)

            weighted_mlm_loss = mlm_loss_weight * mlm_loss
            weighted_mean_alignment_loss = mean_alignment_loss_weight * mean_alignment_loss
            weighted_weighted_alignment_loss = weighted_alignment_loss_weight * weighted_alignment_loss

            total_loss = weighted_mlm_loss + weighted_mean_alignment_loss + weighted_weighted_alignment_loss

            if not torch.isfinite(total_loss):
                logger.warning(f"Batch {batch_idx}: Non-finite loss encountered. Skipping this batch.")
                logger.warning(f"MLM Loss: {mlm_loss.item():.4f}, Mean Alignment Loss: {mean_alignment_loss.item():.4f}, "
                               f"Weighted Alignment Loss: {weighted_alignment_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
                skipped_batches += 1
                optimizer.zero_grad()
                continue

            accelerator.backward(total_loss)
            optimizer.step()

            predictions = masked_logits.argmax(dim=-1)
            mask_indices = (masked_labels != -100)
            correct_predictions += (predictions[mask_indices] == masked_labels[mask_indices]).sum().item()
            total_masked_tokens += mask_indices.sum().item()
            epoch_loss += total_loss.item()

            if pbar is not None:
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f} (skipped: {skipped_batches})"})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        train_loss_epoch = epoch_loss / total_batches if total_batches > 0 else float("inf")
        train_accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0.0

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_accuracy:.4%}")

        # Run validation.
        val_loss, val_accuracy = validate(model,
                                 val_dataloader,
                                 token_map,
                                 num_global_tokens,
                                 None,
                                 mask_ratio=mask_ratio,
                                 attention_window=attention_window,
                                 mean_alignment_loss_weight=mean_alignment_loss_weight,
                                 mlm_loss_weight=mlm_loss_weight,
                                 weighted_alignment_loss_weight=weighted_alignment_loss_weight,
                                 accelerator=accelerator)

        logger.info(f"[device: {device}] Waiting for everyone to finish validation...")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

            # Save best validation loss model.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_config_if_needed()
                best_val_model_path = os.path.join(save_dir, "model_best_val_loss.pt")
                save_checkpoint(best_val_model_path, epoch, train_loss_epoch, val_loss)
                logger.info(f"Saved best validation model to {best_val_model_path}")

            # Save best training loss model.
            if train_loss_epoch < best_train_loss:
                best_train_loss = train_loss_epoch
                save_config_if_needed()
                best_train_model_path = os.path.join(save_dir, "model_best_train_loss.pt")
                save_checkpoint(best_train_model_path, epoch, train_loss_epoch, val_loss)
                logger.info(f"Saved best training model to {best_train_model_path}")

            # Save best accuracy model.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_config_if_needed()
                best_val_accuracy_model_path = os.path.join(save_dir, "model_best_val_accuracy.pt")
                save_checkpoint(best_val_accuracy_model_path, epoch, train_loss_epoch, val_loss, {"val_accuracy": val_accuracy})
                logger.info(f"Saved best accuracy model to {best_val_accuracy_model_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Best Validation Loss: {best_val_loss}")

        final_model_path = os.path.join(save_dir, "model_final.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        final_checkpoint = {
            "epoch": num_epochs,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss_epoch,
            "val_loss": val_loss,
            "config": {
                "d_model": d_model,
                "attention_window": attention_window,
                "max_seq_len": max_seq_len,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "attention_probs_dropout_prob": attention_probs_dropout_prob,
                "hidden_dropout_prob": hidden_dropout_prob,
                "output_attentions": OUTPUT_ATTENTIONS,
            },
        }
        torch.save(final_checkpoint, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

    # Clean up.
    accelerator.wait_for_everyone()
    del model, optimizer, train_dataloader, val_dataloader, criterion
    torch.cuda.empty_cache()
    gc.collect()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Completed Training.")



def main():
    parser = argparse.ArgumentParser(description="Train Longformer for Masked Language Modeling with CLS token")
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML configuration file (optional)")

    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config_path)

    # Ensure mandatory paths are provided
    if not config.train_data_folder_path or not config.validation_data_folder_path:
        raise ValueError("Both train_data_folder_path and validation_data_folder_path must be specified.")

    # Ensure checkpoint directory exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory ensured: {config.checkpoint_dir}")

    # Display the loaded configuration.
    logger.info(f"***Loaded configuration****:\n{OmegaConf.to_yaml(config)}\n**********")

    # Train model
    train_longformer_mlm(**config, config=config)


if __name__ == "__main__":
    main()
