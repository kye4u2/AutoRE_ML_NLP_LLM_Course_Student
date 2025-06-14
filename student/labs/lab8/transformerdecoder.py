import argparse
import math
import os
from pathlib import Path
from typing import List, Union, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from logzero import setup_logger
from torch.utils.data import DataLoader

from lab_common.common import PositionalEncoding, collate_fn, ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer
from lab_common.labs.lab8.embedding_caption_context import EmbeddingCaptionDataset
from lab_common.longformermlm.dataset_utils import DataToken, load_token_map


logger = setup_logger(name=Path(__file__).stem, level="INFO")

PRE_TRAINED_TRANSFORMER_DECODER_CHECKPOINT_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH,
    "lab_datasets",
    "lab8",
    "transformer_decoder_2025-04-11_14-17-49"
)


#######################################
# Transformer Decoder Model
#######################################
class TransformerDecoder(nn.Module):
    def __init__(self, token_map: dict[str, DataToken], d_model=512, nhead=8, num_hidden_layers=6, dropout=0.1,
                 seq_len=5):
        """
        vocab_size: Number of tokens in the target vocabulary.
        d_model: Embedding/model dimension.
        nhead: Number of attention heads.
        num_decoder_layers: Number of Transformer decoder layers.
        dropout: Dropout probability.
        max_len: Maximum caption length.
        """
        super(TransformerDecoder, self).__init__()
        vocab_size = len(token_map)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_hidden_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.token_map = token_map

        # Initialize Accelerator for device management
        self.accelerator = Accelerator()

    def generate_square_subsequent_mask(self, sz):
        """
        Creates a square mask to prevent the decoder from attending to future tokens.
        """
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.accelerator.device)
        return mask

    def forward(self, tgt, embedding, tgt_mask=None, tgt_key_padding_mask=None):
        """
        tgt: Tensor of target token indices, shape (batch_size, tgt_seq_len).
        embedding: Provided embedding, expected shape (S, batch_size, d_model). For a single vector per sample, S should be 1.
        tgt_mask: (Optional) Mask for the target sequence.
        tgt_key_padding_mask: (Optional) Padding mask for target tokens.
        """
        tgt, embedding = tgt.to(self.accelerator.device), embedding.to(self.accelerator.device)

        # Embed target tokens and scale embeddings.
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        # Transformer expects inputs of shape (seq_len, batch_size, d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        output = self.transformer_decoder(tgt_emb, embedding,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)  # back to (batch_size, seq_len, d_model)
        logits = self.fc_out(output)
        return logits

    def greedy_decode(
            self,
            embeddings: Union[torch.Tensor, List[torch.Tensor]],
            max_len: Optional[int] = None,
            remove_special_tokens: bool = True
    ) -> List[List[str]]:
        """
        Generates a sequence using greedy decoding.

        Args:
            embeddings: A tensor of shape (S=1, batch_size, d_model) or a list of such tensors.
            max_len: (Optional) Maximum length of the generated sequence.
            remove_special_tokens: (Default=True) If True, removes special tokens from the output.

        Returns:
            List of token sequences for each batch element.
        """
        token_map = self.token_map

        decoded_sequences: List[List[str]] = []

        ### YOUR CODE HERE ###




        ### END YOUR CODE HERE ###

        return decoded_sequences  # Return list of generated token sequences

    def beam_search_decode(
            self,
            embeddings: Union[torch.Tensor, List[torch.Tensor]],
            beam_width: int = 3,
            max_len: Optional[int] = None,
            remove_special_tokens: bool = True
    ) -> List[List[str]]:
        """
        Generates sequences using beam search decoding.

        Args:
            embeddings: A tensor of shape (S=1, batch_size, d_model) or a list of such tensors.
            beam_width: Number of sequences to keep at each step.
            max_len: Maximum sequence length.
            remove_special_tokens: If True, remove special tokens from final output.

        Returns:
            List of decoded token sequences (one per batch element).
        """
        token_map = self.token_map

        start_symbol = token_map["[SOS]"].id
        eos_symbol = token_map["[EOS]"].id
        pad_symbol = token_map["[PAD]"].id
        unknown_symbol = token_map["[UNK]"].id
        special_tokens = {start_symbol, eos_symbol, pad_symbol, unknown_symbol}

        if max_len is None:
            max_len = self.seq_len

        device = self.accelerator.device

        # Handle list of embeddings
        if isinstance(embeddings, list):
            embeddings = torch.cat([e.to(device) for e in embeddings], dim=1)
        else:
            embeddings = embeddings.to(device)

        batch_size = embeddings.size(1)
        reverse_token_map = {v.id: k for k, v in token_map.items()}
        decoded_outputs: List[List[str]] = []

        for b in range(batch_size):
            single_embedding = embeddings[:, b:b + 1, :]  # shape: (1, 1, d_model)

            # Initialize beam with [SOS] token
            sequences = [(torch.full((1, 1), start_symbol, dtype=torch.long, device=device), 0.0)]

            for _ in range(max_len - 1):
                all_candidates = []

                for seq, score in sequences:
                    tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(device)
                    logits = self.forward(seq, single_embedding, tgt_mask=tgt_mask)
                    prob = torch.log_softmax(logits[:, -1, :], dim=-1)

                    top_k_probs, top_k_tokens = torch.topk(prob, beam_width, dim=-1)

                    for k in range(beam_width):
                        next_word = top_k_tokens[0, k].unsqueeze(0).unsqueeze(0)
                        new_seq = torch.cat([seq, next_word], dim=1)
                        new_score = score + top_k_probs[0, k].item()
                        all_candidates.append((new_seq, new_score))

                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

                if eos_symbol is not None and all(seq[0, -1].item() == eos_symbol for seq, _ in sequences):
                    break

            best_seq = sequences[0][0].squeeze(0)  # shape: (seq_len,)

            # Convert token IDs to tokens
            decoded = [
                reverse_token_map[token.item()]
                for token in best_seq
                if not (remove_special_tokens and token.item() in special_tokens)
            ]
            decoded_outputs.append(decoded)

        return decoded_outputs

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_dir: str,
                        model_filename: str = "model_best_val.pt",
                        accelerator: Optional[Accelerator] = None,
                        load_optimizer: bool = False,
                        ) -> tuple['TransformerDecoder', dict[str, DataToken]]:
        """
        Load a TransformerDecoder model from a checkpoint.

        Args:
            checkpoint_dir: Path to the model checkpoint.
            model_filename: Name of the model checkpoint file.

        Returns:
            TransformerDecoder model.
        """
        # Create an Accelerator instance if not provided
        if accelerator is None:
            accelerator = Accelerator()

        # Load the token_map
        token_map_path = os.path.join(checkpoint_dir, "token_map.jsonl")

        token_map: dict[str, DataToken] = load_token_map(token_map_path)

        logger.info(f"Accelerator device: {accelerator.device}")

        checkpoint_path = os.path.join(checkpoint_dir, model_filename)

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=accelerator.device)

        config = checkpoint["config"]

        model = cls(
            token_map=token_map,
            d_model=config["d_model"],
            nhead=config["num_attention_heads"],
            num_hidden_layers=config["num_hidden_layers"],
            dropout=config["dropout"],
            seq_len=config["seq_len"]
        )

        # Handle "module." prefix in state_dict keys
        state_dict = checkpoint["model_state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            logger.info("Detected 'module.' prefix in state_dict. Removing it.")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        # Move the model to the appropriate device using Accelerator
        model = accelerator.prepare(model)

        model = accelerator.prepare(model)

        if load_optimizer:
            # Ensure optimizer state exists in the checkpoint
            if "optimizer_state_dict" in checkpoint:
                # Reconstruct the optimizer with the model's parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Default LR; adjust as needed
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # Use Accelerator to prepare the optimizer
                optimizer = accelerator.prepare(optimizer)
            else:
                raise ValueError("Optimizer state not found in the checkpoint.")

        return model, token_map

    @classmethod
    def semantic_evaluate(
            cls,
            model,
            dataloader,
            token_map: Dict[str, DataToken],
            accelerator: Accelerator = None,
            verbose: bool = True,
            use_beam_search: bool = False,
            max_samples: int = 200
    ) -> float:
        """
        Evaluates the model using scaled cosine similarity (range: 0 to 1).

        Args:
            model: The model to evaluate.
            dataloader: Dataloader providing batches of (embeddings, captions).
            token_map: Dictionary mapping token strings to DataToken objects.
            accelerator: Device accelerator.
            verbose: If True, prints detailed logs.
            use_beam_search: If True, uses beam search decoding instead of greedy decoding.
            max_samples: Maximum number of samples to process. Defaults to 100.

        Returns:
            Average scaled cosine similarity score over the dataset.
        """

        sentence_summarizer = SentenceSummarizer()
        model.eval()
        total_similarity = 0.0
        total_samples = 0

        if accelerator is None:
            accelerator = Accelerator()

        reverse_token_map = {v.id: k for k, v in token_map.items()}

        special_tokens = {
            token_map["[SOS]"].id,
            token_map["[EOS]"].id,
            token_map["[PAD]"].id,
            token_map["[UNK]"].id,
        }

        with torch.no_grad():
            for batch_idx, (embeddings, captions) in enumerate(dataloader, start=1):
                embeddings = embeddings.to(accelerator.device)
                captions = captions.to(accelerator.device)
                embeddings = embeddings.unsqueeze(0)  # Ensure correct shape

                batch_size = captions.shape[0]

                if use_beam_search:
                    predicted_sequences = model.beam_search_decode(embeddings)
                else:
                    predicted_sequences = model.greedy_decode(embeddings)

                if verbose:
                    logger.info("=" * 60)
                    logger.info(f"Batch {batch_idx} | Processed Samples: {total_samples}/{max_samples}")
                    logger.info("=" * 60)

                for i in range(batch_size):
                    if total_samples >= max_samples:
                        break

                    expected_caption = " ".join(
                        reverse_token_map[idx.item()]
                        for idx in captions[i]
                        if idx.item() in reverse_token_map and idx.item() not in special_tokens
                    )
                    actual_caption = " ".join(predicted_sequences[i])

                    expected_embedding = torch.tensor(sentence_summarizer.summarize(expected_caption))
                    actual_embedding = torch.tensor(sentence_summarizer.summarize(actual_caption))

                    cosine_sim = F.cosine_similarity(expected_embedding, actual_embedding, dim=0).item()
                    scaled_similarity = (cosine_sim + 1) / 2

                    total_similarity += scaled_similarity
                    total_samples += 1

                    if verbose:
                        logger.info(f"Sample {total_samples}/{max_samples}")
                        logger.info(f"{'Expected:':<10} {expected_caption}")
                        logger.info(f"{'Actual:':<10} {actual_caption}")
                        logger.info(f"{'Similarity:':<10} {scaled_similarity:.4f}")
                        logger.info("-" * 50)

                if total_samples >= max_samples:
                    break

        if total_samples > 0:
            logger.info(f"Average Scaled Similarity: {total_similarity / total_samples:.4f}")

        return total_similarity / total_samples if total_samples > 0 else 0.0

    @classmethod
    def semantic_evaluate_from_checkpoint(cls,
                                          checkpoint_dir: str,
                                          batch_size: int = 32,
                                          embedded_caption_dataset: EmbeddingCaptionDataset = None,
                                          use_beam_search:bool = False) -> float:
        """
        Load a TransformerDecoder model from a checkpoint and perform semantic evaluation.

        Args:
            checkpoint_dir: Path to the model checkpoint directory.
            batch_size: Batch size for evaluation.
            embedded_caption_dataset: (Optional) Pre-embedded caption dataset.
            use_beam_search: (Optional) If True, use beam search decoding instead of greedy decoding.

        Returns:
            Average scaled cosine similarity score over the dataset.
        """
        model, token_map = cls.from_checkpoint(checkpoint_dir)

        if embedded_caption_dataset is None:
            logger.info("Loading test dataset from the checkpoint directory for semantic evaluation.")

            test_dataset_file_path = os.path.join(checkpoint_dir, "test_dataset.jsonl")

            embedded_caption_dataset = EmbeddingCaptionDataset.load_from_jsonl(test_dataset_file_path)

        test_loader = DataLoader(embedded_caption_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=collate_fn)

        avg_similarity = TransformerDecoder.semantic_evaluate(model, test_loader, token_map,
                                                              use_beam_search=use_beam_search)

        return avg_similarity


def main():
    # Load model from checkpoint
    parser = argparse.ArgumentParser(description="Load TransformerDecoder model from checkpoint.")
    parser.add_argument("--checkpoint_dir", default=PRE_TRAINED_TRANSFORMER_DECODER_CHECKPOINT_PATH,
                        help="Path to the model checkpoint directory.")

    # add batch size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")

    # Add flag to perform semantic evaluation
    parser.add_argument("--semantic_evaluation", action="store_true", help="Perform semantic evaluation.")

    # Add flag to use beam search
    parser.add_argument("--use_beam_search", action="store_true", help="Use beam search for decoding.")

    args = parser.parse_args()

    if args.semantic_evaluation:

        similarity_avg = TransformerDecoder.semantic_evaluate_from_checkpoint(args.checkpoint_dir,
                                                                              args.batch_size,
                                                                              use_beam_search=args.use_beam_search)

        logger.info(f"Average Scaled Similarity: {similarity_avg:.4f}")

    else:

        model, token_map = TransformerDecoder.from_checkpoint(args.checkpoint_dir)

        logger.info("Model loaded successfully.")


if __name__ == "__main__":
    main()
