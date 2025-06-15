import argparse
import logging
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator
from logzero import setup_logger

from labs.lab7.lab_7_1.convert_tokens_to_tensor import convert_tokens_to_tensor
from lab_common.longformermlm.longformermlm import LongformerMLM
from lab_common.nlp.tokenizer import Tokenizer

logger = setup_logger(name=Path(__file__).stem, level=logging.INFO)


class SentenceLongformerMLM(LongformerMLM):
    def __init__(self, *args, **kwargs):
        """
        Initialize SentenceLongformerMLM by passing all positional and keyword arguments
        to the parent class LongformerMLM.
        """
        super().__init__(*args, **kwargs)

        self._tokenizer = Tokenizer()


    def generate_embedding(self, text:str)-> torch.Tensor:
        """
        Generate embeddings for a given input text using the Longformer MLM model.
        If a list of tokens is provided, it is first converted into a tensor.

        Args:
            text (str): Input text to generate embeddings for.

        Returns:
            torch.Tensor: CLS token embedding as the sequence representation.
        """

        # Initialize CLS token embedding
        cls_embedding = torch.zeros(self._d_model)

        ### YOUR CODE HERE ###





        ### END YOUR CODE HERE ###

        return cls_embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text embeddings.
        """
        # initialize similarity to be inf or -
        similarity_value = float('inf')

        ### YOUR CODE HERE ###




        ### END YOUR CODE HERE ###


        return similarity_value

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

        tokenized_tensor = convert_tokens_to_tensor(tokens, self._token_map, self._max_seq_len)

        return tokenized_tensor # Shape: (1, seq_len)



def main():
    parser = argparse.ArgumentParser(description="Load Longformer MLM model from a checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the checkpoint directory.")
    parser.add_argument("--load_optimizer", action="store_true",
                        help="Flag to load the optimizer state from the checkpoint.")
    parser.add_argument("--output_attentions", action="store_true",
                        help="Flag to override the checkpoint configuration to output attentions.")
    parser.add_argument("-c", "--compare_texts", action="store_true",
                        help="Flag to compute similarity between two default texts.")

    args = parser.parse_args()

    # Load model from checkpoint
    model, _ = SentenceLongformerMLM.from_checkpoint(
        args.checkpoint_dir,
        load_optimizer=args.load_optimizer,
        output_attentions=args.output_attentions,
        model_filename="model_best_val_accuracy.pt",
    )

    tokenizer = Tokenizer()

    # If comparison flag is set, compute similarity
    if args.compare_texts:

        default_text1 = "security is important"
        default_text2 = "security is the highest priority"

        #default_text1 = "security is boring"
        #default_text2 = "security is  very interesting"

        #default_text1 = "security is fascinating"
        #default_text2 = "security is  very interesting"

        #default_text1 = "play the record"
        #default_text2 = "record the play"

        tokenized_text1 = tokenizer.tokenize(default_text1)
        logger.info(f"Tokenized text 1: {tokenized_text1}")

        tokenized_text2 = tokenizer.tokenize(default_text2)
        logger.info(f"Tokenized text 2: {tokenized_text2}")

        token_to_id1 = model.tokens_to_tensor(tokenized_text1)
        token_to_id2 = model.tokens_to_tensor(tokenized_text2)

        # Convert tensors to lists and slice first 15 tokens
        token_list1 = token_to_id1.squeeze(0).tolist()[:15]  # Remove batch dim if present
        token_list2 = token_to_id2.squeeze(0).tolist()[:15]

        # Log token IDs and first 15 token strings
        logger.info(f"Tokenized text 1 IDs (first 15): {token_list1}")
        logger.info(f"Tokenized text 2 IDs (first 15): {token_list2}")

        similarity = model.compute_similarity(default_text1, default_text2)
        logger.info(f"Similarity between default texts: {similarity:.4f}")


if __name__ == "__main__":
    main()
