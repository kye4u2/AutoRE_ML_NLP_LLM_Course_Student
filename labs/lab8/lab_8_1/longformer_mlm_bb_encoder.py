import os
import queue
import uuid
from typing import List, Union

import torch
from accelerate import Accelerator
from tqdm import tqdm

from blackfyre.datatypes.contexts.vex.vexbbcontext import VexBasicBlockContext
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.longformermlm.longformermlm import LongformerMLM
from labs.lab7.lab_7_1.convert_tokens_to_tensor import convert_tokens_to_tensor


class LongformerMLMBasicBlockEncoder(LongformerMLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_embedding(self, instruction_sequence: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a given input text using the Longformer MLM model.
        If a list of tokens is provided, it is first converted into a tensor.

        Args:
            instruction_sequence (List[str]): Input text to generate embeddings for.

        Returns:
            torch.Tensor: CLS token embedding as the sequence representation.
        """

        # Initialize CLS token embedding
        cls_embedding = torch.zeros(self._d_model)

        # Use Accelerator to manage device placement
        if self._accelerator is None:
            self._accelerator = Accelerator()

        # Set model to evaluation mode
        self.eval()


        # Ensure the sequence starts with [CLS]
        if len(instruction_sequence) == 0 or instruction_sequence[0] != "[CLS]":
            instruction_sequence = ["[CLS]"] + instruction_sequence

        tokenized_instruction_sequence = self.tokens_to_tensor(tokens=instruction_sequence)
        attention_mask = (tokenized_instruction_sequence != self._token_map["[PAD]"]["id"]).long()  # Mask non-PAD tokens
        global_attention_mask = torch.zeros_like(tokenized_instruction_sequence).long()  # Default: no global attention
        global_attention_mask[:, :self._num_global_tokens] = 1  # CLS + first few tokens get global attention

        # Move data to the appropriate device
        tokenized_instruction_sequence = tokenized_instruction_sequence.to(self._accelerator.device)
        attention_mask = attention_mask.to(self._accelerator.device)
        global_attention_mask = global_attention_mask.to(self._accelerator.device)

        # Pass data through the model to generate embeddings
        with torch.no_grad():
            outputs = self.longformer(
                input_ids=tokenized_instruction_sequence,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            # Use CLS token embedding as the sequence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS embedding

        return cls_embedding


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

        return tokenized_tensor  # Shape: (1, seq_len)

    def encode(self, basic_block_contexts: list[VexBasicBlockContext]) -> torch.Tensor:
        """
        Encode the basic block context into a longformer MLM input
        """
        embeddings = torch.zeros((len(basic_block_contexts), self._d_model))

        ### YOUR CODE HERE ###


        ### END YOUR CODE HERE ###

        return embeddings

def main():
    DEFAULT_TEST_BCC_FILE_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "Blackfyre", "test",
                                              "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")

    DEFAULT_CHECKPOINT_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                                  "lab_datasets",
                                                  "lab8",
                                                  "longformer_mlm_bb_encoder",
                                                  "small",
                                                  "checkpoints",
                                                  "longformer_mlm_2025-03-10_21-01-20")

    (basic_block_encoder, token_map) = LongformerMLMBasicBlockEncoder.from_checkpoint(
        DEFAULT_CHECKPOINT_FOLDER_PATH)

    # Load the basic block contexts
    vex_binary_context: VexBinaryContext = VexBinaryContext.load_from_file(DEFAULT_TEST_BCC_FILE_PATH)

    FUNCTION_ADDRESS = 0x2986c

    vex_function_context: VexFunctionContext = vex_binary_context.get_function_context(FUNCTION_ADDRESS)

    vex_basic_block_context:VexBasicBlockContext
    for vex_basic_block_context in vex_function_context.basic_block_contexts:

        size = vex_basic_block_context.end_address - vex_basic_block_context.start_address

        if 20 < size < 100:
            print(f"Basic Block Context:{vex_basic_block_context.start_address:x}")
            print(vex_basic_block_context)
            embedding = basic_block_encoder.encode([vex_basic_block_context])
            print(f"Embedding shape: {embedding.shape}")
            print(embedding)
            break


if __name__ == '__main__':
    main()


