import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from accelerate import Accelerator

from logzero import setup_logger

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer
from labs.lab4.binary_rank import BinaryRankContext, BinaryRankTimeoutError, compute_global_import_ranks, \
    compute_uniform_import_ranks, compute_median_proximity_weights, compute_uniform_strings_ranks, \
    compute_global_strings_ranks, FunctionRankContext, BasicBlockRankContext
from labs.lab8.lab_8_1.longformer_mlm_bb_encoder import LongformerMLMBasicBlockEncoder

logger = setup_logger(name=Path(__file__).stem, level="INFO")

DEFAULT_TOP_N_BB = 10  # Number of top-ranked basic blocks to consider for each function

DEFAULT_TIMEOUT_SECONDS = 60  # Default timeout in seconds for the encoder


class BinaryRankLFMLMFunctionEncoder(object):
    """Binary Rank LongFormer MLM Function Encoder """

    def __init__(self,
                 longformer_bb_encoder: LongformerMLMBasicBlockEncoder,
                 top_n_bb: int = DEFAULT_TOP_N_BB,
                 include_imports_embedding: bool = False,
                 include_strings_embedding: bool = False,
                 sentence_summarizer: Optional[SentenceSummarizer] = None,
                 timeout_seconds: Optional[float] = None
                 ):
        self._longformer_bb_encoder = longformer_bb_encoder
        self._top_n_bb = top_n_bb  # Number of top-ranked basic blocks to consider for each function

        self._include_imports_embedding = include_imports_embedding  # Include imports in the function embedding

        self._include_strings_embedding = include_strings_embedding  # Include strings in the function embedding

        self._sentence_summarizer: Optional[SentenceSummarizer] = sentence_summarizer  # Sentence summarizer for imports

        self._binary_rank_context: Optional[BinaryRankContext] = None  # Binary rank context for function encoding

        self._accelerator = Accelerator()

        self._timeout_seconds = timeout_seconds

        self._unwrapped_model = self._accelerator.unwrap_model(self._longformer_bb_encoder)

    def generate_imports_embedding(self,
                                   vex_binary_context: VexBinaryContext,
                                   timeout_seconds: Optional[float] = None) -> torch.Tensor:
        """
        Generate an embedding for the imports section of the binary using a weighted average.

        Parameters:
        - vex_binary_context (VexBinaryContext): Binary context containing function information.

        Returns:
        - torch.Tensor: The weighted average of the import embeddings.
        """
        if timeout_seconds is None:
            timeout_seconds = self._timeout_seconds

        # Set deadline for the encoder
        start_time = time.time()
        deadline = start_time + timeout_seconds if timeout_seconds is not None else float('inf')

        if self._sentence_summarizer is None:
            self._sentence_summarizer = SentenceSummarizer()
            self._sentence_summarizer.initialize()

        timeout_seconds = deadline - time.time()
        if self._binary_rank_context is None:

            try:
                self._binary_rank_context = BinaryRankContext.from_vex_binary_context(vex_binary_context,
                                                                                      timeout_seconds=timeout_seconds)
            except BinaryRankTimeoutError as e:
                logger.warning(
                    f"Binary rank computation timed out ({timeout_seconds}s) for {vex_binary_context.name} {vex_binary_context.sha256_hash}: {e}")
                self._binary_rank_context = None
            except Exception as e:
                logger.error(
                    f"Error computing binary rank for {vex_binary_context.name} {vex_binary_context.sha256_hash}: {e}")
                self._binary_rank_context = None

        if self._binary_rank_context is None:
            # Will assume uniform weights if no binary rank context is available
            logger.warning("No binary rank context available. Using uniform import ranks.")
            import_rank_dict: Dict[str, float] = compute_uniform_import_ranks(vex_binary_context)

        else:
            # Compute the initial import ranks from the binary context.
            import_rank_dict: Dict[str, float] = compute_global_import_ranks(self._binary_rank_context)

            # Compute the weights based on median proximity.
        weights = compute_median_proximity_weights(list(import_rank_dict.values()))

        # Check if the weights are None (no valid ranks found).
        if weights is None:
            return torch.zeros(self._sentence_summarizer.vector_size)

        # Update the dictionary with the new weights (optional if you wish to keep the dict updated)
        for import_name, weight in zip(import_rank_dict.keys(), weights):
            import_rank_dict[import_name] = weight

        # Compute weighted embeddings and the total weight.
        weighted_embeddings = []
        for import_name, weight in zip(import_rank_dict.keys(), weights):

            remaining_time = deadline - time.time()

            if remaining_time <= 0:
                logger.warning("Timeout reached while summarizing imports. Returning zero tensor.")
                return torch.zeros(self._sentence_summarizer.vector_size)

            # Get the embedding for the import as a torch.Tensor.
            embedding = self._sentence_summarizer.summarize(import_name)
            weighted_embeddings.append(embedding * weight)

        total_weight = sum(weights)

        # If there are no embeddings or the total weight is zero, return a zero tensor.
        if not weighted_embeddings or total_weight == 0:
            # Assuming the embedding dimension is defined in the sentence summarizer.
            return torch.zeros(self._sentence_summarizer.vector_size)

        weighted_embeddings = np.stack(weighted_embeddings, axis=0)

        weighted_embeddings = torch.from_numpy(weighted_embeddings)

        # Sum the weighted embeddings along the first dimension.
        sum_weighted = weighted_embeddings.sum(dim=0)

        import_embedding = sum_weighted / total_weight

        # Return the weighted average.
        return import_embedding

    def generate_strings_embedding(self,
                                   vex_binary_context: VexBinaryContext,
                                   timeout_seconds: Optional[float] = None) -> torch.Tensor:
        """
        Generate an embedding for the strings section of the binary using a weighted average.
        """
        if timeout_seconds is None:
            timeout_seconds = self._timeout_seconds

        # Set deadline for the encoder
        start_time = time.time()
        deadline = start_time + timeout_seconds if timeout_seconds is not None else float('inf')

        if self._sentence_summarizer is None:
            self._sentence_summarizer = SentenceSummarizer()
            self._sentence_summarizer.initialize()

        timeout_seconds = deadline - time.time()
        if self._binary_rank_context is None:
            try:
                self._binary_rank_context = BinaryRankContext.from_vex_binary_context(vex_binary_context,
                                                                                      timeout_seconds=timeout_seconds)
            except BinaryRankTimeoutError as e:
                logger.warning(
                    f"Binary rank computation timed out ({timeout_seconds}s) for {vex_binary_context.name} {vex_binary_context.sha256_hash}: {e}")
                self._binary_rank_context = None
            except Exception as e:
                logger.error(
                    f"Error computing binary rank for {vex_binary_context.name} {vex_binary_context.sha256_hash}: {e}")
                self._binary_rank_context = None

        if self._binary_rank_context is None:
            # Will assume uniform weights if no binary rank context is available
            strings_rank_dict: Dict[str, float] = compute_uniform_strings_ranks(vex_binary_context)
        else:
            # Compute the initial strings ranks from the binary context.
            strings_rank_dict: Dict[str, float] = compute_global_strings_ranks(self._binary_rank_context)

            # Compute the weights based on median proximity.
        weights = compute_median_proximity_weights(list(strings_rank_dict.values()))

        # Check if the weights are None (no valid ranks found).
        if weights is None:
            return torch.zeros(self._sentence_summarizer.vector_size)

        # Update the dictionary with the new weights (optional if you wish to keep the dict updated)
        for string_name, weight in zip(strings_rank_dict.keys(), weights):
            strings_rank_dict[string_name] = weight

        # Compute weighted embeddings and the total weight.
        weighted_embeddings = []
        for string_name, weight in zip(strings_rank_dict.keys(), weights):
            # Get the embedding for the string as a torch.Tensor.

            remaining_time = deadline - time.time()

            if remaining_time <= 0:
                logger.warning("Timeout reached while summarizing strings. Returning zero tensor.")
                return torch.zeros(self._sentence_summarizer.vector_size)

            embedding = self._sentence_summarizer.summarize(string_name)
            weighted_embeddings.append(embedding * weight)

        total_weight = sum(weights)

        # If there are no embeddings or the total weight is zero, return a zero tensor.
        if not weighted_embeddings or total_weight == 0:
            # Assuming the embedding dimension is defined in the sentence summarizer.
            return torch.zeros(self._sentence_summarizer.vector_size)

        weighted_embeddings = np.stack(weighted_embeddings, axis=0)

        weighted_embeddings = torch.from_numpy(weighted_embeddings)

        # Sum the weighted embeddings along the first dimension.
        sum_weighted = weighted_embeddings.sum(dim=0)

        strings_embedding = sum_weighted / total_weight

        # Return the weighted average.

        return strings_embedding

    def encode(self,
               vex_binary_context: VexBinaryContext,
               function_addresses: List[int],
               timeout_seconds: Optional[float] = None) -> List[torch.Tensor]:
        """
        Encode the function context into a Longformer MLM input using only the top_n basic blocks.

        NOTE: The ranking of basic blocks is computed **before** encoding to avoid unnecessary computation,
        ensuring that only the top-ranked basic blocks are processed for efficiency.

        Parameters:
        - vex_binary_context (VexBinaryContext): Binary context containing function information.
        - function_addresses (List[int]): List of function addresses to encode.
        - top_n_bb (int): Number of top-ranked basic blocks to consider for each function.
        - timeout_seconds (Optional[float]): Timeout in seconds for the encoder.

        Returns:
        - List[torch.Tensor]: A list of function embeddings, preserving order.
        """
        start_time = time.time()
        function_embeddings: List[torch.Tensor] = []

        d_model = self._unwrapped_model.d_model

        logger.info("Starting function encoding...")

        imports_embedding = None
        if self._include_imports_embedding:
            # Generate the imports embedding
            imports_embedding = self.generate_imports_embedding(vex_binary_context, timeout_seconds=timeout_seconds)

        strings_embedding = None
        if self._include_strings_embedding:
            # Generate the strings embedding
            time_remaining = timeout_seconds - (time.time() - start_time)
            strings_embedding = self.generate_strings_embedding(vex_binary_context, timeout_seconds=time_remaining)

        for function_address in function_addresses:

            ### YOUR CODE HERE ###


            ### END YOUR CODE HERE ###
            pass

        return function_embeddings


def compute_function_embeddings(vex_binary_context,
                                checkpoint_folder_path: str,
                                function_addresses: list[int],
                                top_n_bb: int,
                                include_imports_embedding: bool = False,
                                include_strings_embedding: bool = False,
                                timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS) -> List[torch.Tensor]:
    """
    Computes function embedding for a list of function addresses.

    Args:
        vex_binary_context: The VexBinaryContext instance loaded from a BCC file.
        checkpoint_folder_path (str): Path to the encoder checkpoint folder.
        function_addresses (list[int]): List of function addresses to encode.
        top_n_bb (int): Number of top-ranked basic blocks to consider for each function.
        include_imports_embedding (bool): Whether to include the imports embedding in the function encoding.
        include_strings_embedding (bool): Whether to include the strings embedding in the function encoding.
        timeout_seconds (Optional[float]): Timeout in seconds for the encoder.

    Returns:
        list: The list of computed function encodings.
    """
    start_time = time.time()

    # Initialize the basic block encoder from the checkpoint
    basic_block_encoder, token_map = LongformerMLMBasicBlockEncoder.from_checkpoint(checkpoint_folder_path)

    remaining_time = timeout_seconds - (time.time() - start_time)

    # Initialize the function encoder with the specified top_n_bb parameter
    function_encoder = BinaryRankLFMLMFunctionEncoder(basic_block_encoder,
                                                      top_n_bb,
                                                      include_imports_embedding=include_imports_embedding,
                                                      include_strings_embedding=include_strings_embedding,
                                                      timeout_seconds=remaining_time)

    remaining_time = timeout_seconds - (time.time() - start_time)
    encodings = function_encoder.encode(vex_binary_context,
                                        function_addresses,
                                        timeout_seconds=remaining_time)
    elapsed_time = time.time() - start_time

    return encodings


def main():
    default_bcc_file = os.path.join(
        ROOT_PROJECT_FOLDER_PATH,
        "Blackfyre",
        "test",
        "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc"
    )

    "/opt/AutoRE_ML_NLP_LLM_Course/lab_datasets/lab8/transformer_decoder_2025-03-14_20-54-53"

    default_checkpoint_dir = os.path.join(
        ROOT_PROJECT_FOLDER_PATH,
        "lab_datasets",
        "lab8",
        "longformer_mlm_bb_encoder",
        "small",
        "checkpoints",
        "longformer_mlm_2025-03-10_21-01-20"
    )

    parser = argparse.ArgumentParser(
        description="Compute function encodings using a basic block encoder."
    )
    parser.add_argument(
        "--bcc_file",
        type=str,
        default=default_bcc_file,
        help="Path to the BCC file."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=default_checkpoint_dir,
        help="Path to the checkpoint folder."
    )
    parser.add_argument(
        "--top_n_bb",
        type=int,
        default=DEFAULT_TOP_N_BB,
        help="Number of top-ranked basic blocks to consider for each function."
    )
    # Add an optional flag to include the imports embedding
    parser.add_argument(
        "--include_imports_embedding",
        action="store_true",
        help="Include the imports embedding in the function encoding."
    )

    # Add an optional flag to include the strings embedding
    parser.add_argument(
        "--include_strings_embedding",
        action="store_true",
        help="Include the strings embedding in the function encoding."
    )

    args = parser.parse_args()

    # Load the VexBinaryContext from the specified BCC file
    vex_binary_context = VexBinaryContext.load_from_file(args.bcc_file)

    # Create a list of function addresses by filtering the function contexts
    # function_addresses = [
    #     fc.start_address for fc in vex_binary_context.function_contexts
    #     if 20 < fc.size < 50
    # ]
    FUNCTION_ADDRESS = 0x2986c

    # Compute and retrieve the function encodings
    encodings = compute_function_embeddings(vex_binary_context,
                                            args.checkpoint_dir,
                                            [FUNCTION_ADDRESS],
                                            args.top_n_bb,
                                            include_imports_embedding=args.include_imports_embedding,
                                            include_strings_embedding=args.include_strings_embedding,
                                            timeout_seconds=DEFAULT_TIMEOUT_SECONDS)

    # Print the shape of the first encoding
    print(f"Function encoding shape: {encodings[0].shape}")
    print(f"Function encoding: {encodings[0]}")


if __name__ == '__main__':
    main()
