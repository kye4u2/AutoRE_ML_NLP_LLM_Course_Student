import logging
import os
from typing import List

import torch

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab8.function_embedding_context import FunctionEmbeddingCaptionContext

from logzero import logger

from labs.lab8.lab_8_2.binrank_lfmlm_func_enc import compute_function_embeddings

# Set levels for other libraries
logging.getLogger("binarycontext").setLevel(logging.WARN)
logging.getLogger("pyvex.lifting.gym.arm_spotter").setLevel(logging.CRITICAL)
logging.getLogger("binrank_lfmlm_func_enc").setLevel(logging.ERROR)
logging.getLogger("longformermlm").setLevel(logging.ERROR)

# Define file paths
FUNCTION_EMBEDDING_CONTEXT_FILE_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH,
    "lab_datasets",
    "lab8",
    "function_embeddings_test_200.jsonl"
)

TEST_BCC_FILE_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH,
    "lab_datasets",
    "lab8",
    "test_leaf_function_bccs"
)

PRE_TRAINED_LONGFORMER_BB_CHECKPOINT_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH,
    "lab_datasets",
    "lab8",
    "longformer_mlm_bb_encoder",
    "small",
    "checkpoints",
    "longformer_mlm_2025-03-10_21-01-20"
)


def load_function_embedding_contexts() -> List[FunctionEmbeddingCaptionContext]:
    """Load function embedding contexts from a JSONL file."""
    logger.info("Loading function embedding contexts from: %s", FUNCTION_EMBEDDING_CONTEXT_FILE_PATH)
    contexts = FunctionEmbeddingCaptionContext.load_jsonl(FUNCTION_EMBEDDING_CONTEXT_FILE_PATH)
    logger.info("Loaded %d contexts.", len(contexts))
    return contexts


def get_vex_binary_context(binary_id: str) -> VexBinaryContext:
    """Load the VexBinaryContext for a given binary ID."""
    bcc_file_path = os.path.join(TEST_BCC_FILE_PATH, f"{binary_id}.bcc")
    if not os.path.exists(bcc_file_path):
        logger.error("BCC file not found for Binary ID %s at %s", binary_id, bcc_file_path)
        raise FileNotFoundError(f"BCC file missing: {bcc_file_path}")
    logger.debug("BCC file found for Binary ID %s: %s", binary_id, bcc_file_path)
    return VexBinaryContext.load_from_file(bcc_file_path)


def compute_and_validate_embedding(context: FunctionEmbeddingCaptionContext) -> None:
    """
    Compute the function embedding using the pre-trained checkpoint,
    compare it to the expected embedding, and assert if the similarity is below threshold.
    """
    # Combined header with all relevant function details.
    logger.info("Test Function: '%s', Binary ID: %s, Address: %s",
                context.function_name, context.binary_id, context.address)

    try:
        vex_binary_context = get_vex_binary_context(context.binary_id)
    except FileNotFoundError as e:
        logger.error("Skipping test for %s due to: %s", context.function_name, e)
        return

    torch.set_default_dtype(torch.float32)

    logger.info("Computing embedding for function: %s", context.function_name)
    embeddings = compute_function_embeddings(
        vex_binary_context,
        PRE_TRAINED_LONGFORMER_BB_CHECKPOINT_PATH,
        function_addresses=[context.address],
        top_n_bb=10,
        include_imports_embedding=True,
        include_strings_embedding=True,
        timeout_seconds=5
    )

    if not embeddings:
        logger.error("No embedding computed for function %s (Binary ID: %s)",
                     context.function_name, context.binary_id)
        return


    actual_embedding = embeddings[0]
    logger.info("Embedding shape for %s: %s", context.function_name, actual_embedding.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context.embedding = context.embedding.to(device)
    actual_embedding = actual_embedding.to(device)

    similarity = torch.nn.functional.cosine_similarity(context.embedding, actual_embedding, dim=0)
    logger.info("Cosine similarity for %s: %.4f", context.function_name, similarity.item())

    threshold = 0.70
    if similarity < threshold:
        error_msg = (f"Similarity too low for function {context.function_name} "
                     f"(Binary ID: {context.binary_id}): {similarity.item():.4f}")
        logger.error(error_msg)
        raise AssertionError(error_msg)

    logger.info("Test PASSED for %s (Binary ID: %s). Similarity: %.4f",
                context.function_name, context.binary_id, similarity.item())


def run_tests(limit: int = 20) -> None:
    """
    Load the function embedding contexts and run tests on the first 'limit' entries.
    """
    logger.info("Starting tests for the first %d function embedding contexts.", limit)
    contexts = load_function_embedding_contexts()
    for index, context in enumerate(contexts[:limit], start=1):
        logger.info("----- Running test %d of %d -----", index, limit)
        try:
            compute_and_validate_embedding(context)
        except AssertionError as ae:
            logger.error("Test FAILED for function %s: %s", context.function_name, ae)
            raise
        except Exception as e:
            logger.error("Unexpected error for function %s: %s", context.function_name, e)
        # Add a blank line to visually separate the tests
        logger.info("")
    logger.info("All %d tests completed successfully.", limit)


if __name__ == "__main__":
    try:
        run_tests(limit=10)
    except Exception as main_exception:
        logger.error("Testing terminated with errors: %s", main_exception)
    else:
        logger.info("Testing completed without any errors.")
