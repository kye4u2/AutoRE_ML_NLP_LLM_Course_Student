import argparse
import enum
import os.path
import re
import logging
from typing import Optional

import torch
import numpy as np
from numpy.ma.extras import average, median
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import tiktoken  # OpenAI's tokenization library

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab10.function_data_processor import FunctionDataSet, FunctionData
from lab_common.llm.llm_common import get_llm_client_config
from labs.lab7.sentence_longformermlm import SentenceLongformerMLM

# Configure logging to show warnings.
logging.basicConfig(level=logging.WARNING)

# Initialize the OpenAI client only once (assumes OPENAI_API_KEY is set)
api_key = os.environ.get("OPENAI_API_KEY",get_llm_client_config().api_key)
client = OpenAI(api_key=api_key)

SENTENCE_SUMMARIZER_CHECKPOINT_DIR = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets/lab10/sentence_longformer_mlm/longformer_mlm_2025-03-14_16-33-56")

FUNC_SUMMARY_TEST_DATASET_FILE_PATH = os.path.join(
    ROOT_PROJECT_FOLDER_PATH, "lab_datasets/lab10/mini/function_data_summaries_mini_test.jsonl"
)


class QuantizationMode(enum.Enum):
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"
    NOT_QUANT = "not quant"


def compute_embedding_similarity(text1: str, text2: str, model: str = "text-embedding-ada-002") -> float:
    """
    Computes the cosine similarity between embeddings of two texts using OpenAI's API.
    """
    embedding1 = client.embeddings.create(input=text1, model=model).data[0].embedding
    embedding2 = client.embeddings.create(input=text2, model=model).data[0].embedding

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def truncate_summary_if_needed(summary: str, max_tokens: int = 500, use_tiktoken: bool = False,
                               transformers_tokenizer: Optional[AutoTokenizer] = None) -> str:
    """
    Truncates the summary if it exceeds max_tokens.

    If use_tiktoken is True, the tiktoken library (with the model "text-embedding-ada-002")
    is used to count tokens. Otherwise, the provided transformers tokenizer is used.

    Logs a warning if truncation occurs.
    """
    if use_tiktoken:
        encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
        tokens = encoding.encode(summary)
        token_count = len(tokens)
        if token_count > max_tokens:
            logging.warning("Generated summary exceeds %d tokens (actual: %d). Truncating summary.",
                            max_tokens, token_count)
            tokens = tokens[:max_tokens]
            summary = encoding.decode(tokens)
        return summary
    else:
        if transformers_tokenizer is None:
            raise ValueError("A transformers tokenizer must be provided if not using tiktoken.")
        tokens = transformers_tokenizer.encode(summary, add_special_tokens=False)
        token_count = len(tokens)
        if token_count > max_tokens:
            logging.warning("Generated summary exceeds %d tokens (actual: %d). Truncating summary.",
                            max_tokens, token_count)
            tokens = tokens[:max_tokens]
            summary = transformers_tokenizer.decode(tokens)
        return summary


def strip_function_name_references(func_data: FunctionData) -> FunctionData:
    """
    Replaces references to the function's own name in the decompiled code with its function address.
    """
    pattern = r'\b' + re.escape(func_data.function_name) + r'\b'
    replacement = f"func_{hex(func_data.function_address)}"
    modified_code = re.sub(pattern, replacement, func_data.decompiled_code)
    modified_code = re.sub(r'\s+', ' ', modified_code).strip()
    func_data.decompiled_code = modified_code
    return func_data





def parse_quantization_mode(value: str) -> QuantizationMode:
    try:
        return QuantizationMode(value)
    except ValueError:
        valid_modes = [e.value for e in QuantizationMode]
        raise argparse.ArgumentTypeError(
            f"Invalid quantization mode: {value}. Choose from: {valid_modes}"
        )


def run_inference(jsonl_file: str,
                  max_data_points: int,
                  adapter_path: Optional[str],
                  quantization: QuantizationMode,
                  sentence_longformermlm: SentenceLongformerMLM,
                  use_openai_similarity: bool,
                  max_tokens: int = 500):
    similarities = []
    ### YOUR CODE HERE ###


    ### END OF YOUR CODE ###


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference on function data to summarize decompiled code."
    )
    parser.add_argument("--input", type=str, default=FUNC_SUMMARY_TEST_DATASET_FILE_PATH, help="Path to the input JSONL file.")
    parser.add_argument("--max_data_points", type=int, default=50, help="Maximum number of data points to process.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/opt/LLaMA-Factory/saves/Llama-3.2-1B-Instruct/lora/train_2025-03-20-11-43-16/",
        help="Path to the LoRA adapter. Set to empty string to skip using an adapter."
    )
    parser.add_argument("-d", "--disable_adapter", action="store_true", help="Disable the LoRA adapter.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=SENTENCE_SUMMARIZER_CHECKPOINT_DIR,
                        help="Path to the checkpoint directory.")
    parser.add_argument(
        "-q","--quantization",
        type=parse_quantization_mode,
        choices=list(QuantizationMode),
        default=QuantizationMode.NOT_QUANT,
        help="Quantization option: '4bit', '8bit', or 'not quant' for no quantization."
    )
    parser.add_argument("--use_openai_similarity", action="store_true",
                        help="Use OpenAI embeddings to compute cosine similarity instead of the built-in method.")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens allowed in the generated summary. Summaries longer than this will be truncated.")
    args = parser.parse_args()

    if args.disable_adapter:
        args.adapter_path = None

    sentence_longformermlm, _ = SentenceLongformerMLM.from_checkpoint(
        args.checkpoint_dir,
        model_filename="model_best_val_loss.pt",
    )

    # If using openai embedding to compute similarity , make sure the OPENAI_API_KEY is set
    if args.use_openai_similarity:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set to use OpenAI similarity.")

    run_inference(
        jsonl_file=args.input,
        max_data_points=args.max_data_points,
        adapter_path=args.adapter_path,
        quantization=args.quantization,
        sentence_longformermlm=sentence_longformermlm,
        use_openai_similarity=args.use_openai_similarity,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
