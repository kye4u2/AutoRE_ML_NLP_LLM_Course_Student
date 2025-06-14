# Get the parent's parent directory path
import math
import multiprocessing
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
from torch import nn

from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext

ROOT_PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANDOM_SEED = 0

MIN_FUNCTION_SIZE = 100

DEFAULT_LRU_CACHE_MAX_SIZE = 1024

DEFAULT_MALWARE_PERCENTAGE = .50

NUM_CPUS = multiprocessing.cpu_count()

DEFAULT_MIN_FUNCTION_SIZE = 25

DEFAULT_MAX_FUNCTION_SIZE = 3000

DEFAULT_DROPOUT_RATE = .20

DEFAULT_KERAS_PATIENCE = 5

IGNORE_FUNCTION_NAMES = ["sub_", "GLOBAL_", "_start", "register_tm", "__do", "frame_dummy", "operator"]



class Label(Enum):
    BENIGN = 1
    MALWARE = 2
    MATCH = 3
    NO_MATCH = 4
    NONE = 5


class EmbeddingType(Enum):
    MEDIAN_PROXIMITY = 1
    CUSTOM = 2


@dataclass
class FunctionEmbeddingContext:
    """Function embedding context."""
    embedding: np.ndarray
    vex_function_context: VexFunctionContext
    binary_name: str
    bcc_file_path: Optional[str] = None


def compute_cosine_similarity(p: np.ndarray, q: np.ndarray):
    """
    Compute the cosine similarity between the two given embeddings.

    :param p: The first embedding.
    :param q: The second embedding.
    :return: The cosine similarity between the embeddings.

    """

    if np.all(p == 0) or np.all(q == 0):
        # Return a default lowest distance if one of the embeddings has zero norm
        return 0.0

    return np.dot(p, q) / (
            np.linalg.norm(p) * np.linalg.norm(q))


def compute_cosine_distance(p: np.ndarray, q: np.ndarray):
    """Compute the cosine distance between the two given embeddings."""

    return 1 - compute_cosine_similarity(p, q)


def filter_function_contexts(vex_binary_context,
                             min_function_size=DEFAULT_MIN_FUNCTION_SIZE,
                             max_function_size=DEFAULT_MAX_FUNCTION_SIZE,
                             min_callees = 2,
                             min_string_references = 3,):
    """
    Filter the function contexts in the given VEX binary context by the given size range.

    :param vex_binary_context: The VEX binary context to filter.
    :param min_function_size: The minimum function size.
    :param max_function_size: The maximum function size.
    :param min_callees: The minimum number of callees.
    :param min_string_references: The minimum number of string references.
    :return: The filtered function contexts.
    """

    filtered_function_contexts: list[VexFunctionContext] = []

    function_context: VexFunctionContext
    for function_context in vex_binary_context.function_contexts:

        if function_context.is_thunk:
            continue

        if function_context.size < min_function_size:
            continue

        if function_context.size > max_function_size:
            continue

        if any(ignore_function_name in function_context.name
               for ignore_function_name in IGNORE_FUNCTION_NAMES):
            continue

        if len(function_context.callees) < min_callees:
            continue

        if len(function_context.string_ref_dict.values()) < min_string_references:
            continue

        filtered_function_contexts.append(function_context)


    return filtered_function_contexts

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: Dimensionality of embeddings.
        dropout: Dropout probability.
        max_len: Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix 'pe' with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def collate_fn(batch):
    """
    Collate function to combine embeddings and captions into batches.
    - Embeddings are stacked.
    - Captions are padded to the maximum length in the batch (assuming 0 is the padding index).
    """
    embeddings, captions = zip(*batch)
    embeddings = torch.stack(embeddings, dim=0)  # (batch_size, d_model)
    caption_lengths = [cap.size(0) for cap in captions]
    max_len = max(caption_lengths)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :cap.size(0)] = cap
    return embeddings, padded_captions