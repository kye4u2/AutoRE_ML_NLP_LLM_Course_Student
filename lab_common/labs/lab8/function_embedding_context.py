import json
import os
from dataclasses import dataclass, field
from typing import List, Union
import torch
from tqdm import tqdm

from lab_common.labs.lab8.embedding_caption_context import EmbeddingCaptionContext
from lab_common.longformermlm.dataset_utils import load_tokens_from_jsonl




class FunctionEmbeddingCaptionContext(EmbeddingCaptionContext):

    def __init__(self, binary_id: int, caption: str, embedding: Union[torch.Tensor, List[float]], address: int):
        super().__init__(caption=caption, embedding=embedding)
        self.binary_id = binary_id
        self.address = address

    @property
    def function_name(self) -> str:
        return self.caption


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load a JSONL file or all JSONL files from a directory into FunctionEmbeddingContext instances with progress bars."
    )
    parser.add_argument("path", help="Path to the JSONL file or directory containing JSONL files.")

    #Add argument for token map path
    parser.add_argument("--token_map_path", required=True, type=str, default=None, help="Path to the token map JSONL file.")

    args = parser.parse_args()

    token_map = load_tokens_from_jsonl(args.token_map_path)

    dataset, token_map = FunctionEmbeddingCaptionContext.load_dataset(args.path, token_map=token_map)
    # Print the first few entries for inspection.
    if len(dataset) > 0:
        embedding, caption = dataset[0]
        print(f"First entry:")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Caption token IDs: {caption}")
        print(f"  Caption shape: {caption.shape}")

    # Print the token map for inspection, sorting by count.
    print(f"Token map: (#tokens={len(token_map)})")

    # Print the total size of the dataset.
    print(f"Total size of the dataset: {len(dataset)}")
    for token, data_token in sorted(token_map.items(), key=lambda x: x[1].count, reverse=True)[:20]:
        print(f"{token}: count={data_token.count} id={data_token.id}")
