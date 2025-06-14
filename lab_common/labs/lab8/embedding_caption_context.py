import argparse
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from logzero import setup_logger

from lab_common.longformermlm.dataset_utils import DataToken
from lab_common.nlp.tokenizer import Tokenizer

SPECIAL_TOKENS = {"[SOS]", "[PAD]", "[UNK]", "[CLS]", "[MASK]", "[EOS]"}

logger = setup_logger(name=Path(__file__).stem, level="INFO")


class EmbeddingCaptionDataset(Dataset):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Args:
            data (List[Tuple[torch.Tensor, torch.Tensor]]):
                A list where each element is a tuple:
                  - embedding: torch.Tensor of shape (d_model,)
                  - caption: torch.Tensor of token IDs, shape (seq_len,)
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    def save_to_jsonl(self, output_path: str):
        """
        Save the dataset to a JSONL file.

        Args:
            output_path (str): Path to save the JSONL file.
        """
        with open(output_path, 'w') as f:
            for embedding, caption in self.data:
                entry = {
                    "embedding": embedding.tolist(),
                    "caption": caption.tolist()
                }
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Saved {len(self.data)} instances to {output_path}")

    @classmethod
    def load_from_jsonl(cls, path: str) -> "EmbeddingCaptionDataset":
        """
        Load a JSONL file and return an EmbeddingCaptionDataset.

        Args:
            path (str): Path to the JSONL file.

        Returns:
            EmbeddingCaptionDataset: A dataset where each item is a tuple:
                - embedding: torch.Tensor of shape (d_model,)
                - caption: torch.Tensor of token IDs, shape (seq_len,)
        """
        data = []
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                embedding = torch.tensor(entry["embedding"], dtype=torch.float)
                caption = torch.tensor(entry["caption"], dtype=torch.long)
                data.append((embedding, caption))
        return cls(data)


class EmbeddingCaptionContext:
    def __init__(self, embedding: Union[torch.Tensor, List[float]], caption: str):

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float)

        self._embedding = embedding

        self._caption = caption

    @property
    def embedding(self) -> torch.Tensor:

        return self._embedding

    @embedding.setter
    def embedding(self, value: Union[torch.Tensor, List[float]]):
        # Convert to torch.Tensor if necessary.
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float)
        self._embedding = value

    @property
    def caption(self) -> str:
        return self._caption

    @caption.setter
    def caption(self, value: str):
        self._caption = value

    def to_dict(self) -> dict:
        """
        Convert the instance into a dictionary including all attributes.
        If an attribute is a torch.Tensor, it is converted to a list.
        This method will capture any additional attributes defined by child classes.
        """
        out = {}
        for key, value in self.__dict__.items():
            # If the attribute is stored with a leading underscore but there's a property,
            # we output it using the public name.
            public_key = key[1:] if key.startswith("_") else key
            if isinstance(value, torch.Tensor):
                out[public_key] = value.tolist()
            else:
                out[public_key] = value
        return out

    def to_jsonl_entry(self) -> str:
        """
        Serialize the instance to a JSON string (suitable for JSONL files)
        using the dictionary representation.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json_entry(cls, entry: Union[str, dict]) -> "EmbeddingCaptionContext":
        """
        Create an instance from a JSON string or a dictionary.
        The keys of the dictionary should match the constructor parameters.
        Unexpected keys will be ignored.
        Private attributes should match even if provided without the leading '_'.

        """
        if isinstance(entry, str):
            entry = json.loads(entry)

        # Get all attribute names defined in __init__ by inspecting cls.__init__.__code__.co_varnames
        init_params = cls.__init__.__code__.co_varnames[1:]  # Skip 'self'

        # Create a mapping where private attributes (_attr) are also accessible without '_'
        attr_map = {k.lstrip("_"): k for k in init_params}

        # Filter entry keys: match either exact attribute names or their private versions
        filtered_entry = {
            attr_map.get(k, k): v for k, v in entry.items() if k in attr_map
        }

        return cls(**filtered_entry)

    @classmethod
    def load_jsonl(cls, path: str) -> List["EmbeddingCaptionContext"]:
        """
        Load a JSONL file or all JSONL files in a folder and return a list of
        EmbeddingCaptionContext instances. Each non-empty line in a file is
        treated as a JSON entry.
        """
        entries = []
        if os.path.isdir(path):
            # If the path is a directory, iterate over each file ending with '.jsonl'
            for filename in os.listdir(path):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(path, filename)
                    entries.extend(cls.load_jsonl(file_path))
        elif os.path.isfile(path):
            # Process the single JSONL file.
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        instance = cls.from_json_entry(line)
                        entries.append(instance)
                    except Exception as e:
                        logger.exception(f"Error processing line: {line}\nError: {e}")
        else:
            raise ValueError(f"Provided path '{path}' is neither a file nor a directory.")
        return entries

    @classmethod
    def load_embedding_only_dataset(cls, path: str) -> EmbeddingCaptionDataset:
        """
        Load an embedding-only dataset from a JSONL file.

        This method reads a JSONL file containing embedding contexts (instances of EmbeddingCaptionContext)
        and constructs an EmbeddingCaptionDataset. For each loaded context, the method uses only the embedding
        and replaces the caption with a zero vector. The zero vector is created as a 1-dimensional tensor of zeros,
        with dtype torch.long.

        Args:
            path (str): Path to the JSONL file containing the embedding contexts.

        Returns:
            EmbeddingCaptionDataset: A dataset where each entry is a tuple:
                - embedding: torch.Tensor (from the loaded context)
                - caption: torch.Tensor, a zero vector placeholder (shape: [1])
        """
        # Load embedding contexts from the JSONL file using the existing load_jsonl method.
        embedding_caption_contexts: List["EmbeddingCaptionContext"] = cls.load_jsonl(path)

        # For each loaded context, keep the embedding and substitute the caption with a zero vector.
        data = [
            (context.embedding, torch.zeros(1, dtype=torch.long))
            for context in embedding_caption_contexts
        ]

        # Return an EmbeddingCaptionDataset constructed with the new data.
        return EmbeddingCaptionDataset(data)

    @classmethod
    def load_dataset(
            cls,
            path: str,
            token_map: Dict[str, DataToken] = None,
            allow_token_addition: Optional[bool] = None,
            max_vocab_size: int = 0,
            seq_len: int = 10,
            min_token_length: int = 3,
            min_tokens_in_caption: int = 3

    ) -> tuple[EmbeddingCaptionDataset, dict[str, DataToken] | dict]:
        """
        Load JSONL entries from a file or folder, convert each instance to a tuple of
        (embedding, caption_token_ids), and return an EmbeddingCaptionDataset.

        Args:
            path (str): Path to a JSONL file or a folder containing JSONL files.
            tokenizer_file_path (str): Path to the tokenizer file.
            token_map (dict, optional): A dictionary mapping tokens to DataToken instances.
            allow_token_addition (bool, optional): If True, new tokens encountered will be added to the vocabulary.
            max_vocab_size (int): Maximum number of tokens to retain in the vocabulary.
            seq_len (int): The fixed length each caption’s token sequence should have.
            min_token_length (int): Minimum length of a token to be included in the vocabulary. If 0, no minimum length is enforced.
            min_tokens_in_caption (int): Minimum number of tokens in a caption for it to be included in the dataset.

        Returns:
            EmbeddingCaptionDataset: A dataset where each item is a tuple:
                - embedding: torch.Tensor of shape (d_model,)
                - caption: torch.Tensor of token IDs, shape (seq_len,)
            dict: A dictionary mapping tokens to DataToken instances.
        """
        embedding_caption_contexts: List["EmbeddingCaptionContext"] = cls.load_jsonl(path)

        embedding_caption_dataset, token_map = load_dataset_from_embedding_caption_contexts(embedding_caption_contexts,
                                                                                            token_map,
                                                                                            allow_token_addition,
                                                                                            max_vocab_size,
                                                                                            seq_len,
                                                                                            min_token_length,
                                                                                            min_tokens_in_caption)

        return embedding_caption_dataset, token_map


def load_dataset_from_embedding_caption_contexts(
        embedding_caption_contexts: List["EmbeddingCaptionContext"],
        token_map: dict = None,
        allow_token_addition: Optional[bool] = None,
        max_vocab_size: int = 0,
        seq_len: int = 10,
        min_token_length: int = 3,
        min_tokens_in_caption: int = 3
) -> tuple[EmbeddingCaptionDataset, dict[str, DataToken] | dict]:
    """
    Load a dataset from a list of EmbeddingCaptionContext instances.

    For each caption, the tokenizer's tokenize() method is used to get a list of tokens.
    Each unique token is assigned a unique ID while tracking its occurrence count.


    After processing all data points, if max_vocab_size > 0 the vocabulary is pruned:
        - Only the top max_vocab_size tokens (by frequency) are retained.
        - Special tokens (i.e. "[PAD]", "[UNK]", "[CLS]", "[MASK]", "[EOS]") are always kept.
        - Tokens not present in the pruned vocabulary are removed from each data point's token sequence.
        - Data points with no tokens left after pruning are discarded.

    Finally, each data point’s token sequence is normalized so that:
        - The final token is always the EOS token.
        - If the sequence is longer than seq_len, it is truncated (keeping EOS at the end).
        - If the sequence is shorter than seq_len, it is padded with [PAD] tokens (before EOS)
          so that the final sequence has length seq_len.

    Args:
        embedding_caption_contexts (List[EmbeddingCaptionContext]):
            List of embedding-caption contexts.
        token_map (dict, optional):
            Mapping from token strings to DataToken instances. If None, a new vocabulary is created.
        allow_token_addition (bool, optional):
            If True, new tokens are added to the vocabulary; if False, unknown tokens map to [UNK].
        max_vocab_size (int):
            Maximum number of tokens to retain in the vocabulary. If 0, all tokens are kept.
        seq_len (int):
            The fixed length each caption’s token sequence should have.
        min_token_length (int):
            Minimum length of a token to be included in the vocabulary. If 0, no minimum length is enforced.
        min_tokens_in_caption (int):
            Minimum number of tokens in a caption for it to be included in the dataset.

    Returns:
        tuple:
            - EmbeddingCaptionDataset: Each item is a tuple of:
                  (embedding: torch.Tensor of shape (d_model,),
                   caption: torch.Tensor of token IDs, shape (seq_len,))
            - dict: Final mapping of tokens to DataToken instances.
    """
    tokenizer = Tokenizer()

    # Initialize vocabulary with default tokens if not provided.
    if token_map is None:
        # Initialize the token map with the SPECIAL_TOKENS.
        token_map = {token: DataToken(i, 0) for i, token in enumerate(SPECIAL_TOKENS)}
        allow_token_addition = True
        next_id = len(SPECIAL_TOKENS)  # Next available token ID after default tokens.
    else:
        # Make sure the special tokens are present in the token map.
        for special_token in SPECIAL_TOKENS:
            if special_token not in token_map:
                raise ValueError(f"Special token '{special_token}' is missing from the token map.")
        # If token_map is provided, decide on token addition if not explicitly specified.
        if allow_token_addition is None:
            allow_token_addition = False
        if token_map:
            next_id = max(token.id for token in token_map.values()) + 1
        else:
            next_id = len(SPECIAL_TOKENS)

    def get_token_id(token: str) -> int:
        nonlocal next_id
        if token in token_map:
            token_map[token].count += 1
            return token_map[token].id
        else:
            if allow_token_addition:
                token_map[token] = DataToken(next_id, 1)
                token_id = next_id
                next_id += 1
                return token_id
            else:
                # If token addition is not allowed, unknown tokens are mapped to [UNK].
                return token_map["[UNK]"].id

    # Process each embedding-caption context to create the initial dataset.
    data = []
    for embedding_caption_context in tqdm(embedding_caption_contexts, desc="Processing instances"):
        embedding = embedding_caption_context.embedding  # torch.Tensor of shape (d_model,)
        caption_text = embedding_caption_context.caption  # Caption string

        # Tokenize the caption.
        tokens = tokenizer.tokenize(caption_text)

        # Filter out tokens that are too short.
        tokens = [token for token in tokens if len(token) >= min_token_length or min_token_length == 0]

        # Skip the data point if no tokens remain.
        if len(tokens) < min_tokens_in_caption:
            continue

        # Add the SOS token at the beginning of the caption.
        tokens = ["[SOS]"] + tokens

        token_ids = [get_token_id(token) for token in tokens]
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        data.append((embedding, token_ids_tensor))

    # If a maximum vocabulary size is specified, prune the vocabulary.
    if max_vocab_size > 0:
        # Separate special tokens from non-special tokens.
        special_token_items = {token: dt for token, dt in token_map.items() if token in SPECIAL_TOKENS}
        non_special_tokens = [(token, dt) for token, dt in token_map.items() if token not in SPECIAL_TOKENS]

        # Sort non-special tokens by frequency (highest count first).
        non_special_tokens_sorted = sorted(non_special_tokens, key=lambda x: x[1].count, reverse=True)

        # Calculate the number of non-special tokens to keep.
        num_to_keep = max_vocab_size - len(special_token_items)
        if num_to_keep < 0:
            num_to_keep = 0  # In case max_vocab_size is very small.

        # Retain the top non-special tokens.
        allowed_non_special = {token: dt for token, dt in non_special_tokens_sorted[:num_to_keep]}

        # Combine special tokens with the selected non-special tokens.
        pruned_token_map = {**special_token_items, **allowed_non_special}

        # Build a set of allowed token IDs.
        allowed_ids = {dt.id for dt in pruned_token_map.values()}

        # For each data point, filter out tokens not in the pruned vocabulary.
        pruned_data = []
        for embedding, token_ids_tensor in data:
            token_ids_list = token_ids_tensor.tolist()
            filtered_token_ids = [tid for tid in token_ids_list if tid in allowed_ids]

            # Skip the data point if no tokens remain.
            if not filtered_token_ids:
                continue

            special_token_ids = [ data_token.id for data_token in special_token_items.values()]

            # Compute the number of tokens that are non-special tokens
            num_non_special_tokens = len([tid for tid in filtered_token_ids if tid not in special_token_ids])

            if num_non_special_tokens < min_tokens_in_caption:
                continue


            # Append the EOS token if it is not already the last token.
            eos_token_id_local = token_map["[EOS]"].id
            if filtered_token_ids[-1] != eos_token_id_local:
                filtered_token_ids.append(eos_token_id_local)

            pruned_data.append((embedding, torch.tensor(filtered_token_ids, dtype=torch.long)))
        data = pruned_data

        # --- Reassign token IDs to be contiguous from 0 to (vocab_size - 1) ---
        # We'll create a new token_map with new sequential IDs and update all data sequences.

        old_token_map = pruned_token_map
        new_token_map = {}
        id_remap = {}
        new_id = 0

        # Ensure special tokens get the first IDs in the order defined in SPECIAL_TOKENS.
        for token in SPECIAL_TOKENS:
            if token in old_token_map:
                old_dt = old_token_map[token]
                new_token_map[token] = DataToken(new_id, old_dt.count)
                id_remap[old_dt.id] = new_id
                new_id += 1

        # Reassign IDs for remaining tokens.
        for token, dt in old_token_map.items():
            if token not in SPECIAL_TOKENS:
                new_token_map[token] = DataToken(new_id, dt.count)
                id_remap[dt.id] = new_id
                new_id += 1

        token_map = new_token_map

        # Update token IDs in the dataset to use the new IDs.
        remapped_data = []
        for embedding, token_ids_tensor in data:
            token_ids_list = token_ids_tensor.tolist()
            new_ids = [id_remap[tid] for tid in token_ids_list if tid in id_remap]
            remapped_data.append((embedding, torch.tensor(new_ids, dtype=torch.long)))
        data = remapped_data

    # Normalize all sequences to have exactly seq_len tokens.
    pad_token_id = token_map["[PAD]"].id
    eos_token_id = token_map["[EOS]"].id

    def normalize_sequence(token_ids: List[int], seq_len: int, eos_token_id: int, pad_token_id: int) -> List[int]:
        """
        Ensure that:
          - The sequence ends with the EOS token.
          - If the sequence is too long, it is truncated so that the EOS is the last token.
          - If the sequence is too short, it is padded (with [PAD]) before the EOS so that its total length is seq_len.
        """
        # Ensure the sequence ends with EOS.
        if not token_ids or token_ids[-1] != eos_token_id:
            token_ids = token_ids + [eos_token_id]

        # Truncate if the sequence is too long.
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len - 1] + [eos_token_id]
        # Pad if the sequence is too short.
        elif len(token_ids) < seq_len:
            # Temporarily remove EOS.
            token_ids = token_ids[:-1]
            pad_count = seq_len - (len(token_ids) + 1)
            token_ids = token_ids + [pad_token_id] * pad_count + [eos_token_id]
        return token_ids

    normalized_data = []
    for embedding, token_ids_tensor in data:
        token_ids_list = token_ids_tensor.tolist()
        norm_ids = normalize_sequence(token_ids_list, seq_len, eos_token_id, pad_token_id)
        normalized_data.append((embedding, torch.tensor(norm_ids, dtype=torch.long)))
    data = normalized_data

    return EmbeddingCaptionDataset(data), token_map


def main():
    parser = argparse.ArgumentParser(
        description="Load a JSONL file or folder containing JSONL files into an EmbeddingCaptionDataset."
    )
    parser.add_argument("path", help="Path to a JSONL file or folder containing JSONL files.")

    args = parser.parse_args()

    dataset, token_map = EmbeddingCaptionContext.load_dataset(args.path)
    print(f"Loaded dataset with {len(dataset)} entries.")

    if len(dataset) > 0:
        embedding, caption = dataset[0]
        print(f"First entry:")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Caption token IDs: {caption}")
        print(f"  Caption shape: {caption.shape}")


if __name__ == "__main__":
    main()
