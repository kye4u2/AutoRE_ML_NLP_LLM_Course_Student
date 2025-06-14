import json
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, random_split

from dataclasses import dataclass, field

import json
import numpy as np
from typing import Any, List, Optional


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], max_seq_len: int):
        """
        A generic dataset for handling sequence data with optional truncation and padding.

        Args:
            sequences (List[np.ndarray]): A list of sequences (e.g., tokenized code, text, or other sequential data).
            max_seq_len (int): The maximum sequence length (including padding).
        """
        self.max_length = max_seq_len - 1  # Adjust for attention window or other constraints
        self.sequences = [seq[:self.max_length] for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Pad sequence to max_length if necessary
        padded_sequence = np.pad(
            sequence, (0, self.max_length - len(sequence)), mode='constant'
        )

        return torch.tensor(padded_sequence, dtype=torch.long)


class DataSequence:
    """
    A base class for handling general sequence data.

    Attributes:
        sequence_id (Optional[int]): Unique identifier for the sequence.
        sequence (List[Any]): A generic list representing the sequence.
        tokenized_sequence (np.ndarray): Tokenized representation of the sequence.
    """

    def __init__(self, sequence_id: Optional[int] = None):
        self.sequence_id: Optional[int] = sequence_id
        self.sequence: List[Any] = []
        self.tokenized_sequence: np.ndarray = np.array([])

    @property
    def as_dict(self) -> dict:
        """
        Provides a dictionary representation of the DataSequence instance.
        Converts the tokenized_sequence to a list if it is a NumPy array.
        """
        data = self.__dict__.copy()
        if isinstance(data.get("tokenized_sequence"), np.ndarray):
            data["tokenized_sequence"] = data["tokenized_sequence"].tolist()
        return data

    def to_jsonl_item(self) -> str:
        """
        Convert the DataSequence instance into a JSON-formatted string (a jsonl item).

        Returns:
            str: JSON string representation of the DataSequence.
        """
        return json.dumps(self.as_dict)

    @classmethod
    def from_jsonl_item(cls, json_item: str) -> 'DataSequence':
        """
        Create a DataSequence instance from a JSON-formatted string.

        Args:
            json_item (str): JSON string representation of the DataSequence.

        Returns:
            DataSequence: An instance of DataSequence reconstructed from the JSON string.
        """
        data = json.loads(json_item)
        if "tokenized_sequence" in data and isinstance(data["tokenized_sequence"], list):
            data["tokenized_sequence"] = np.array(data["tokenized_sequence"])
        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance


class StringSequence(DataSequence):
    """
    A specialized class for handling string sequences, inheriting from DataSequence.

    Attributes:
        string_id (int): Unique identifier for the string sequence.
        string (str): The string data (aliased to `sequence`).
    """

    def __init__(self, string_id: int, string: str):
        super().__init__(sequence_id=string_id)
        self.string_id: int = string_id
        self.string: str = string
        # Alias string to sequence for consistency.
        self.sequence = self.string

    def to_jsonl_item(self) -> str:
        """
        Convert the StringSequence instance into a JSON-formatted string (a jsonl item).

        Returns:
            str: JSON string representation of the StringSequence.
        """
        return super().to_jsonl_item()

    @classmethod
    def from_jsonl_item(cls, json_item: str) -> 'StringSequence':
        """
        Create a StringSequence instance from a JSON-formatted string.

        Args:
            json_item (str): JSON string representation of the StringSequence.

        Returns:
            StringSequence: An instance of StringSequence reconstructed from the JSON string.
        """
        data = json.loads(json_item)
        if "tokenized_sequence" in data and isinstance(data["tokenized_sequence"], list):
            data["tokenized_sequence"] = np.array(data["tokenized_sequence"])
        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance


class InstructionSequence(DataSequence):
    """
    A specialized class for handling instruction sequences, inheriting from DataSequence.

    Attributes:
        binary_id (int): Identifier for the binary file associated with the instructions.
        start_address (int): The starting memory address of the instruction sequence.
        instructions (List[str]): List of instruction strings (aliased to `sequence`).
    """

    def __init__(self, binary_id: int, start_address: int, instructions: List[str]):
        super().__init__()
        self.binary_id: int = binary_id
        self.start_address: int = start_address
        self.instructions: List[str] = instructions
        # Alias instructions to sequence for consistency.
        self.sequence = self.instructions

    def to_jsonl_item(self) -> str:
        """
        Convert the InstructionSequence instance into a JSON-formatted string (a jsonl item).

        Returns:
            str: JSON string representation of the InstructionSequence.
        """
        return super().to_jsonl_item()

    @classmethod
    def from_jsonl_item(cls, json_item: str) -> 'InstructionSequence':
        """
        Create an InstructionSequence instance from a JSON-formatted string.

        Args:
            json_item (str): JSON string representation of the InstructionSequence.

        Returns:
            InstructionSequence: An instance of InstructionSequence reconstructed from the JSON string.
        """
        data = json.loads(json_item)
        if "tokenized_sequence" in data and isinstance(data["tokenized_sequence"], list):
            data["tokenized_sequence"] = np.array(data["tokenized_sequence"])
        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance


class DataToken:
    """
    Represents a token with an id and a count.
    """

    def __init__(self, token_id: int, count: int = 0):
        self.id = token_id
        self.count = count

    def to_dict(self) -> dict:
        """
        Convert the DataToken instance to a dictionary.
        """
        return {"id": self.id, "count": self.count}

    @classmethod
    def from_string(cls, s: str) -> 'DataToken':
        """
        Create a DataToken instance from a JSON-formatted string.

        :param s: JSON string representation of a DataToken.
        :return: DataToken instance.
        """
        data = json.loads(s)
        return cls(token_id=data.get("id"), count=data.get("count", 0))

    def __str__(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class BinaryInfo:
    binary_info_id: int
    binary_sha256: str
    binary_name: str
    bcc_id: int
    concatenated_function_addresses: str

    @property
    def function_addresses(self) -> List[int]:
        """Convert the concatenated function addresses string into a list of integers."""
        return list(map(int, self.concatenated_function_addresses.split(',')))

    def __str__(self):
        return (f"BinaryInfo(binary_id={self.binary_info_id}, "
                f" binary_sha256={self.binary_sha256}, "
                f"binary_name={self.binary_name}, bcc_id={self.bcc_id}, "
                f"function_addresses={self.function_addresses})")

    @classmethod
    def load_from_jsonl(cls, file_path: str) -> List['BinaryInfo']:
        """Load binary information from a JSONL file, ignoring any unwanted fields."""
        with open(file_path) as f:
            json_data = [json.loads(line) for line in f.readlines()]

        binary_infos = []
        for item in json_data:
            # Filter out unwanted fields by keeping only the keys that match BinaryInfo fields
            filtered_item = {k: item[k] for k in cls.__dataclass_fields__ if k in item}
            binary_infos.append(cls(**filtered_item))

        return binary_infos


def load_tokens_from_jsonl(file_path: str) -> Dict[str, DataToken]:
    """
    Load token information from a JSONL file into a dictionary.

    :param file_path: Path to the JSONL file.
    :return: Dictionary mapping tokens to DataToken instances.
    """
    tokens_dict = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            token = data.get("token")
            if token:
                tokens_dict[token] = DataToken(token_id=data.get("id"), count=data.get("count", 0))

    return tokens_dict


def split_dataset(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: Optional[int] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits a torch dataset into train, validation, and test datasets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Fraction of the dataset to use for training. Default is 0.8.
        val_ratio (float): Fraction of the dataset to use for validation. Default is 0.1.
        test_ratio (float): Fraction of the dataset to use for testing. Default is 0.1.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the train, validation, and test datasets.

    Raises:
        ValueError: If the sum of ratios does not equal 1.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    # Assign any remaining samples to the test set to ensure all data is included.
    test_size = total_size - train_size - val_size

    # Create a generator for reproducibility if a seed is provided.
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def save_token_map(token_map: Dict[str, DataToken], file_path: str) -> None:
    """
    Save a token map to a JSONL file.

    :param token_map: Dictionary mapping tokens to DataToken instances.
    :param file_path: Path to the output JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for token, data_token in token_map.items():
            file.write(json.dumps({"token": token, "id": data_token.id, "count": data_token.count}) + "\n")


def load_token_map(file_path: str) -> Dict[str, DataToken]:
    """
    Load a token map from a JSONL file.

    :param file_path: Path to the JSONL file.
    :return: Dictionary mapping tokens to DataToken instances.
    """
    token_map = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            token_map[data["token"]] = DataToken(token_id=data["id"], count=data["count"])
    return token_map
