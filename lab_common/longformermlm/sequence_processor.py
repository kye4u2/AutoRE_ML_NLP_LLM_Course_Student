import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from logzero import setup_logger

from lab_common.longformermlm.dataset_utils import DataSequence, DataToken

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]

# Import the generic DataSequence class which includes the from_jsonl_item method.
logger = setup_logger(name=Path(__file__).stem, logfile=f"{Path(__file__).stem}.log", level="INFO")


def filter_sequences_by_unk_and_length(
        all_sequences: List[DataSequence],
        token_map: Dict[str, DataToken],
        min_sequence_length: int,
        max_unk_ratio: float,
        mask_ratio: float
) -> List[DataSequence]:
    """
    Filters sequences based on:
    - The ratio of [UNK] tokens to valid tokens (excluding special tokens).
    - The minimum number of valid tokens (excluding special tokens).
    - Ensuring that mask_ratio * sequence length is at least 1.
    - Uses tqdm to show real-time stats on filtering.

    Args:
        all_sequences (List[DataSequence]): List of sequences (each sequence contains a `tokenized_sequence` list).
        token_map (Dict[str, DataToken]): Dictionary mapping token names to `DataToken` objects containing IDs.
        min_sequence_length (int): Minimum number of valid (non-special) tokens required.
        max_unk_ratio (float): Maximum allowed ratio of [UNK] tokens to valid tokens.
        mask_ratio (float): Ratio used for additional filtering.

    Returns:
        List[DataSequence]: Filtered list of sequences meeting the given criteria.
    """

    # Retrieve special token IDs from token_map
    pad_token_id: int = token_map["[PAD]"].id
    unk_token_id: int = token_map["[UNK]"].id
    cls_token_id: int = token_map["[CLS]"].id
    mask_token_id: int = token_map["[MASK]"].id

    new_sequences: List[DataSequence] = []
    filtered_due_to_ratio: int = 0
    filtered_due_to_min_tokens: int = 0
    filtered_due_to_mask_ratio: int = 0

    for ds in tqdm(all_sequences, desc="Filtering Sequences", unit="seq"):
        tokenized_seq: np.ndarray = np.array(ds.tokenized_sequence)  # Convert to NumPy array for efficient processing
        special_token_mask: np.ndarray = np.isin(tokenized_seq,
                                                 [pad_token_id, unk_token_id, cls_token_id, mask_token_id])

        # Compute counts of [UNK] tokens and valid tokens (excluding special tokens)
        unk_count: int = np.sum(tokenized_seq == unk_token_id)
        valid_tokens_count: int = np.sum(~special_token_mask)  # Count tokens that are NOT special tokens
        total_tokens: int = len(tokenized_seq)

        # Ensure mask_ratio * sequence length is at least 1
        if mask_ratio * total_tokens < 1:
            filtered_due_to_mask_ratio += 1
        # Check minimum sequence length condition (ignoring special tokens)
        elif valid_tokens_count < min_sequence_length:
            filtered_due_to_min_tokens += 1
        # Check the [UNK] token ratio condition
        elif (unk_count / valid_tokens_count if valid_tokens_count > 0 else 0) > max_unk_ratio:
            filtered_due_to_ratio += 1
        else:
            new_sequences.append(ds)

    # Log details about sequences removed
    removed_sequences: int = len(all_sequences) - len(new_sequences)
    logger.info(
        f"Filtering sequences based on UNK ratio {max_unk_ratio}, min valid token count {min_sequence_length}, and mask ratio {mask_ratio}:")
    logger.info(f"  - Initial sequences: {len(all_sequences)}")
    logger.info(f"  - Removed due to high [UNK] ratio: {filtered_due_to_ratio}")
    logger.info(f"  - Removed due to min valid token requirement: {filtered_due_to_min_tokens}")
    logger.info(f"  - Removed due to mask ratio constraint: {filtered_due_to_mask_ratio}")
    logger.info(f"  - Total removed sequences: {removed_sequences}")
    logger.info(f"  - Final sequences after filtering: {len(new_sequences)}")

    return new_sequences  # Return the filtered sequences


def compute_token_frequencies(token_map: Dict[str, DataToken], display_vocab_distribution: bool = True):
    """Computes and optionally displays the token frequency distribution with statistical metrics."""
    token_frequencies = [dt.count for dt in token_map.values()]

    if token_frequencies:
        min_freq = np.min(token_frequencies)
        max_freq = np.max(token_frequencies)
        median_freq = np.median(token_frequencies)
        quartiles = np.percentile(token_frequencies, [25, 50, 75, 90])

        if display_vocab_distribution:
            logger.info("Token Frequency Statistics:")
            logger.info(f"Min: {min_freq}, Max: {max_freq}, Median: {median_freq}")
            logger.info(f"25th percentile: {quartiles[0]}, 50th percentile: {quartiles[1]}, "
                  f"75th percentile: {quartiles[2]}, 90th percentile: {quartiles[3]}")

    return token_frequencies


def filter_min_token_occurrence_from_sequences(token_map, all_sequences, min_token_occurrence):
    """
    Filters out tokens that do not meet the minimum occurrence threshold, reindexes the valid tokens,
    creates a mapping from old token IDs to new token IDs, and updates the tokenized sequences for each DataSequence.

    Args:
        token_map (dict): Mapping of token strings to DataToken objects.
        all_sequences (list): List of DataSequence objects, each having a tokenized_sequence attribute.
        min_token_occurrence (int): Minimum occurrence threshold to retain a token.

    Returns:
        tuple: A tuple containing:
            - new_token_map (dict): A new token_map with tokens meeting the threshold and updated IDs.
            - all_sequences (list): The updated list of DataSequence objects with reindexed tokenized sequences.
    """


    # 1. Filter out tokens that do not meet the minimum occurrence threshold.
    valid_tokens = {token: dt for token, dt in token_map.items() if dt.count >= min_token_occurrence or  token  in SPECIAL_TOKENS}


    # 2. Reindex the valid tokens.
    new_token_map = {}
    new_token_id = 0
    for token, dt in valid_tokens.items():
        new_token_map[token] = DataToken(new_token_id, dt.count)
        new_token_id += 1

    # 3. Create a mapping from the old token IDs to the new token IDs.
    old_to_new = {}
    for token, dt in token_map.items():
        if token in valid_tokens:
            old_to_new[dt.id] = new_token_map[token].id
        else:
            # Tokens that don't meet the threshold will be remapped to the "[UNK]" new id.
            old_to_new[dt.id] = new_token_map["[UNK]"].id

    # 4. Update the tokenized sequences for each DataSequence.
    for ds in tqdm(all_sequences, desc="Updating Sequences based on pruned vocabulary", unit="seq"):
        ds.tokenized_sequence = np.array(
            [old_to_new[tid] for tid in ds.tokenized_sequence],
            dtype=np.int32
        )

    return new_token_map, all_sequences


def load_data_sequences_from_folder(
        folder_path: str,
        token_map: Dict[str, DataToken] = None,
        max_sequences: int = 0,
        allow_token_addition: Optional[bool] = None,
        min_sequence_length: int = 0,
        max_vocab_size: int = 0,
        max_unk_ratio: float = .20,
        mask_ratio: float = 0.15,
        min_token_occurrence: int = 7,
        display_vocab_distribution: bool = True,
        prepend_cls: bool = True
) -> Tuple[List[DataSequence], Dict[str, DataToken]]:
    """
    Load and tokenize DataSequence data from all .jsonl files in a folder,
    update token counts in the token_map, and create DataSequence instances
    using the from_jsonl_item class method.

    Each JSONL record is expected to contain at least a "sequence" key with a list of items.
    Optionally, additional keys (e.g. "binary_id", "start_address") may be present and will be attached
    to the resulting DataSequence object.

    :param folder_path: Path to the folder containing .jsonl files.
    :param token_map: Optional pre-existing token map mapping token string to DataToken.
                      Tokens not found in the map will be assigned the "[UNK]" token ID.
    :param max_sequences: If > 0, limits the total number of sequences loaded.
    :param allow_token_addition: If True, allows adding new tokens to the token map.
    :param min_sequence_length: Minimum length of sequences to include in the output (0 for no limit).
    :param max_vocab_size: If > 0, limits the vocabulary to the top tokens by occurrence count.
                           Tokens not in the top vocabulary will be replaced with the [UNK] token.
                           A value of 0 means no limit.
    :param max_unk_ratio: If the ratio of [UNK] tokens in a sequence exceeds this value, the sequence is skipped.
    :param mask_ratio: Minimum ratio of tokens to mask during training.
    :param min_token_occurrence: Minimum number of occurrences for a token to be included in the vocabulary.
    :param display_vocab_distribution: If True, displays vocabulary distribution information.
    :param prepend_cls: If True, ensures each tokenized sequence starts with the [CLS] token (default: True).
    :return: A tuple containing:
             - List of DataSequence instances with tokenized sequence data.
             - Updated token map (with DataToken instances for each token).
    """
    # Initialize the token map if not provided.
    if token_map is None:
        token_map = {
            "[PAD]": DataToken(0, 0),
            "[UNK]": DataToken(1, 0),
            "[CLS]": DataToken(2, 0),
            "[MASK]": DataToken(3, 0)
        }
        allow_token_addition = True
    else:
        # Ensure the required tokens are in the provided token map.
        required_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]
        for token in required_tokens:
            if token not in token_map:
                raise ValueError(f"Provided token_map must include '{token}' token.")

        if allow_token_addition is None:
            allow_token_addition = False

    next_token_id = len(token_map)  # Start the next token ID after the existing tokens.

    all_sequences: List[DataSequence] = []
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

    sequence_counter = 0
    skipped_sequences = 0

    for file_name in tqdm(jsonl_files, desc="Processing folder", unit="file"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Processing {file_name}", unit="lines", leave=False):
                # Create a DataSequence instance from the JSON line using from_jsonl_item.
                ds = DataSequence.from_jsonl_item(line.strip())
                # Retrieve the sequence data from the instance.
                sequence_data = ds.sequence

                if min_sequence_length > 0 and len(sequence_data) < min_sequence_length:
                    logger.debug(f"Skipping sequence with length {len(sequence_data)}")
                    skipped_sequences += 1
                    continue

                sequence_counter += 1
                if 0 < max_sequences < sequence_counter:
                    break

                # Tokenize the sequence data.
                tokenized_sequence = []
                for item in sequence_data:
                    if item in token_map:
                        tokenized_sequence.append(token_map[item].id)
                        token_map[item].count += 1  # Increment count.
                    elif allow_token_addition:
                        token_map[item] = DataToken(next_token_id, 1)
                        tokenized_sequence.append(next_token_id)
                        next_token_id += 1
                    else:
                        tokenized_sequence.append(token_map["[UNK]"].id)
                        token_map["[UNK]"].count += 1

                # Optionally, ensure the sequence starts with the [CLS] token.
                if prepend_cls:
                    if not tokenized_sequence or tokenized_sequence[0] != token_map["[CLS]"].id:
                        tokenized_sequence = [token_map["[CLS]"].id] + tokenized_sequence
                        token_map["[CLS]"].count += 1

                # Update the DataSequence instance with the tokenized sequence.
                ds.tokenized_sequence = np.array(tokenized_sequence, dtype=np.int32)
                all_sequences.append(ds)

        if 0 < max_sequences < sequence_counter:
            break


    if display_vocab_distribution:
        logger.info("Computing token frequency distribution before filtering:")
        compute_token_frequencies(token_map, display_vocab_distribution=display_vocab_distribution)


    if 0 < max_sequences < sequence_counter:
        print(f"Stopped early after processing {max_sequences} data sequences.")

    if skipped_sequences > 0:
        logger.info(f"Skipped {skipped_sequences} sequences with length less than {min_sequence_length}.")

    if min_token_occurrence > 0:

        logger.info(f"Filtering tokens with occurrence count less than {min_token_occurrence}.")
        logger.info(f"Token map size before filtering: {len(token_map)}")

        # Filter, reindex, and update sequences based on token occurrence and length.
        token_map, all_sequences = filter_min_token_occurrence_from_sequences(token_map, all_sequences, min_token_occurrence)

        # Log the impact to the original token map.
        logger.info(f"Token map updated with {len(token_map)} tokens.")


    # Apply max_vocabulary limitation if required.
    if 0 < max_vocab_size < len(token_map):
        # Always include special tokens.
        special_tokens = {"[PAD]", "[UNK]", "[CLS]", "[MASK]"}
        total_vocab = len(token_map)
        # Get non-special tokens and sort by occurrence count.
        non_special_tokens = [(token, dt.count) for token, dt in token_map.items() if token not in special_tokens]
        non_special_tokens_sorted = sorted(non_special_tokens, key=lambda x: x[1], reverse=True)
        # Calculate how many non-special tokens can be kept.
        max_vocab_remaining = max_vocab_size - len(special_tokens)
        if max_vocab_remaining < 0:
            max_vocab_remaining = 0
        # Select allowed non-special tokens.
        allowed_non_special_tokens = {token for token, _ in non_special_tokens_sorted[:max_vocab_remaining]}
        # Build the set of allowed tokens.
        allowed_tokens = special_tokens.union(allowed_non_special_tokens)
        # Log details about vocabulary reduction.
        removed_tokens = set(token_map.keys()) - allowed_tokens
        logger.info(
            f"Total vocab before limiting: {total_vocab}. Num tokens removed due to max_vocabulary: {len(removed_tokens)}")

        # Reindex the tokens: keep special tokens in fixed order, then reindex non-special tokens.
        new_token_map = {}
        # Fixed order for special tokens.
        special_order = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]
        new_id = 0
        for token in special_order:
            # Assuming these tokens exist.
            dt = token_map[token]
            new_token_map[token] = DataToken(new_id, dt.count)
            new_id += 1

        # Order allowed non-special tokens by descending count.
        allowed_non_special_list = [(token, token_map[token].count) for token in allowed_non_special_tokens]
        allowed_non_special_sorted = sorted(allowed_non_special_list, key=lambda x: x[1], reverse=True)
        for token, _ in allowed_non_special_sorted:
            dt = token_map[token]
            new_token_map[token] = DataToken(new_id, dt.count)
            new_id += 1

        # Create a mapping from old token IDs to new token IDs.
        old_to_new = {}
        for token in allowed_tokens:
            old_id = token_map[token].id
            new_id = new_token_map[token].id
            old_to_new[old_id] = new_id

        # New id for [UNK] to be used for any token not in allowed tokens.
        unk_new_id = new_token_map["[UNK]"].id
        for ds in all_sequences:
            ds.tokenized_sequence = np.array(
                [old_to_new.get(token_id, unk_new_id) for token_id in ds.tokenized_sequence],
                dtype=np.int32
            )
        token_map = new_token_map

    # Filter sequences based on UNK ratio and minimum sequence length.
    all_sequences = filter_sequences_by_unk_and_length(all_sequences, token_map, min_sequence_length, max_unk_ratio,
                                                       mask_ratio=mask_ratio)

    # Compute and optionally display the token frequency distribution.
    if display_vocab_distribution:
        logger.info("Computing token frequency distribution after filtering:")
        compute_token_frequencies(token_map, display_vocab_distribution=display_vocab_distribution)

    return all_sequences, token_map


def save_token_map(token_map: Dict[str, DataToken], output_path: str):
    """
    Save the token map to a JSONL file.

    :param token_map: Dictionary mapping sequence items to DataToken.
    :param output_path: Path to save the token mapping as JSONL.
    """
    with open(output_path, 'w') as f:
        for token, data in token_map.items():
            f.write(json.dumps({
                "token": token,
                "id": data.id,
                "count": data.count
            }) + "\n")
    print(f"Token map saved to {output_path}")


def compute_sequence_length_distribution(sequences: List[DataSequence]) -> List[int]:
    """
    Compute the distribution of sequence lengths.

    :param sequences: List of DataSequence objects.
    :return: List of lengths of the sequences.
    """
    return [len(seq.sequence) for seq in sequences]


def plot_sequence_length_distribution(lengths: List[int], output_path: str = None):
    """
    Plot and optionally save the distribution of sequence lengths.

    :param lengths: List of sequence lengths.
    :param output_path: Optional path to save the plot as an image file.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if output_path:
        plt.savefig(output_path)
        print(f"Distribution plot saved to {output_path}")
    else:
        plt.show()


def print_sequence_stats(sequences: List[DataSequence]):
    """
    Print summary statistics for the given list of sequences.

    :param sequences: List of DataSequence objects.
    """
    lengths = [len(seq.sequence) for seq in sequences]
    logger.info("Sequence Length Statistics:")
    logger.info(f"  Total sequences: {len(lengths)}")
    logger.info(f"  Min length: {np.min(lengths)}")
    logger.info(f"  Max length: {np.max(lengths)}")
    logger.info(f"  Mean length: {np.mean(lengths):.2f}")
    logger.info(f"  Median length: {np.median(lengths)}")


def main():
    """
    Main function responsible for parsing command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process all JSONL files in a folder into DataSequence instances and analyze sequence lengths."
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help="Path to the folder containing the .jsonl files."
    )
    parser.add_argument(
        '--token_map_path',
        type=str,
        default="token_map.jsonl",
        help="Path to save the token mapping as JSONL. Default: 'token_map.jsonl'."
    )
    parser.add_argument(
        '--output_plot_path',
        type=str,
        default=None,
        help="Path to save the histogram of sequence lengths as an image. Default: None (display only)."
    )
    args = parser.parse_args()

    # Load and tokenize DataSequence data from all JSONL files in the folder.
    sequences, token_map = load_data_sequences_from_folder(args.folder_path)

    # Save the token map.
    save_token_map(token_map, args.token_map_path)

    # Compute the distribution of sequence lengths.
    lengths = compute_sequence_length_distribution(sequences)

    # Print summary statistics.
    print_sequence_stats(sequences)

    # Plot the distribution.
    plot_sequence_length_distribution(lengths, output_path=args.output_plot_path)


if __name__ == "__main__":
    main()
