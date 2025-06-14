import torch
from typing import List, Dict


def convert_tokens_to_tensor(tokens: List[str], token_map: Dict[str, Dict[str, int]], max_sequence: int) -> torch.Tensor:
    """
    Convert a list of tokens into a tensor using the provided token map.
    If a token is not found, the `[UNK]` token is used.
    The sequence is padded with `[PAD]` tokens up to `max_sequence` length or truncated if necessary.

    Args:
        tokens (List[str]): List of tokens to convert.
        token_map (Dict[str, Dict[str, int]]): Mapping from token strings to a dict containing 'id' and 'count'.
        max_sequence (int): The maximum sequence length (including special tokens).

    Returns:
        torch.Tensor: Tensor of token IDs with shape (1, max_sequence).
    """
    tokenized = []

    # default input filled with PAD tokens of 0 with max_seq_len
    input_ids = [token_map["[PAD]"]["id"]] * max_sequence

    tokenized_tensor = None

    ### YOUR CODE HERE ###





    ### END YOUR CODE HERE ###

    return tokenized_tensor  # Shape: (1, max_sequence)


def get_token_map() -> Dict[str, Dict[str, int]]:
    """
    Returns a fresh token map with counts initialized to zero.
    The map includes a larger vocabulary of English library function names that are pre-tokenized.
    """
    return {
        "[PAD]": {"id": 0, "count": 0},
        "[UNK]": {"id": 1, "count": 0},
        "[CLS]": {"id": 2, "count": 0},
        "open": {"id": 3, "count": 0},
        "read": {"id": 4, "count": 0},
        "write": {"id": 5, "count": 0},
        "close": {"id": 6, "count": 0},
        "sort": {"id": 7, "count": 0},
        "filter": {"id": 8, "count": 0},
        "append": {"id": 9, "count": 0},
        "insert": {"id": 10, "count": 0},
        "remove": {"id": 11, "count": 0},
        "pop": {"id": 12, "count": 0},
        "split": {"id": 13, "count": 0},
        "join": {"id": 14, "count": 0},
        "replace": {"id": 15, "count": 0},
        "find": {"id": 16, "count": 0},
        "index": {"id": 17, "count": 0},
        "count": {"id": 18, "count": 0}
    }


def test_tokens_to_tensor():
    # Test Case 1: All known tokens with padding.
    token_map = get_token_map()
    tokens = ["open", "read", "write"]
    max_sequence = 7  # [CLS] + 3 tokens + 3 PAD tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2, 3, 4, 5, 0, 0, 0]])
    print("Test Case 1 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    assert token_map["open"]["count"] == 1, "open count should be 1"
    assert token_map["read"]["count"] == 1, "read count should be 1"
    assert token_map["write"]["count"] == 1, "write count should be 1"

    # Test Case 2: Mixed known and unknown tokens.
    token_map = get_token_map()
    tokens = ["open", "nonexistent_func", "write", "split"]
    max_sequence = 7  # [CLS] + 4 tokens + 2 PAD tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2, 3, 1, 5, 13, 0, 0]])
    print("Test Case 2 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    assert token_map["open"]["count"] == 1, "open count should be 1"
    assert token_map["[UNK]"]["count"] == 1, "[UNK] count should be 1"
    assert token_map["write"]["count"] == 1, "write count should be 1"
    assert token_map["split"]["count"] == 1, "split count should be 1"

    # Test Case 3: Sequence exactly matching max_sequence length.
    token_map = get_token_map()
    tokens = ["open", "read", "write", "close"]  # [CLS] + 4 tokens = 5 tokens total.
    max_sequence = 5
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2, 3, 4, 5, 6]])
    print("Test Case 3 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"

    # Test Case 4: Sequence that exceeds max_sequence (truncation).
    token_map = get_token_map()
    tokens = ["open", "read", "write", "close", "sort", "filter", "append", "insert"]
    max_sequence = 6  # [CLS] + 5 tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    # Full sequence would be: [CLS], open, read, write, close, sort, filter, append, insert (9 tokens)
    # Truncated to first 6 tokens: [CLS], open, read, write, close, sort.
    expected = torch.tensor([[2, 3, 4, 5, 6, 7]])
    print("Test Case 4 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    # Counts are updated for all tokens processed, even those truncated.
    assert token_map["open"]["count"] == 1, "open count should be 1"
    assert token_map["read"]["count"] == 1, "read count should be 1"
    assert token_map["write"]["count"] == 1, "write count should be 1"
    assert token_map["close"]["count"] == 1, "close count should be 1"
    assert token_map["sort"]["count"] == 1, "sort count should be 1"
    assert token_map["filter"]["count"] == 1, "filter count should be 1"
    assert token_map["append"]["count"] == 1, "append count should be 1"
    assert token_map["insert"]["count"] == 1, "insert count should be 1"

    # Test Case 5: Empty token list.
    token_map = get_token_map()
    tokens = []
    max_sequence = 5  # Only [CLS] plus padding expected.
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2, 0, 0, 0, 0]])
    print("Test Case 5 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"

    # Test Case 6: Repeated tokens.
    token_map = get_token_map()
    tokens = ["open", "open", "read", "read", "write"]
    max_sequence = 8  # [CLS] + 5 tokens + 2 PAD tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2, 3, 3, 4, 4, 5, 0, 0]])
    print("Test Case 6 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    assert token_map["open"]["count"] == 2, "open count should be 2"
    assert token_map["read"]["count"] == 2, "read count should be 2"
    assert token_map["write"]["count"] == 1, "write count should be 1"

    # Test Case 7: All tokens are unknown.
    token_map = get_token_map()
    tokens = ["nonexistent1", "nonexistent2", "nonexistent3"]
    max_sequence = 6  # [CLS] + 3 tokens + 2 PAD tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2, 1, 1, 1, 0, 0]])
    print("Test Case 7 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    assert token_map["[UNK]"]["count"] == 3, "[UNK] count should be 3"

    # Test Case 8: max_sequence is 1 (edge case).
    token_map = get_token_map()
    tokens = ["open", "read"]
    max_sequence = 1  # Only the prepended [CLS] will be kept.
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    expected = torch.tensor([[2]])
    print("Test Case 8 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    # Counts update even if the tokens don't appear in the final output.
    assert token_map["open"]["count"] == 1, "open count should be 1"
    assert token_map["read"]["count"] == 1, "read count should be 1"

    # Test Case 9: Input contains special tokens as part of the sequence.
    token_map = get_token_map()
    tokens = ["[CLS]", "open", "write", "[UNK]"]
    max_sequence = 8  # [CLS] prepended + 4 tokens + 3 PAD tokens = 8 tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    # Expected sequence: prepended [CLS] (id 2), then [CLS] (id 2), open (3), write (5), [UNK] (1), and padding.
    expected = torch.tensor([[2, 2, 3, 5, 1, 0, 0, 0]])
    print("Test Case 9 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    # Note: Only tokens in the list are counted.
    assert token_map["[CLS]"]["count"] == 1, "[CLS] count should be 1 from tokens"
    assert token_map["open"]["count"] == 1, "open count should be 1"
    assert token_map["write"]["count"] == 1, "write count should be 1"
    assert token_map["[UNK]"]["count"] == 1, "[UNK] count should be 1"

    # Test Case 10: Sequence with repeated unknown tokens mixed with known ones.
    token_map = get_token_map()
    tokens = ["foobar", "foobar", "open", "foobar", "read"]
    max_sequence = 8  # [CLS] + 5 tokens + 2 PAD tokens
    tensor_output = convert_tokens_to_tensor(tokens, token_map, max_sequence)
    # "foobar" is not in the map so becomes [UNK] (id 1)
    expected = torch.tensor([[2, 1, 1, 3, 1, 4, 0, 0]])
    print("Test Case 10 Output:", tensor_output)
    assert torch.equal(tensor_output, expected), f"Expected {expected}, but got {tensor_output}"
    assert token_map["[UNK]"]["count"] == 3, "[UNK] count should be 3"
    assert token_map["open"]["count"] == 1, "open count should be 1"
    assert token_map["read"]["count"] == 1, "read count should be 1"

    print("All tests passed.")


if __name__ == '__main__':
    test_tokens_to_tensor()
