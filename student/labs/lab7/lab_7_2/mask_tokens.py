import torch


def _mask_tokens(inputs,
                 pad_token_id,
                 cls_token_id,
                 unk_token_id,  # Include UNK token ID
                 mask_token_id,
                 mask_ratio=0.15):
    """
    Masks random tokens in the input sequence for MLM training.
    Excludes PAD, CLS, and UNK tokens from being masked.
    """

    ### YOUR CODE HERE ###




    ### END YOUR CODE HERE ###
    return inputs, labels, mask


def test_mask_tokens_basic_functionality(mask_tokens_fn):
    torch.manual_seed(42)  # For reproducibility

    inputs = torch.tensor([
        [2, 3, 4, 5, 6, 0, 0]  # [CLS] open read write close [PAD] [PAD]
    ])
    pad_token_id = 0
    cls_token_id = 2
    unk_token_id = 1
    mask_token_id = 99
    mask_ratio = 0.5  # Expecting 2 tokens to be masked

    masked_inputs, labels, mask = mask_tokens_fn(
        inputs, pad_token_id, cls_token_id, unk_token_id, mask_token_id, mask_ratio
    )

    assert masked_inputs.shape == inputs.shape
    assert labels.shape == inputs.shape
    assert mask.shape == inputs.shape

    assert mask.sum().item() == 2, "Expected 2 masked tokens"

    for i in range(masked_inputs.size(1)):
        if mask[0, i]:
            assert masked_inputs[0, i].item() == mask_token_id
            assert labels[0, i].item() == inputs[0, i].item()
        else:
            assert labels[0, i].item() == -100

    assert not mask[0, 0], "CLS token should not be masked"
    assert not mask[0, 5], "PAD token should not be masked"
    assert not mask[0, 6], "PAD token should not be masked"


def test_mask_tokens_no_valid_tokens(mask_tokens_fn):
    inputs = torch.tensor([
        [2, 0, 0]  # [CLS] [PAD] [PAD]
    ])
    pad_token_id = 0
    cls_token_id = 2
    unk_token_id = 1
    mask_token_id = 99

    masked_inputs, labels, mask = mask_tokens_fn(
        inputs, pad_token_id, cls_token_id, unk_token_id, mask_token_id, mask_ratio=0.5
    )

    assert torch.all(labels == -100), "All labels should be -100"
    assert torch.all(mask == False), "No tokens should be masked"
    assert torch.equal(masked_inputs, inputs), "Input should be unchanged"


def test_mask_tokens_excludes_unk_token(mask_tokens_fn):
    inputs = torch.tensor([
        [2, 1, 4, 5, 0]  # [CLS] [UNK] read write [PAD]
    ])
    pad_token_id = 0
    cls_token_id = 2
    unk_token_id = 1
    mask_token_id = 99

    masked_inputs, labels, mask = mask_tokens_fn(
        inputs, pad_token_id, cls_token_id, unk_token_id, mask_token_id, mask_ratio=1.0
    )

    assert mask[0, 2] and mask[0, 3], "read and write should be masked"
    assert not mask[0, 0], "[CLS] should not be masked"
    assert not mask[0, 1], "[UNK] should not be masked"
    assert not mask[0, 4], "[PAD] should not be masked"
    assert labels[0, 0] == -100
    assert labels[0, 1] == -100
    assert labels[0, 4] == -100


def test_mask_tokens_batch_consistency(mask_tokens_fn):
    inputs = torch.tensor([
        [2, 3, 4, 5, 0],     # [CLS] open read write [PAD]
        [2, 4, 1, 6, 0]      # [CLS] read [UNK] close [PAD]
    ])
    pad_token_id = 0
    cls_token_id = 2
    unk_token_id = 1
    mask_token_id = 99
    mask_ratio = 0.5

    masked_inputs, labels, mask = mask_tokens_fn(
        inputs, pad_token_id, cls_token_id, unk_token_id, mask_token_id, mask_ratio
    )

    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            if mask[i, j]:
                assert masked_inputs[i, j].item() == mask_token_id
                assert labels[i, j].item() == inputs[i, j].item()
            else:
                assert labels[i, j].item() == -100
                if inputs[i, j].item() not in (pad_token_id, cls_token_id, unk_token_id):
                    assert masked_inputs[i, j].item() == inputs[i, j].item()



def test_mask_tokens_expected_output(mask_tokens_fn):
    """
    Tests a known input with a fixed seed and checks that the output matches expected values.
    """
    torch.manual_seed(0)  # Fixed seed for reproducibility

    inputs = torch.tensor([
        [2, 3, 4, 5, 6, 0, 0]  # [CLS] open read write close [PAD] [PAD]
    ])
    pad_token_id = 0
    cls_token_id = 2
    unk_token_id = 1
    mask_token_id = 99
    mask_ratio = 0.5  # Should mask 2 out of 4 eligible tokens

    masked_inputs, labels, mask = mask_tokens_fn(
        inputs, pad_token_id, cls_token_id, unk_token_id, mask_token_id, mask_ratio
    )

    # Expected: let's say masking randomly selected 'read' and 'write'
    expected_masked_inputs = torch.tensor([
        [2, 99, 99,  5,  6,  0,  0]
    ])
    expected_labels = torch.tensor([
        [-100,    3,    4, -100, -100, -100, -100]
    ])
    expected_mask = torch.tensor([
        [False, True, True, False, False, False, False]
    ])

    assert torch.equal(masked_inputs, expected_masked_inputs), "Masked inputs do not match expected"
    assert torch.equal(labels, expected_labels), "Labels do not match expected"
    assert torch.equal(mask, expected_mask), "Mask tensor does not match expected"

def run_all_tests(mask_tokens_fn):
    print("Running test_mask_tokens_basic_functionality...")
    test_mask_tokens_basic_functionality(mask_tokens_fn)
    print("✅ Passed")

    print("Running test_mask_tokens_no_valid_tokens...")
    test_mask_tokens_no_valid_tokens(mask_tokens_fn)
    print("✅ Passed")

    print("Running test_mask_tokens_excludes_unk_token...")
    test_mask_tokens_excludes_unk_token(mask_tokens_fn)
    print("✅ Passed")

    print("Running test_mask_tokens_batch_consistency...")
    test_mask_tokens_batch_consistency(mask_tokens_fn)
    print("✅ Passed")

    print("Running test_mask_tokens_expected_output...")
    test_mask_tokens_expected_output(mask_tokens_fn)
    print("✅ Passed")


# Optionally define a default entry point
if __name__ == "__main__":

    run_all_tests(_mask_tokens)
