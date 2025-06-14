import os
from typing import Tuple

import torch

from blackfyre.datatypes.contexts.vex.vexbbcontext import VexBasicBlockContext
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from labs.lab8.lab_8_1.longformer_mlm_bb_encoder import LongformerMLMBasicBlockEncoder

DEFAULT_TEST_BCC_FILE_PATH: str = os.path.join(
    ROOT_PROJECT_FOLDER_PATH, "Blackfyre", "test",
    "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc"
)

DEFAULT_CHECKPOINT_FOLDER_PATH: str = os.path.join(
    ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab8", "longformer_mlm_bb_encoder", "small",
    "checkpoints", "longformer_mlm_2025-03-10_21-01-20"
)

FUNCTION_ADDRESS: int = 0x2986c


def load_encoder_and_token_map(checkpoint_path: str) -> Tuple[LongformerMLMBasicBlockEncoder, dict]:
    """
    Load a trained LongformerMLM encoder and associated token map from checkpoint.
    """
    return LongformerMLMBasicBlockEncoder.from_checkpoint(checkpoint_path)


def get_basic_block_context(bcc_path: str, function_addr: int) -> VexBasicBlockContext:
    """
    Extract the basic block context for a specific function address.
    """
    binary_context: VexBinaryContext = VexBinaryContext.load_from_file(bcc_path)
    function_context: VexFunctionContext = binary_context.get_function_context(function_addr)

    if function_addr not in function_context.basic_block_context_dict:
        raise KeyError(f"Function address {hex(function_addr)} not found in basic block context dictionary.")

    return function_context.basic_block_context_dict[function_addr]



def compare_embedding(tensor_a: torch.Tensor, tensor_b: torch.Tensor, atol: float = 1e-4) -> bool:
    """
    Compare two tensors for approximate equality. If not equal, print detailed error metrics.
    """

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_a = tensor_a.to(device)
    tensor_b = tensor_b.to(device)

    if torch.allclose(tensor_a, tensor_b, atol=atol):
        return True

    print("Embeddings do NOT match.")
    abs_diff = torch.abs(tensor_a - tensor_b)
    max_error = abs_diff.max().item()
    print(f"Max absolute error: {max_error:.6f}")

    max_error_idx = torch.argmax(abs_diff)
    idx_unraveled = torch.unravel_index(max_error_idx, abs_diff.shape)
    actual_val = tensor_a[idx_unraveled].item()
    expected_val = tensor_b[idx_unraveled].item()
    print(f"Largest mismatch at {idx_unraveled}: actual={actual_val:.6f}, expected={expected_val:.6f}")

    return False



def main():
    """
    Main function to load encoder, extract basic block context, and generate embedding.
    """
    # Load encoder
    encoder, token_map = load_encoder_and_token_map(DEFAULT_CHECKPOINT_FOLDER_PATH)

    # Get basic block context
    bb_context: VexBasicBlockContext = get_basic_block_context(DEFAULT_TEST_BCC_FILE_PATH, FUNCTION_ADDRESS)

    # Generate embedding
    embedding: torch.Tensor = encoder.encode([bb_context])

    print(f"Embedding shape: {embedding.shape}")
    print(embedding)

    # Optional: Validate against known tensor
    known_embedding = torch.tensor([[[-5.1671e-01, 2.5115e-01, 1.0560e+00, 5.3838e-01, -8.1875e-01,
                                      7.8903e-01, -5.5383e-01, 5.3107e-01, 4.8052e-01, 6.2813e-02,
                                      -1.2771e+00, 1.0327e+00, 1.4458e+00, -9.7107e-01, -6.4656e-02,
                                      -1.1952e+00, 5.8020e-01, 3.0757e-01, -1.2104e+00, -1.1289e+00,
                                      3.1517e-01, 1.3373e+00, 6.5819e-01, 2.1994e-03, -1.1022e+00,
                                      1.0278e+00, -1.9936e-01, 1.2131e-01, 5.9105e-01, -1.3820e-01,
                                      8.4887e-01, 1.4975e+00, 1.5504e+00, -2.0409e-01, -2.0847e-01,
                                      -8.5748e-01, 1.3559e+00, -5.7997e-01, -6.7595e-01, -8.5410e-01,
                                      7.6816e-01, -1.8408e-01, 1.8899e+00, -4.5737e-01, -8.1175e-01,
                                      -2.3619e-01, 5.1276e-01, -7.0673e-01, -1.6261e+00, 1.5655e-01,
                                      1.3116e-01, 7.4538e-01, -5.3078e-01, -3.5701e-01, 1.0445e-01,
                                      -4.7195e-01, 2.2002e-01, 4.4592e-01, -4.8032e-01, -8.0873e-02,
                                      6.3131e-01, 1.6472e-01, -9.3050e-01, -5.4718e-01, 3.5504e-02,
                                      1.7053e-02, 3.4108e-01, 1.3772e+00, 1.1990e+00, 2.5899e-01,
                                      7.1247e-01, -8.7814e-01, -4.0462e-01, -1.2263e+00, -1.5676e+00,
                                      -1.3562e-01, 2.2599e-01, -7.6281e-01, -1.3253e+00, -7.8753e-01,
                                      5.5362e-01, -1.9240e-01, 1.3454e+00, 4.5014e-01, -4.0376e-01,
                                      -1.2927e+00, -4.5419e-01, 5.9248e-02, -8.0923e-01, -4.6231e-01,
                                      -6.0345e-01, -1.4075e+00, 2.6727e+00, 7.5037e-01, -4.1853e-01,
                                      2.4692e+00, 1.0728e+00, -1.1719e-01, 1.4089e+00, -5.9738e-01,
                                      -2.9351e-01, 5.2092e-01, -6.8340e-01, 2.3202e+00, 7.6941e-01,
                                      7.0014e-02, -1.1893e+00, 1.4682e+00, -3.1869e-01, -1.1019e+00,
                                      9.7187e-01, -8.7606e-01, 9.0193e-01, 2.2723e-01, 2.4899e-01,
                                      5.4306e-01, -6.1754e-02, -1.0856e+00, -1.1808e+00, 6.2425e-01,
                                      -6.1015e-01, -1.6525e-01, -5.7595e-01, -2.3368e-03, 9.5845e-01,
                                      1.2309e+00, 9.7590e-03, 6.1832e-01, -1.6294e+00, -1.5920e+00,
                                      3.3466e-01, -6.7881e-01, 7.9938e-01, 2.4998e-01, -7.2092e-01,
                                      3.4851e-01, -6.1547e-01, -9.8467e-01, -1.5783e-01, 9.8786e-01,
                                      -1.0554e-01, 4.2454e-01, -2.3018e-01, 7.4624e-02, 3.2652e-01,
                                      8.9695e-01, 8.9723e-01, -2.6850e-01, 1.2462e+00, 1.0663e+00,
                                      1.1356e+00, -3.8847e-01, 7.1344e-01, 1.4791e+00, -7.8465e-01,
                                      7.1147e-01, 2.4201e-01, 5.3347e-01, 4.4506e-01, 2.5517e-02,
                                      1.1404e+00, -1.4789e-02, 1.4826e+00, -5.8155e-02, -1.4001e+00,
                                      1.3095e+00, 1.4148e+00, -1.1162e+00, 7.4520e-01, 4.1593e-01,
                                      -4.2110e-01, -1.7894e+00, -3.3825e-01, 2.1570e-01, 2.5025e-01,
                                      4.5809e-01, 5.2294e-01, 8.0319e-01, 6.4417e-01, 6.5229e-01,
                                      -5.2616e-01, 5.6316e-02, 1.7879e-01, 4.4420e-01, -4.2515e-01,
                                      -8.8260e-01, 1.3061e+00, -5.5623e-02, -7.8769e-01, 2.3567e-01,
                                      -1.5913e+00, 3.5308e-01, 1.2868e+00, 7.0826e-01, 2.0541e-01,
                                      -5.5879e-02, -6.0047e-01, 6.6796e-02, 6.5159e-01, 3.1295e-01,
                                      7.0116e-01, -5.7294e-01, -9.0561e-02, 1.2452e-01, -9.5468e-01,
                                      -9.9500e-01, -2.3901e-01, 4.4914e-01, -9.2414e-01, 6.0261e-01,
                                      -3.8770e-01, -6.1859e-01, 3.5069e-01, -7.1840e-01, -1.4104e+00,
                                      -1.1527e+00, -9.8006e-01, 8.6285e-01, -8.8554e-01, 6.9237e-01,
                                      -2.7805e-02, -2.9560e-01, 6.2316e-01, -5.6838e-01, -7.3456e-01,
                                      -1.1198e+00, -3.5897e-02, -2.3761e-01, -7.3679e-01, 1.2358e+00,
                                      3.2718e-01, -6.2124e-01, -3.4890e-01, -3.4691e-01, -7.1722e-01,
                                      1.2770e-01, 7.1232e-01, 6.0537e-01, -2.8308e-01, -1.0449e+00,
                                      -9.0704e-01, -1.3935e+00, 4.8175e-01, -7.9538e-01, 2.2417e-01,
                                      -5.5133e-01, -4.3879e-01, -1.7625e-01, -3.3012e-01, 5.0523e-01,
                                      -4.9495e-01, 3.1076e-01, 1.4704e+00, -3.7522e-01, -1.5185e+00,
                                      5.8311e-01, 1.0330e+00, -5.8787e-01, -1.1019e-01, 7.3970e-01,
                                      -1.1986e+00, -1.3840e+00, 5.5282e-01, 6.3641e-01, 2.2111e-01,
                                      5.1445e-01, 1.9142e-01, -8.7065e-01, 4.2844e-01, -6.6971e-01,
                                      3.9662e-01, -1.0861e+00, -3.3327e-01, 4.5522e-01, -9.3875e-01,
                                      -1.4602e-01, 2.1582e-01, -1.7755e-01, -2.7387e-01, 1.1699e-01,
                                      2.2817e-01, 5.0237e-01, -3.3837e-03, 5.8242e-01, 2.4429e-01,
                                      -2.8287e-01, -4.7841e-01, 3.5210e-01, -9.0114e-01, 9.4739e-01,
                                      -2.7628e-01, 3.6726e-01, -9.1046e-02, 1.4438e+00, -1.0600e+00,
                                      1.5404e+00, 1.1635e+00, -4.8013e-02, 8.1436e-01, 7.3656e-01,
                                      -1.2241e+00, 8.5601e-01, -1.5672e+00, -1.4212e+00, 6.2432e-01,
                                      1.1301e+00, 1.3938e+00, 1.3112e+00, 1.0566e+00, 3.0097e-01,
                                      7.8224e-01, 8.9191e-02, -9.8038e-01, 1.9393e-01, -7.7635e-01,
                                      9.5229e-01, 7.2433e-01, -5.3699e-01, -2.7970e-01, 5.0739e-01,
                                      -8.5030e-01, -3.3881e-01, -3.1199e-01, -1.5966e-01, -3.7517e-01,
                                      -9.3810e-01, -8.1206e-01, 2.9686e-01, 7.9378e-01, 4.7632e-01,
                                      -1.0904e-01, -5.4403e-03, -8.7734e-02, -7.8427e-02, -2.2791e-01,
                                      -1.0081e+00, -1.6082e+00, -3.5425e-01, 1.6080e-02, -3.1740e-02,
                                      -7.2266e-02, 1.8402e-02, -1.1057e+00, -3.6096e-01, 9.7488e-01,
                                      -5.6945e-01, 1.7214e+00, 2.0408e+00, -4.9588e-01, -1.0566e+00,
                                      -7.9016e-01, -2.3654e-01, -6.0307e-01, 1.6179e+00, -6.3595e-01,
                                      2.0388e-01, 6.5112e-01, -5.8787e-01, 3.9598e-01, -5.3869e-01,
                                      1.6395e+00, -9.2565e-01, 3.9695e-01, -1.2727e+00, 1.2162e+00,
                                      -1.0687e-01, 8.1038e-01, -2.5639e-01, -3.8452e-01, -1.1664e+00,
                                      -9.6665e-01, 1.7878e-01, 1.7064e+00, -1.0435e+00, 7.4516e-01,
                                      -5.3855e-01, 3.7351e-01, 5.8388e-01, -3.3772e-02, -4.9534e-01,
                                      -1.4510e-01, -9.8006e-01, 5.7447e-01, 3.3462e-01]]])
    if compare_embedding(embedding, known_embedding):
        print("Embedding matches expected tensor.")
    else:
        print("Embedding does NOT match expected tensor.")


# --- Main Execution ---

if __name__ == "__main__":
    main()
