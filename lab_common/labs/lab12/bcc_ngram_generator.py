import logging
import os
from collections import deque
from typing import List, Optional, Set

import numpy as np

from blackfyre.datatypes.contexts.binarycontext import BinaryContext
from blackfyre.datatypes.contexts.functioncontext import FunctionContext
from blackfyre.utils import setup_custom_logger
from lab_common.common import ROOT_PROJECT_FOLDER_PATH

DEFAULT_WINDOW_SIZE = 5

logger = setup_custom_logger(os.path.splitext(os.path.basename(__file__))[0])
logging.getLogger("binarycontext").setLevel(logging.WARN)
logger.setLevel(logging.INFO)

LAB_12_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab12")

TEST_BCC_FILE_PATH = os.path.join(LAB_12_DATASET, "benign",
                                  "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")


class BCCNGramGenerator(object):

    def __init__(self, bcc_file_path: str):

        self._bcc_file_path = bcc_file_path

        self._binary_context: Optional[BinaryContext] = None

        self._ngram_list: Optional[List[str]] = None

        self._initialized: bool = False

    def initialize(self):

        if self._initialized:
            return

        self._binary_context = BinaryContext.load_from_file(self._bcc_file_path)

        self._initialized = True

    def generate_ngrams(self, window_size: int = DEFAULT_WINDOW_SIZE) -> List[List[bytes]]:
        self.initialize()

        binary_context = self._binary_context

        binary_bytes: List[bytes] = []

        # Extracting the opcode bytes from the binary context
        for function_context in binary_context.function_contexts:
            for basic_block_context in function_context.basic_block_contexts:
                for instruction_context in basic_block_context.native_instruction_contexts:
                    binary_bytes.extend([b for b in instruction_context.opcode_bytes])
                    pass

        # Extract the string representation of the binary bytes
        for binary_string in binary_context.string_refs.values():
            binary_bytes.extend([b for b in binary_string.encode()])

        # Extract the import name representation of the binary bytes
        for import_symbol in binary_context.import_symbols:
            import_name = import_symbol.import_name
            binary_bytes.extend([b for b in import_name.encode()])

        # Extract the defined data representation of the binary bytes
        for defined_data in binary_context.defined_data_map.values():
            binary_bytes.extend([b for b in defined_data.data_bytes])

        # Generating n-grams as lists of bytes using a sliding window
        ngram_list = [binary_bytes[i:i + window_size] for i in range(len(binary_bytes) - window_size + 1)]

        return ngram_list

    def generate_ngrams_as_ndarray(self, window_size: int = DEFAULT_WINDOW_SIZE) -> np.ndarray:

        # Generate n-grams using the generate_ngrams method
        ngram_list = self.generate_ngrams(window_size=window_size)

        # Convert list of byte lists to a NumPy array
        if len(ngram_list) > 0:
            # Flatten the list of byte lists, then reshape into the correct form
            flat_list = [byte for sublist in ngram_list for byte in sublist]
            ngrams_ndarray = np.array(flat_list, dtype=np.uint8).reshape(-1, window_size)
        else:
            # Return an empty array if no n-grams can be formed
            ngrams_ndarray = np.empty((0, window_size), dtype=np.uint8)

        return ngrams_ndarray

    @property
    def binary_context(self) -> BinaryContext:
        self.initialize()
        return self._binary_context


def main():
    bcc_file_path = TEST_BCC_FILE_PATH

    binary_ngram_generator = BCCNGramGenerator(bcc_file_path)

    ngram_list = binary_ngram_generator.generate_ngrams()

    logger.info(f"Generated ngram list for first {10} n-grams: {ngram_list[:10]}")

    ngrams_ndarray = binary_ngram_generator.generate_ngrams_as_ndarray()

    logger.info(f"Generated ngram ndarray shape: {ngrams_ndarray.shape}")

    # Print 10 n-grams from the ndarray
    num_ngram_to_display = min(50, ngrams_ndarray.shape[0])
    for i in range(num_ngram_to_display):
        logger.info(f"Generated n-gram {i}: {ngrams_ndarray[i]}")


if __name__ == "__main__":
    main()
