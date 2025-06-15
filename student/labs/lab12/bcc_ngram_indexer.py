import argparse
import os
import re
import time
from collections import defaultdict
from typing import Dict, Optional, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm



from logzero import logger

from lab_common.labs.lab12.bcc_ngram_generator import BCCNGramGenerator
from lab_common.labs.lab12.constants import LAB_12_TEST_BENIGN_FOLDER, TEST_BCC_FILE_PATH

DEFAULT_WINDOW_SIZE = 5

DEFAULT_K_NEAREST_NEIGHBORS = 100


class BCCNgramIndexer:
    def __init__(self,
                 bcc_folder_path: str = LAB_12_TEST_BENIGN_FOLDER,
                 window_size: int = DEFAULT_WINDOW_SIZE):
        self._bcc_folder_path = bcc_folder_path
        self._bcc_files = []
        self._window_size = window_size
        self._ngram_index: Optional[faiss.IndexLSH] = None
        self._initialized = False
        self._lsh_num_bits = 20
        self._ngram_index_map: Dict[int, str] = {}  # Map lsh index to the sha256 hash

    def initialize(self):

        if self._initialized:
            return

        def alphanum_key(s):
            # Split the string into list of strings and ints
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        self._bcc_files = [
            os.path.join(self._bcc_folder_path, f)
            for f in sorted(os.listdir(self._bcc_folder_path), key=alphanum_key)
        ]

        self._ngram_index = self._initialize_ngram_index(self._ngram_index_map, self._lsh_num_bits)

        self._initialized = True

    def _initialize_ngram_index(self, ngram_index_map: Dict[int, str] , lsh_num_bits: int) -> faiss.IndexLSH:
        """
        Objective:
        To initialize and populate a Locality-Sensitive Hashing (LSH) index with n-grams extracted from Binary Context
        Container (BCC) files. This process involves generating n-grams from each file, hashing them, and adding them
         to the LSH index for efficient similarity searches.

        Parameters:
        - ngram_index_map (Dict[int, str]): A mapping from n-gram indices to their corresponding BCC file hashes,
                                           to track the origin of each n-gram.
        - lsh_num_bits (int): The number of bits to use for the LSH index, determining its precision and storage requirements.
                              This affects the granularity of hash collisions and thus the balance between the index's
                              accuracy and its memory footprint.

        Return Type:
        - faiss.IndexLSH: An LSH index populated with n-grams from the BCC files, ready for performing fast similarity searches.
                        This index allows for efficient querying of similar n-grams based on their hash values.

        Steps:
        1. Iterate over all BCC files in the specified directory,
        2. For each file, generate n-grams using the specified window size and compute the SHA256 hash of the file's binary context.
        3. Store the n-grams in the dictionary, keyed by the file's SHA256 hash.
        4. Determine the dimension (`d`) of the n-grams by examining the shape of the first n-gram list in the dictionary.
           This dimension is crucial for configuring the LSH index.
        5. Create an LSH index using the determined dimension (`d`) and the specified number of bits (`lsh_num_bits`).
        6. For each set of n-grams in the dictionary, map the n-gram's index (i.e. the row where the n-gram is in the ngram list)
           to its originating BCC file's hash,
           reshape each n-gram (i.e. `ngram.reshape(1, d)`)  for compatibility with the LSH index, and add it to the index.
        7. Return the populated LSH index. This index is now ready to be used for similarity searches among the indexed n-grams,
           facilitating efficient retrieval of similar binary contexts.
        """

        ngram_dict: Dict[str, np.ndarray] = {}  # Hash -> ngram  (Will need to modify to store the ngrams for each file)

        bcc_folder_path = self._bcc_folder_path  # READ-ONLY variable for the folder path

        ngram_index: Optional[
                     faiss.IndexLSH:] = None  # Placeholder for the LSH index. Will need to modify this variable

        ### YOUR CODE HERE ###



        ### END YOUR CODE ###

        return ngram_index

    def search(self, ngram: np.ndarray, k_nearest_neighbors: int = DEFAULT_K_NEAREST_NEIGHBORS) -> list[str]:
        self.initialize()

        if self._ngram_index is None:
            logger.warning("The ngram index has not been initialized.")
            return []


        D, I = self._ngram_index.search(ngram, k_nearest_neighbors)

        # Filtering for nearest neighbors where distance is 0
        zero_distance_indices = I[D == 0]

        nearest_neighbor_hashes = [self._ngram_index_map[i] for i in zero_distance_indices]

        return nearest_neighbor_hashes

    def search_all(self,
                   ngrams: np.ndarray,
                   k_nearest_neighbors: int = DEFAULT_K_NEAREST_NEIGHBORS) -> Dict[ Tuple[np.ndarray], List[str]]:
        """

        Objective: To efficiently search for and identify the exact matches (distance of 0) of each n-gram within a given array against
                     a pre-indexed database, returning a mapping of each n-gram to the list of BCC hashes where
                     it is found. This function aims to facilitate precise malware pattern identification by
                     leveraging the specificity of n-gram matches.

        Parameters:
        - ngrams (np.ndarray): An array of n-grams represented as numerical vectors, intended for exact match searches.
                              Each n-gram in this array is a candidate for matching against the indexed data.
        - k_nearest_neighbors (int, optional): Specifies the number of nearest neighbors to be considered for each
                                               n-gram in the search process. Defaults to a predefined constant value
                                               that optimizes search specificity and performance.

        Returns:
        - Dict[Tuple[np.ndarray], List[str]]: A dictionary where each key is an n-gram (converted to a tuple for hashability)
                and each value is a list of unique BCC hashes (strings) identifying where the n-gram has been exactly found.
                This structure enables direct association of n-grams with their occurrences in indexed content.

        Key Steps:
        1. Execute the search on the n-gram index with the provided n-grams and k_nearest_neighbors parameter,
           retrieving distances and indices of nearest neighbors.
        2. Iterate over each n-gram and its search results:
           a. Convert the n-gram from a numpy array to a tuple for use as a dictionary key.
           b. Identify indices with a distance of 0, indicating an exact match.
           c. Retrieve BCC hashes for these indices and append them to the list associated with the n-gram key in
              `ngram_to_bcc_hashes_dict`.
        """
        self.initialize()

        ngram_to_bcc_hashes_dict: Dict[
            Tuple[np.ndarray], List[str]] = {}  # Will need to modify to store the ngrams for each file

        ### YOUR CODE HERE ###


        ### END YOUR CODE ###

        return ngram_to_bcc_hashes_dict

    @property
    def window_size(self) -> int:
        return self._window_size


def test_ngram_search(bcc_ngram_indexer, test_ngram, expected_hashes):
    test_ngram = np.array(test_ngram).reshape(1, bcc_ngram_indexer.window_size)
    unique_nearest_neighbor_hashes = list(set(bcc_ngram_indexer.search(test_ngram)))

    logger.info("-" * 40)
    logger.info(f"Test N-Gram: {test_ngram.flatten().tolist()}")

    # Test result
    test_passed = set(unique_nearest_neighbor_hashes) == set(expected_hashes)

    # More explicit pass/fail status logging
    if test_passed:
        logger.info("\tTest Status: PASSED - The expected and actual sets of nearest neighbors are equal.")
    else:
        logger.error("\tTest Status: FAILED - The sets of nearest neighbors do not match.")
        logger.info(f"\t\tUnique nearest neighbors found: {', '.join(unique_nearest_neighbor_hashes)}")
        logger.info(f"\t\tExpected nearest neighbors: {', '.join(expected_hashes)}")

    logger.info("-" * 40)

    assert test_passed, "Test failed. The sets of nearest neighbors do not match."


def test_index(bcc_folder, window_size):
    bcc_ngram_indexer = BCCNgramIndexer(bcc_folder_path=bcc_folder, window_size=window_size)

    # Initialize indexer and measure elapsed time
    start_time = time.time()
    bcc_ngram_indexer.initialize()
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for indexing: {elapsed_time:.2f} seconds")

    # Define test cases as tuples of candidate ngram and expected hashes
    test_cases = [
        ([3, 0, 145, 248, 95], ['2a1d8692f445791dc9dc9700e11f2ce68fce9ac5ad2abd56aa0f41f1047b38f1']),
        ([23, 0, 148, 40, 0], ['2a1d8692f445791dc9dc9700e11f2ce68fce9ac5ad2abd56aa0f41f1047b38f1']),
        ([148, 3, 0, 0, 20], ['2a1d8692f445791dc9dc9700e11f2ce68fce9ac5ad2abd56aa0f41f1047b38f1'])
    ]

    logger.info("Testing N-Gram Search")
    # Test each case
    for test_ngram, expected_hashes in test_cases:
        test_ngram_search(bcc_ngram_indexer, test_ngram, expected_hashes)


def test_search_all(bcc_folder, window_size):
    # Test

    bcc_ngram_indexer = BCCNgramIndexer(bcc_folder_path=bcc_folder, window_size=window_size)
    # Initialize indexer and measure elapsed time
    start_time = time.time()
    bcc_ngram_indexer.initialize()
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for indexing: {elapsed_time:.2f} seconds")

    logger.info(f"Generating ngrams from {TEST_BCC_FILE_PATH} for testing search_all(). Can take around 30 seconds...")
    binary_ngram_generator = BCCNGramGenerator(TEST_BCC_FILE_PATH)
    ngrams = binary_ngram_generator.generate_ngrams_as_ndarray(window_size)

    ngram_to_bcc_hashes_dict = bcc_ngram_indexer.search_all(ngrams)


    # Total unique n-grams
    unique_ngrams = set([tuple(ngram) for ngram in ngrams])
    num_unique_ngrams = len(unique_ngrams)

    assert num_unique_ngrams == 21270, f"The number of unique n-grams is incorrect. Expected 21270, but received {num_unique_ngrams}."

    logger.info(f"Total unique n-grams: {num_unique_ngrams}")

    # Ngrams that are not in the ngram_to_bcc_hashes_dict
    ngrams_with_no_nearest_neighbor = [ngram for ngram in unique_ngrams
                                       if ngram not in ngram_to_bcc_hashes_dict]

    # number of unique n-grams with no nearest neighbor
    num_ngrams_with_no_nearest_neighbor = len(ngrams_with_no_nearest_neighbor)
    logger.info(f"Number of n-grams with no nearest neighbor: {num_ngrams_with_no_nearest_neighbor}")

    assert num_ngrams_with_no_nearest_neighbor == 1, f"The number of n-grams with no nearest neighbor is incorrect. Expected 1, but received {num_ngrams_with_no_nearest_neighbor}."

    # Print the ngrams that have no nearest neighbor
    for ngram in ngrams_with_no_nearest_neighbor:
        logger.info(f"ngram: {ngram} has no nearest neighbor")

    expected_ngrams_with_no_nearest_neighbor = [(34, 0, 185, 104, 2)]
    assert ngrams_with_no_nearest_neighbor == expected_ngrams_with_no_nearest_neighbor,\
        (f"The n-grams with no nearest neighbor are incorrect. Expected {expected_ngrams_with_no_nearest_neighbor}"
         f" but received {ngrams_with_no_nearest_neighbor}.")

    num_ngrams_with_exact_matches = len(ngram_to_bcc_hashes_dict)

    expected_num_ngrams_with_exact_matches = 21269
    assert num_ngrams_with_exact_matches == expected_num_ngrams_with_exact_matches, \
        (f"The number of n-grams with exact matches is incorrect ({num_ngrams_with_exact_matches})."
         f" Expected {expected_num_ngrams_with_exact_matches}")

    logger.info(f"Number of n-grams with exact matches: {num_ngrams_with_exact_matches}")

    logger.info("Testing search_all() logic PASSED successfully.")

    # # Show the first 10 n-grams that have exact matches
    # for ngram, bcc_hashes in list(ngram_to_bcc_hashes_dict.items())[:30]:
    #     logger.info(f"n-gram: {ngram} -> bcc_hashes: {bcc_hashes}")



def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Index BCC files using ngrams')
    parser.add_argument('-f', '--bcc_folder', type=str, default=LAB_12_TEST_BENIGN_FOLDER,
                        help='Path to the BCC folder.')
    parser.add_argument('-w', '--window_size', type=int, default=DEFAULT_WINDOW_SIZE, help='Window size for ngrams.')

    parser.add_argument("-ti", "--test_index", action="store_true",
                        help="Test the _initialize_ngram_index() logic.")

    parser.add_argument("-ts", "--test_search_all", action="store_true",
                        help="Test the search_all() logic.")

    args = parser.parse_args()

    # Check if all the flags set and raise an error
    if args.test_index and args.test_search_all:
        raise ValueError("Both test_index and test_search_all flags cannot be set at the same time.")

    if args.test_index:
        test_index(args.bcc_folder, args.window_size)

    elif args.test_search_all:
        test_search_all(args.bcc_folder, args.window_size)




if __name__ == '__main__':
    main()
