import argparse
import hashlib
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict

from logzero import logger
from gensim.models import Word2Vec
from tqdm import tqdm

from blackfyre.datatypes.contexts.binarycontext import BinaryContext
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.labs.lab12.constants import LAB_12_TEST_BENIGN_FOLDER, LAB_12_CACHE_FOLDER

from lab_common.nlp.tokenizer import Tokenizer
from labs.lab4.binary_rank import BinaryRankContext, compute_global_strings_ranks

logging.getLogger("binarycontext").setLevel(logging.WARN)


DEFAULT_TOLERANCE = 0.01


class StringIndexer:
    def __init__(self,
                 bcc_folder_path: str = LAB_12_TEST_BENIGN_FOLDER,
                 ):
        self._bcc_folder_path: str = bcc_folder_path

        self._bcc_files: Optional[List[str]] = None

        # Key is the  string that maps to a dictionary with the file_sha256 as the key and the rank as the value
        self._string_index: Optional[Dict[str, Dict[str, float]]] = None

        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self._bcc_files = [os.path.join(self._bcc_folder_path, f) for f in os.listdir(self._bcc_folder_path)]

        self._initialize_index()

        self._initialized = True

    def search(self, string: str, rank: float, tol: float = DEFAULT_TOLERANCE) -> List[str]:
        """

        String search function that returns a list of bcc hashes that contain the string with rank within the tolerance.

        :param string: String to be searched for.
        :param rank: Target rank for the string.
        :param tol: Tolerance for the rank.
        :return: List of bcc hashes that contain the string with rank within the tolerance.
        """

        self.initialize()

        if string not in self._string_index:
            return []

        bcc_files = set()

        for bcc_file, string_rank in self._string_index[string].items():
            if abs(string_rank - rank) <= tol:
                bcc_files.add(bcc_file)

        return list(bcc_files)

    def _initialize_index(self):

        # Compute the string index cache path as string hex hash of the folder path

        def path_to_hexdigest(file_path):
            # Create a SHA-256 hash object
            hasher = hashlib.sha256()

            # Update the hasher with the file path encoded as bytes
            hasher.update(file_path.encode('utf-8'))

            # Return the hexadecimal digest of the hash
            return hasher.hexdigest()

        index_cache_file_path = os.path.join(LAB_12_CACHE_FOLDER,
                                             f"string_index_{path_to_hexdigest(self._bcc_folder_path)}.pkl")

        if os.path.exists(index_cache_file_path):
            logger.info(f"Loading string index from {index_cache_file_path}")
            self._string_index = pickle.load(open(index_cache_file_path, "rb"))
            return

        pbar = tqdm(self._bcc_files, desc=f"Indexing strings in BCC files in folder {self._bcc_folder_path}",
                    unit="file")

        self._string_index = dict()

        for bcc_file in pbar:


            binary_rank_context = BinaryRankContext.from_bcc_file_path(bcc_file)

            string_rank_dict = compute_global_strings_ranks(binary_rank_context)

            binary_context = VexBinaryContext.load_from_file(bcc_file)
            bcc_file_hash = binary_context.sha256_hash

            StringIndexer.display_top_n_strings(string_rank_dict)

            for string, rank in string_rank_dict.items():
                if string not in self._string_index:
                    self._string_index[string] = dict()
                self._string_index[string][bcc_file_hash] = rank

        logger.info(f"Saving string index to {index_cache_file_path}")

        # Create the cache folder if it does not exist
        os.makedirs(LAB_12_CACHE_FOLDER, exist_ok=True)
        pickle.dump(self._string_index, open(index_cache_file_path, "wb"))

    @staticmethod
    def display_top_n_strings(string_rank_dict, n: int = 10):

        sorted_string_rank_dict = dict(sorted(string_rank_dict.items(), key=lambda item: item[1], reverse=True))
        for i, (string, rank) in enumerate(sorted_string_rank_dict.items()):

            logger.info(f"{i + 1}. {string} : {rank}")
            if i == n - 1:
                break

    @staticmethod
    def display_top_n_strings_from_bcc_file_path(bcc_file_path: str, n: int = 10):

        binary_rank_context = BinaryRankContext.from_bcc_file_path(bcc_file_path)

        string_rank_dict = compute_global_strings_ranks(binary_rank_context)

        StringIndexer.display_top_n_strings(string_rank_dict, n)


def main():
    parser = argparse.ArgumentParser(description='String Indexer')
    parser.add_argument('--bcc_folder_path', type=str, default=LAB_12_TEST_BENIGN_FOLDER,
                        help='Path to the BCC folder.')

    #  String to search for in the index
    parser.add_argument('--search_string', type=str, default='"Unicows.dll"',
                        help='String to search for.')
    # Rank of the string to search for
    parser.add_argument('--rank', type=float, default=0.12, help='Rank of the string to search for.')

    # Tolerance for the rank
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE, help='Tolerance for the rank.')
    args = parser.parse_args()

    string_indexer = StringIndexer(bcc_folder_path=args.bcc_folder_path)

    string_indexer.initialize()

    bcc_hashes = string_indexer.search(args.search_string, args.rank, args.tolerance)

    logger.info(f"Found {len(bcc_hashes)} BCC files containing the string '{args.search_string}' with rank {args.rank} "
                f"within the tolerance {args.tolerance}.")

    for index, bcc_hash in enumerate(bcc_hashes, start=1):
        logger.info(f"\t {index}. BCC hash: {bcc_hash}")


if __name__ == "__main__":
    main()
