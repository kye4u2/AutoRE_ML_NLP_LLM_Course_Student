import argparse
import logging
import os
from typing import Optional, Dict, List

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm
import logzero
from logzero import logger

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH, compute_cosine_distance
from lab_common.nlp.tokenizer import Tokenizer

logger.setLevel(logging.INFO)


class StringSearch(object):

    def __init__(self):
        pass

    def load_word2vec_model(self, model_path: str) -> Word2Vec:
        """Load the Word2Vec model from the given file path."""
        return Word2Vec.load(model_path)

    def generate_string_embedding(self, string: str, model: Word2Vec, tokenizer: Tokenizer) -> np.ndarray:
        """
        Generate the string embedding for a given string using the specified Word2Vec model.

        :param string: The string to generate the embedding for.
        :param model: The Word2Vec model to use.
        :param tokenizer: The tokenizer to use.
        :return: The embedding as a numpy array.
        """

        raise NotImplementedError("generate_string_embedding() must be implemented by a subclass.")



    def find_closest_strings(self,
                             search_string: str,
                             string_embedding_map: Dict[str, np.ndarray],
                             threshold: float,
                             n: int,
                             word2vec_model,
                             tokenizer: Tokenizer) -> Dict[str, float]:
        """
        Find the N closest strings to the search string within a given threshold.

        :param word2vec_model: Word2Vec model.
        :param search_string: The string to search for.
        :param string_embedding_map: A dictionary mapping strings to their embeddings.
        :param threshold: The similarity threshold for considering a string as close.
        :param n: The number of closest strings to find.
        :return: A dictionary of the N closest strings and their similarity scores.
        """
        search_string_embedding = self.generate_string_embedding(search_string, word2vec_model, tokenizer)
        if search_string_embedding is None:
            return {}

        closest_strings = {}
        for string, string_embedding in string_embedding_map.items():
            # cosine distance
            distance = compute_cosine_distance(search_string_embedding, string_embedding)

            # Display the distance between the search string and the current string
            logger.debug(f"Distance between '{search_string[:80]}' and '{string[:80]}': {distance:.3f}")

            if distance < threshold:
                closest_strings[string] = distance

        # Sort the dictionary by distance and return the top N
        return dict(sorted(closest_strings.items(), key=lambda item: item[1])[:n])

    def format_closest_strings(self, closest_string_dict: Dict[str, float], search_string: str):
        """
        Format the closest strings and their similarity scores into a structured table,
        capping the string display length to a maximum of 80 characters.

        :param closest_string_dict: Dictionary of strings and their similarity scores.
        :param search_string: The search string used for finding closest strings.
        """
        if not closest_string_dict:
            logger.info(f"No close strings found for '{search_string}'.")
            return

        # Cap the maximum length of the string for display
        max_display_length = 80

        # Sort the closest strings dictionary by similarity score
        sorted_closest_string_dict = dict(sorted(closest_string_dict.items(), key=lambda item: item[1]))

        # Prepare the header of the table
        header = f"{'String'.ljust(max_display_length)} | Distance"
        logger.info(f"\nClosest strings to '{search_string}':")
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

        # Log each string and its distance in a table format, capping string length
        for string, distance in sorted_closest_string_dict.items():
            display_string = (
                string if len(string) <= max_display_length else string[:max_display_length - 3] + '...').ljust(
                max_display_length)
            logger.info(f"{display_string} | {distance:.3f}")

    def process_strings(self, search_string: str, similarity_threshold: float, n_closest: int):
        """
        Process the data and find the closest strings.
        """
        word2vec_model: Word2Vec = self.load_word2vec_model(
            os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common/nlp/word2vec.model"))

        BCC_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets/lab3")
        bcc_file_paths = [os.path.join(BCC_FOLDER_PATH, file_name) for file_name in os.listdir(BCC_FOLDER_PATH)]

        tokenizer = Tokenizer()

        string_embedding_map = {}
        bcc_strings = []
        for index, bcc_file_path in enumerate(
                tqdm(bcc_file_paths, desc="Extracting strings from bccs", unit="bcc_file")):

            vex_binary_context = VexBinaryContext.load_from_file(bcc_file_path)
            current_bcc_strings = list(vex_binary_context.string_refs.values())
            bcc_strings.extend(current_bcc_strings)
            for current_bcc_string in current_bcc_strings:
                logger.debug(f"Processing string {current_bcc_string}")

            for current_bcc_string in current_bcc_strings:
                string_embedding = self.generate_string_embedding(current_bcc_string, word2vec_model, tokenizer)
                if string_embedding is not None:
                    string_embedding_map[current_bcc_string] = string_embedding

        closest_string_dict = self.find_closest_strings(search_string, string_embedding_map, similarity_threshold,
                                                        n_closest,
                                                        word2vec_model, tokenizer)

        self.format_closest_strings(closest_string_dict, search_string)


def main():
    parser = argparse.ArgumentParser(description='Find closest strings using Word2Vec embeddings.')
    parser.add_argument('--search_string', type=str, default="invalid format specifier for char",
                        help='String to search for.')
    parser.add_argument('--similarity_threshold', type=float, default=.1, help='Similarity threshold.')
    parser.add_argument('--n_closest', type=int, default=10, help='Number of closest strings to find.')
    args = parser.parse_args()

    string_search = StringSearch()

    string_search.process_strings(args.search_string, args.similarity_threshold, args.n_closest)


if __name__ == '__main__':
    main()
