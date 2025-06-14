import argparse
import logging
from typing import Optional, Dict, List

import numpy as np
from gensim.models import Word2Vec, KeyedVectors

from logzero import logger

from lab_common.labs.lab3.string_search import StringSearch
from lab_common.nlp.tokenizer import Tokenizer

logger.setLevel(logging.INFO)


class InverseFrequencyStringSearch(StringSearch):

    def __init__(self):

        super().__init__()

    def generate_string_embedding(self, string: str, model: Word2Vec, tokenizer: Tokenizer) -> np.ndarray:
        """
        Generate the string embedding for a given string using the specified Word2Vec model.

        :param string: The string to generate the embedding for.
        :param model: The Word2Vec model to use.
        :param tokenizer: The tokenizer to use.
        :return: The embedding as a numpy array.
        """

        """
        Steps:
            1. Initialize a weighted average embedding vector of zeros.
            
            2. Tokenize the input string using the Tokenizer class.
               - Break the string into individual words or tokens.
               
            3. Filter out tokens not in both the tokenizer's token_freq_dict and Word2Vec model's vocabulary.
               - token_freq_dict is a property of the Tokenizer class which holds the frequency of tokens.
               
            4. If no valid tokens are found after filtering, return the zero vector.
            
            5. Compute the token inverse frequency for each valid token using token_freq_dict.
               - Use the formula: alpha / (alpha + token frequency in token_freq_dict).
               - Normalize the weights so they sum up to 1.
               
            6. Summarize the token vectors using the normalized inverse frequency weights.
               - Multiply each token's embedding by its weight and sum them up.
               
            7. Return the final embedding as a numpy array.
        
        Notes:
         *  To get the embedding of a token from the Word2Vec model, use model.wv[token].
         *  model.wv is a KeyedVectors object, which is a wrapper around the word vectors. To check if a token is present in
             the model, use the `in` operator: token in model.wv.
         *  See Lecture 5 (Introduction to NLP) slides for more information on inverse frequency weighting.
        """

        weighted_average_embedding = np.zeros(model.vector_size)

        ### YOUR CODE HERE ###




        ### END YOUR CODE HERE ###

        return weighted_average_embedding


def main():
    parser = argparse.ArgumentParser(description='Find closest strings using Word2Vec embeddings.')
    parser.add_argument('--search_string', type=str, default="invalid format specifier for char",
                        help='String to search for.')
    parser.add_argument('--similarity_threshold', type=float, default=.1, help='Similarity threshold.')
    parser.add_argument('--n_closest', type=int, default=10, help='Number of closest strings to find.')
    args = parser.parse_args()

    string_search = InverseFrequencyStringSearch()

    string_search.process_strings(args.search_string, args.similarity_threshold, args.n_closest)


if __name__ == '__main__':
    main()
