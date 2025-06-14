import argparse
import logging

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from logzero import logger

from lab_common.labs.lab3.string_search import StringSearch
from lab_common.nlp.tokenizer import Tokenizer

logger.setLevel(logging.INFO)

# Set log level for the binary context module to suppress unnecessary output
logging.getLogger("binarycontext").setLevel(logging.ERROR)

class SimpleAverageStringSearch(StringSearch):

    def __init__(self):
        super().__init__()

    def generate_string_embedding(self, string: str, model: Word2Vec, tokenizer: Tokenizer) -> np.ndarray:
        """
        Generate the string embedding for a given string using the specified Word2Vec model.

        :param string: The string to generate the embedding for.
        :param model: The Word2Vec model to use.
        :param tokenizer: The tokenizer to use.
        :return: The embedding as a numpy array, or None if no tokens are found.
        """

        """
        Steps:
            1. Tokenize the input string using the provided Tokenizer class.
               - The string is broken into individual words or tokens.
               
            2. Filter out tokens not in the Word2Vec model's vocabulary.
               - Retain only tokens present in the model.
               
            3. If no valid tokens are found, return the zero vector (np.zeros(model.vector_size)).
               
            4. Calculate the weighted average embedding of the tokens.
               - Obtain each token's embedding from the model.
               - Use a uniform weighting approach where each token contributes equally.
               
            5. Return the final embedding as a numpy array.
               - This represents the input string in the model's space
           
        Notes:
         * To get the embedding of a token from the Word2Vec model, use model.wv[token].
         
         * model.wv is a KeyedVectors object, which is a wrapper around the word vectors. To check if a token is present in
             the model, use the `in` operator: token in model.wv.
        
        """
        weighted_average_embedding = np.zeros(model.vector_size)

        ### YOUR CODE HERE ###



        ### END YOUR CODE HERE ###

        return weighted_average_embedding


def main():
    parser = argparse.ArgumentParser(description='Find closest strings using Word2Vec embeddings.')
    parser.add_argument('--search_string', type=str, default="invalid format specifier for char",
                        help='String to search for.')
    parser.add_argument('--similarity_threshold', type=float, default=0.1, help='Similarity threshold.')
    parser.add_argument('--n_closest', type=int, default=10, help='Number of closest strings to find.')
    args = parser.parse_args()

    string_search = SimpleAverageStringSearch()

    string_search.process_strings(args.search_string, args.similarity_threshold, args.n_closest)


if __name__ == '__main__':
    main()
