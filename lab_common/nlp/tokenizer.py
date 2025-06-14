import argparse
import gzip
import logging
import os
import re
import timeit
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
import wordninja

from blackfyre.common import BINARY_CONTEXT_CONTAINER_EXT
from blackfyre.datatypes.contexts.binarycontext import BinaryContext
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.utils import setup_custom_logger


from lab_common.common import ROOT_PROJECT_FOLDER_PATH

logger = setup_custom_logger(os.path.splitext(os.path.basename(__file__))[0])
# logger.setLevel(logging.DEBUG)
# logging.getLogger("binarycontext").setLevel(logging.WARN)

MIN_TOKEN_LENGTH = 3

WORD_NINJA_MODEL_FILE_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common/nlp/word_ninja_re_model.txt.gz")

VOCAB_FILE_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common/nlp/vocab.txt")

DEFAULT_LRU_CACHE_MAX_SIZE = 1024

class Tokenizer(object):

    def __init__(self, min_token_length=MIN_TOKEN_LENGTH,
                 word_ninja_model_file_path=WORD_NINJA_MODEL_FILE_PATH,
                    vocab_file_path=VOCAB_FILE_PATH):

        self._initialize = False

        self._word_ninja_lm = None

        self._min_token_length = min_token_length

        self._word_ninja_model_file_path = word_ninja_model_file_path

        self._vocab_file_path = vocab_file_path

        self._token_freq_dict: Dict[str, float] = None

    def initialize(self):

        if self._initialize:
            return

        self._word_ninja_lm = wordninja.LanguageModel(self._word_ninja_model_file_path)

    @lru_cache(maxsize=DEFAULT_LRU_CACHE_MAX_SIZE)
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the given text using a Word Ninja language model after preprocessing.

        Parameters:
        text (str): The text string to be tokenized.

        Returns:
        List[str]: A list of tokens extracted from the input text.
        """

        self.initialize()

        pre_processed_text = Tokenizer.pre_process_binary_text(text, self._min_token_length)

        tokens = [token for token in self._word_ninja_lm.split(pre_processed_text)
                  if len(token) >= self._min_token_length]

        return tokens

    def _load_token_freq(self):
        # Read the vocab from disk
        with open(self._vocab_file_path, 'r') as file_handler:
            vocabs = file_handler.readlines()

        # vocab has the following format: 'word frequency\n'
        token_freq_dict = {vocab.split(" ")[0]: int(vocab.split(" ")[1]) for vocab in vocabs}

        return token_freq_dict

    @classmethod
    def _camel_case_split(cls, word: str):

        camel_split = re.findall("[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", word)

        # if we can't find the camel split, then return the original word
        camel_split = camel_split if len(camel_split) > 0 else word

        return camel_split

    @classmethod
    def _flatten_list(cls, list_of_lists):

        list_of_lists = [item if type(item) is list else [item] for item in list_of_lists]

        return [item for sublist in list_of_lists for item in sublist]

    @classmethod
    def pre_process_binary_text(cls, binary_text: str, min_token_length):

        words = [word for word in re.split(r'[ \'\d!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]', binary_text)
                 if len(word) >= min_token_length]

        # Camel case extraction
        words = [cls._camel_case_split(word) for word in words]

        # Flatten the list
        words = cls._flatten_list(words)

        # Filter out tokens that do not exceed the min length
        words = [word.strip() for word in words if len(word.strip()) >= min_token_length]

        # Join the list into one string
        pre_processed_text = " ".join(words)

        # Convert the text to lower case
        pre_processed_text = pre_processed_text.lower()

        logger.debug(f"'{binary_text}' -->  '{pre_processed_text}'")

        return pre_processed_text

    @property
    def token_freq_dict(self):
        if self._token_freq_dict is None:
            self._token_freq_dict = self._load_token_freq()

        return self._token_freq_dict


def main():
    parser = argparse.ArgumentParser(description="Tokenize a given string using custom settings.")

    # Required argument for the string to be tokenized
    # Optional argument for the string to be tokenized with a default value
    parser.add_argument('--string_to_tokenize', type=str, default="helloworldprint",
                        help='String that needs to be tokenized')

    # Optional arguments with default values
    parser.add_argument('--min_token_length', type=int, default=MIN_TOKEN_LENGTH, help='Minimum token length')
    parser.add_argument('--word_ninja_model_file_path', type=str, default=WORD_NINJA_MODEL_FILE_PATH, help='Path to the Word Ninja model file')
    parser.add_argument('--vocab_file_path', type=str, default=VOCAB_FILE_PATH, help='Path to the vocab file')

    args = parser.parse_args()

    tokenizer = Tokenizer(
        min_token_length=args.min_token_length,
        word_ninja_model_file_path=args.word_ninja_model_file_path,
        vocab_file_path=args.vocab_file_path
    )

    tokens = tokenizer.tokenize(args.string_to_tokenize)
    print(tokens)

if __name__ == "__main__":
    main()