
import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from gensim.models import Word2Vec

from lab_common.common import ROOT_PROJECT_FOLDER_PATH, DEFAULT_LRU_CACHE_MAX_SIZE
from lab_common.nlp.tokenizer import Tokenizer

MIN_TOKEN_LENGTH = 3


class SentenceSummarizer(object):

    def __init__(self):

        self._word2vec_model = None

        self._initialized: bool = False

        self._language_model = None

        self._tokenizer: Optional[Tokenizer] = None

        self._token_freq_dict: Dict[str, float] = dict()

        self._vector_size: Optional[int] = None

    def initialize(self):

        if self._initialized:
            return

        # Initialize the Tokenizer
        self._tokenizer = Tokenizer()
        self._tokenizer.initialize()

        self._word2vec_model: Word2Vec = Word2Vec.load(
            os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common/nlp/word2vec.model"))

        # Initialize the token frequency dictionary
        self._token_freq_dict = self._tokenizer.token_freq_dict

        self._vector_size = self._word2vec_model.vector_size

        self._initialized = True

    @lru_cache(maxsize=DEFAULT_LRU_CACHE_MAX_SIZE)
    def summarize(self, sentence: str) -> np.ndarray:

        self.initialize()

        # Get tokens
        tokens = [token for token in self._tokenizer.tokenize(sentence) if len(token) >= MIN_TOKEN_LENGTH]

        summarization = self._summarize_with_gensim(tokens)


        return summarization



    def _summarize_with_gensim(self, tokens: List[str]) -> np.ndarray:

        # Make sure the token is in our vocab and in the language model
        tokens = [token for token in tokens if token in self._token_freq_dict and token in self._word2vec_model.wv]

        token_inverse_freq_dict: Dict[str, float] = dict()

        alpha = .001  # constant used for inverse frequency

        # Compute the token inverse frequency
        for token in set(tokens):
            token_inverse_freq_dict[token] = alpha / (alpha + self._token_freq_dict[token])

        # Normalize the weights
        total_weights = sum(list(token_inverse_freq_dict.values()))
        for token in set(tokens):
            token_inverse_freq_dict[token] *= (1 / total_weights)

        # Summarize the token vectors using the normalized inverse frequency weights
        token_summarization = np.zeros(self._vector_size)
        num_tokens = len(tokens)
        for token in tokens:
            token_summarization += (1 / num_tokens) * self._word2vec_model.wv[token] * token_inverse_freq_dict[token]

        return token_summarization

    @property
    def vector_size(self):
        self.initialize()
        return self._vector_size

    @property
    def tokenizer(self) -> Tokenizer:
        self.initialize()
        return self._tokenizer

    @property
    def word2vec_model(self):
        self.initialize()
        return self._word2vec_model


def main():
    sentence_summarizer = SentenceSummarizer()

    sentence_summarizer.initialize()

    embedding = sentence_summarizer.summarize("GetWorld")

    print(embedding)


if __name__ == "__main__":
    main()
