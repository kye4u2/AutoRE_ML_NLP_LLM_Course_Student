import argparse
import copy
import os
from typing import Optional

import numpy as np

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import Label, EmbeddingType, ROOT_PROJECT_FOLDER_PATH
from lab_common.embedding import BinaryEmbedding
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer


class BinaryEmbeddingBuilder(object):

    __slots__ = ['_name', '_base_name', '_sha256_hash', '_embedding', '_embedding_type', '_binary_embedding',
                 '_label', '_sentence_summarizer','_initialized', '_binary_context_file_path']

    def __init__(self, binary_context_file_path: str,
                 label: Label,
                 sentence_summarizer: SentenceSummarizer = None,
                 embedding_type: EmbeddingType = EmbeddingType.CUSTOM):
        self._binary_context_file_path = binary_context_file_path

        self._sentence_summarizer: Optional[SentenceSummarizer] = sentence_summarizer

        self._initialized: bool = False

        self._label: Label = label

        self._embedding_type: EmbeddingType = embedding_type

        self._name: Optional[str] = None

        self._base_name: Optional[str] = None

        self._sha256_hash: Optional[str] = None

        self._binary_embedding: Optional[BinaryEmbedding] = None

    def initialize(self):
        if self._initialized:
            return self._initialized

        # Load the vex binary context from file
        vex_binary_context = VexBinaryContext.load_from_file(self._binary_context_file_path)

        # Note: Shallow copies of the attributes of the vex_binary_context, so we can delete the object. Otherwise,
        #       the vex_binary_context is tied to the lifetime of the references of th attributes
        self._name = copy.copy(vex_binary_context.name)

        self._base_name = copy.copy(vex_binary_context.base_name)

        self._sha256_hash = copy.copy(vex_binary_context.sha256_hash)

        self._binary_embedding: BinaryEmbedding = self._generate_embedding(vex_binary_context)

        # delete the object
        del vex_binary_context

        self._initialized = True

    def _generate_embedding(self, binary_context: VexBinaryContext) -> BinaryEmbedding:

        feature_vector = self._build_feature_vector(binary_context)

        binary_embedding = BinaryEmbedding(feature_vector,
                                           self._label,
                                           self._embedding_type,
                                           self._name,
                                           self._base_name,
                                           self._sha256_hash)

        return binary_embedding

    def _build_feature_vector(self, binary_context: VexBinaryContext) -> np.ndarray:
        raise NotImplementedError

    @property
    def embedding(self):
        self.initialize()
        return self._binary_embedding



