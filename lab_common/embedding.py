from pathlib import Path
from typing import List

from blackfyre.common import BINARY_CONTEXT_CONTAINER_EXT
from blackfyre.datatypes.contexts.binarycontext import BinaryContext
from blackfyre.datatypes.contexts.functioncontext import FunctionContext
from lab_common.common import Label, EmbeddingType, DEFAULT_MIN_FUNCTION_SIZE, DEFAULT_MAX_FUNCTION_SIZE


class LabeledDataPoint(object):

    def __init__(self, label, embedding):
        self._label = label

        self._embedding = embedding

    @property
    def label(self):
        return self._label

    @property
    def embedding(self):
        return self._embedding


class Embedding(object):
    __slots__ = ['_embedding', '_label', '_type']

    def __init__(self, embedding, label: Label, embedding_type: EmbeddingType):
        self._embedding = embedding
        self._label = label
        self._type = embedding_type

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def label(self):
        return self._label

    @property
    def type(self):
        return self._type


class FunctionEmbedding(Embedding):
    __slots__ = ['_name', '_address', '_sha256_hash',
                 '_vex_binary_context']

    def __init__(self, embedding, label: Label, embedding_type: EmbeddingType,
                 name: str, address: int, sha256_hash: str):
        super().__init__(embedding, label, embedding_type)

        self._name = name

        self._address = address

        self._sha256_hash = sha256_hash

    @property
    def name(self):
        return self._name

    @property
    def sha256_hash(self):
        return self._sha256_hash

    @property
    def address(self):
        return self._address


class BinaryEmbedding(Embedding):
    __slots__ = ['_name', '_base_name', '_sha256_hash',
                 '_vex_binary_context']

    def __init__(self, embedding, label: Label, embedding_type: EmbeddingType,
                 name: str, base_name: str, sha256_hash: str):
        super().__init__(embedding, label, embedding_type)

        self._name = name

        self._base_name = base_name

        self._sha256_hash = sha256_hash

    @property
    def name(self):
        return self._name

    @property
    def sha256_hash(self):
        return self._sha256_hash

    @property
    def base_name(self):
        return self._base_name


def get_non_thunk_functions_within_file_size_range(binary_context: BinaryContext,
                                                   min_function_size=DEFAULT_MIN_FUNCTION_SIZE,
                                                   max_function_size=DEFAULT_MAX_FUNCTION_SIZE) -> List[
    FunctionContext]:
    functions = [function_context for function_context in binary_context.function_contexts
                 if not function_context.is_thunk and
                 max_function_size >= function_context.size >= min_function_size]

    return functions


def get_binary_context_files(binary_context_folder_path, max_bin_contexts_to_load=-1):
    binary_context_files = [str(path) for index, path in
                            enumerate(
                                Path(binary_context_folder_path).rglob(f'*.{BINARY_CONTEXT_CONTAINER_EXT}'))
                            if index < max_bin_contexts_to_load or max_bin_contexts_to_load==-1]

    return binary_context_files