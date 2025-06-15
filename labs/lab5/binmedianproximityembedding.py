import argparse
import logging
import os
import pdb
import timeit
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
from logzero import logger
import numpy as np

from blackfyre.common import IRCategory, BINARY_CONTEXT_CONTAINER_EXT
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext
from blackfyre.datatypes.contexts.vex.vexinstructcontext import VexInstructionContext
from blackfyre.datatypes.importsymbol import ImportSymbol

from lab_common.binembeddingbuilder import BinaryEmbeddingBuilder
from lab_common.common import Label, EmbeddingType, ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer
from labs.lab4.binary_rank import BinaryRankContext, BasicBlockRankContext, compute_median_proximity_weights

logging.getLogger("binarycontext").setLevel(logging.WARN)

DEFAULT_TEST_BCC_FILE_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                          "lab_datasets",
                                          "lab6",
                                          "benign",
                                          "3f1bfc4be00fa0c07778a986e81bf21453ba45f76fecd261d7_3f1bfc4be00fa0c07778a986e81bf21453ba45f76fecd261d7e88e7db76a15a1.bcc")


# logger.setLevel(logging.DEBUG)


class MedianProximityEmbeddingBuilder(BinaryEmbeddingBuilder):
    __slots__ = ['_vex_binary_context', '_function_analysis_contexts', '_bb_analysis_contexts',
                 '_binary_import_execution_percentage_dict', '_binary_import_weight_dict',
                 '_binary_context_file_path']

    def __init__(self, binary_context_file_path, label: Label, sentence_summarizer: SentenceSummarizer):
        super().__init__(binary_context_file_path, label, sentence_summarizer, EmbeddingType.MEDIAN_PROXIMITY)

        if not isinstance(sentence_summarizer, SentenceSummarizer):
            raise ValueError(f"Expected to be of type  {type(SentenceSummarizer)}.. "
                             f"Received type of '{type(sentence_summarizer)}'")

        self._initialized = False

        self._binary_context_file_path = binary_context_file_path

    def _build_feature_vector(self, vex_binary_context: VexBinaryContext) -> np.ndarray:

        """
            Creates a feature vector from a VexBinaryContext, emphasizing strings and import functions. This process
            assigns weights based on median proximity to assess their relevance, integrating them into a unified feature set.

            Steps:
            1. Aggregate collective ranks for strings and imports by analyzing each basic block.
               This includes summing ranks for strings and import function calls within the basic blocks,
               indicating the significance of each feature within the binary's execution flow.

            2. Compute median proximity weights for strings and import function names using the `_compute_median_proximity_weights`
               function, focusing on:
               - Identifying the median rank among strings and imports to anchor the weighting process.
               - Weighting features based on their distance from the median, prioritizing those more representative of
                 the binary's feature usage.

            3. Summarize each string and import function into a vector representation using sentence summarization, then
               apply the computed weights to these vectors to reflect their importance.

            4. Accumulate weighted vectors into `string_feature_vector` and `import_feature_vector`, ensuring
               proportional representation of each feature's relevance.

            Notes:
            - `string_feature_vector` and `import_feature_vector` are pre-initialized according to the sentence
               summarizer's vector size and will be updated with weighted feature information.
            - Identification of import functions within a basic block is based on matching the call target name with an
              import function's name.

            Parameters:
            - vex_binary_context (VexBinaryContext): Contains basic blocks, strings, and import symbols necessary for constructing the feature vector.

            Returns:
            - np.ndarray: A unified feature vector representing the binary, incorporating weighted information from strings and import functions based on their median proximity and relevance.
        """

        feature_vector = None

        ### YOUR CODE HERE ###




        ### END YOUR CODE HERE ###

        return feature_vector


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Binary Embedding Builder')
    parser.add_argument('-f', '--file', type=str, default=DEFAULT_TEST_BCC_FILE_PATH,
                        help='Path to the BCC file. Default is set to the DEFAULT_TEST_BCC_FILE_PATH.')

    # Assuming 'Label' is an enum with predefined values
    label_choices = [label.name for label in Label]  # List all enum names
    parser.add_argument('-l', '--label', type=str, default=Label.BENIGN.name, choices=label_choices,
                        help='Label for the embedding. Defaults to "benign".')

    # Parse arguments
    args = parser.parse_args()

    # Extract the file path and label
    bcc_file_path = args.file
    label = Label[args.label]  # Access enum by name

    # Create and initialize the BinaryEmbeddingBuilder
    sentence_summarizer = SentenceSummarizer()
    binary_embedding_builder = MedianProximityEmbeddingBuilder(bcc_file_path, label, sentence_summarizer)
    binary_embedding_builder.initialize()

    # Get the embedding (optional, depends on what you want to do next)
    embedding = binary_embedding_builder.embedding


if __name__ == '__main__':
    main()
