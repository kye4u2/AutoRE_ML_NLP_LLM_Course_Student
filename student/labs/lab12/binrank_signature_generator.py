import argparse
import os
from typing import Optional, Dict, Tuple, List

from logzero import logger

from lab_common.labs.lab12.bcc_ngram_generator import LAB_12_DATASET
from lab_common.labs.lab12.constants import LAB_12_TEST_BENIGN_FOLDER
from lab_common.llm.client import llm_completion
from lab_common.labs.lab12.string_indexer import StringIndexer
from labs.lab4.binary_rank import compute_global_strings_ranks, BinaryRankContext

TEST_BCC_FILE_PATH = os.path.join(LAB_12_DATASET,
                                  "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")


class BinaryRankSignatureGenerator:
    def __init__(self,
                 bcc_file_path: str,
                 bcc_folder_path: str = LAB_12_TEST_BENIGN_FOLDER,
                 min_string_length: int = 3,
                 top_n: int = 10,
                 max_strings_in_signature: int = 10):

        self._bcc_file_path = bcc_file_path

        self._bcc_folder_path = bcc_folder_path

        self._min_string_length = min_string_length

        self._max_strings_in_signature = max_strings_in_signature

        self._top_n = top_n

        self._tolerance = 0.2

        self._bcc_string_rank_dict: Optional[Dict[str, float]] = None

        self._string_indexer: Optional[StringIndexer] = None

        self._initialized = False

    def initialize(self):

        if self._initialized:
            return


        binary_rank_context = BinaryRankContext.from_bcc_file_path(self._bcc_file_path)

        self._bcc_string_rank_dict = compute_global_strings_ranks(binary_rank_context)

        # Filter out strings that have less than 3 characters
        self._bcc_string_rank_dict = {string: rank for string, rank in self._bcc_string_rank_dict.items() if
                                      len(string) >= self._min_string_length}

        self._bcc_string_rank_dict = sorted(self._bcc_string_rank_dict.items(), key=lambda x: x[1], reverse=True)

        self._string_indexer = StringIndexer(bcc_folder_path=self._bcc_folder_path)

        self._initialized = True

    def generate_signature(self):

        self.initialize()


        PROMPT  = """
          
        **Objective**: Generate a conceptual YARA-like rule to detect malware using structured string data, which includes specific strings, their ranks, and tolerances. Note that the generated rule will be for conceptual use and will not be valid in actual YARA implementations but will serve as a basis for developing practical detection tools.
        
        **Input Data**: Each data point consists of a tuple with a string, its rank, and a tolerance level. The rank quantifies the string's frequency or importance in malware binaries, and the tolerance specifies the permissible variation in rank for potential matches.
        
        **Rule Requirements**:
        1. **String Data Structure**: Each string tuple is defined in the format `s$ = {"string", "rank", "tolerance"}`.
        2. **Strings**: Include malware-indicative strings to search within the binary.
        3. **Rank**: Decimal value indicating the string's typical prominence or frequency in malware.
        4. **Tolerance**: Decimal value indicating the allowable deviation in rank for matching scenarios.
        
        **Rule Structure**:
        - Formulate the rule using a YARA-like syntax to illustrate how it might look.
        - Declare each string with an identifier (`$s1`, `$s2`, etc.), using the specified format.
        - Append a comment next to each string for clarity, indicating its rank and tolerance.
        
        **Example Rule**:
        ```yara
        rule ConceptualMalwareDetection
        {
            meta:
                description = "Conceptual YARA-like rule to detect malware based on strings, ranks, and tolerances. Note: This rule is not syntactically valid for YARA."
        
            strings:
                $s1 = { "percent_define(", "0.027789", "0.2" } /* String: percent_define(, Rank: 0.027789, Tolerance: 0.2 */
                $s2 = { "src/AnnotationList.c", "0.021100", "0.2" } /* String: src/AnnotationList.c, Rank: 0.021100, Tolerance: 0.2 */
                // Continue with additional strings in the specified format
        
            condition:
                $s1 and $s2 and // Include all defined strings with AND logic
        }
        ```
        
        **Instructions for Model**:
        - Reflect each string’s rank and tolerance in the specified format next to the string declaration in the rule.
        - Require all listed strings to match with their respective ranks and tolerances to confirm a detection.
        - Clearly state that the rule is conceptual and intended to demonstrate how data might be used in a YARA-like rule for malware detection, but it is not usable directly with YARA tools.     
          
          
         Here is the data:       
        """

        string_rank_tuples: List[Tuple[str, float, float]] = []

        for string, rank in self._bcc_string_rank_dict:
            # Check if the string is already in the index and the rank is within  the tolerance range
            bcc_hashes = self._string_indexer.search(string, rank, self._tolerance)

            if len(string_rank_tuples) >= self._max_strings_in_signature:
                break

            if len(bcc_hashes) > 0:
                # Log that the string is already in the index
                logger.info(
                    f"String '{string}' with rank {rank}  within tolerance {self._tolerance} is already in the index. \n"
                    f" BCC hashes: {bcc_hashes}")
            else:
                # Log that the string is not in the index
                logger.info(f"String '{string}' with rank {rank} is not in the index.")
                string_rank_tuples.append((string, rank, self._tolerance))


        # Log the string rank tuples that will be used for the signature
        logger.info(f"Strings that will be used for the signature:")
        for string, rank, tolerance in string_rank_tuples:
            logger.info(f"\tString: '{string}' Rank: {rank} Tolerance: {tolerance}")


        prompt = PROMPT + "\n".join([f"$s{i} = {{\"{string}\", \"{rank}\", \"{tolerance}\"}} /* String: {string},"
                                     f" Rank: {rank}, Tolerance: {tolerance} */" for i, (string, rank, tolerance)
                                     in enumerate(string_rank_tuples, start=1)])

        llm_context = llm_completion(prompt)
        yara_like_rule = llm_context.response if llm_context else ""

        logger.info(f"YARA-like rule: {yara_like_rule}")

    def display_top_n_strings(self, n: int = 10):

        self.initialize()

        sorted_string_rank_dict = dict(
            sorted(self._bcc_string_rank_dict.items(), key=lambda item: item[1], reverse=True))
        for i, (string, rank) in enumerate(sorted_string_rank_dict.items()):

            logger.info(f"{i + 1}. {string} : {rank} (length: {len(string)})")
            if i == n - 1:
                break


def main():
    parser = argparse.ArgumentParser(description='Generate binary rank signatures')
    parser.add_argument('--bcc_file', type=str, default=TEST_BCC_FILE_PATH,
                        help='BCC file path.')
    parser.add_argument('--top_n',
                        type=int,
                        default=10,
                        help='Number of top strings to use for generating the signature.')

    args = parser.parse_args()

    binary_rank_signature_generator = BinaryRankSignatureGenerator(bcc_file_path=args.bcc_file,
                                                                   top_n=args.top_n)

    binary_rank_signature_generator.generate_signature()


if __name__ == '__main__':
    main()
