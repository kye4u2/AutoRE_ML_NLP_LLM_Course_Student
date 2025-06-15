import argparse
import json
import os
from typing import Optional, List

from logzero import logger

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab12.bcc_ngram_generator import BCCNGramGenerator
from lab_common.llm.client import llm_completion
from labs.lab12.bcc_ngram_indexer import BCCNgramIndexer

LAB_12_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab12")

LAB_12_TEST_BENIGN_FOLDER = os.path.join(LAB_12_DATASET, "small_test_benign")

TEST_BCC_FILE_PATH = os.path.join(LAB_12_DATASET,
                                  "libgpuwork.so.bcc")

TEST_MALWARE_BCC_FILE_PATH = os.path.join(LAB_12_DATASET,
                                          "malware",
                                          "poseidon",
                                          "63E5FA6CB5305B00A8146D0865D63B17_341a17bb23972ed69002ae91677dc134aea820c916c2e88a1d5dbc7de8e4b181.bcc")


class YaraRuleGenerator:
    def __init__(self, bcc_file_path: str):
        self.bcc_file_path = bcc_file_path
        self.rules = []

        self._bcc_ngram_generator: Optional[BCCNGramGenerator] = None

        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self._bcc_ngram_generator = BCCNGramGenerator(self.bcc_file_path)
        self._bcc_ngram_generator.initialize()

        self._initialized = True

    def generate_rule(self,
                      min_yara_strings: int = 5,
                      max_yara_input_strings: int = 30,
                      max_yara_output_strings: int = 10) -> str:

        """
        Objective: Generate a YARA rule that efficiently identifies malware with minimal false positives.
                   The function strategically selects specific n-grams based on their uniqueness and lack of
                    nearest neighbors, constructs a YARA rule from these n-grams, and ensures the rule meets specified
                    performance and specificity criteria.

        Parameters:
        - min_yara_strings (int, optional): The minimum number of n-grams required to form a YARA rule. Default is 5.
        - max_yara_input_strings (int, optional): The maximum number of n-grams to consider for the rule. Default is 30.
        - max_yara_output_strings (int, optional): The maximum number of n-grams to include in the final YARA rule. Default is 10.

        Return Type:
        - str: The finalized YARA rule in string format.

        Key Steps to Implement:
        1. Generate n-grams from malware samples using an existing n-gram generator.
        2. Search for these n-grams in a pre-indexed database to find which have nearest neighbors and which do not.
        3. Filter out n-grams that have no nearest neighbors, indicating high uniqueness.
        4. If the count of unique n-grams is below the minimum required, select additional n-grams based on the
             fewest nearest neighbors and ensure uniqueness.
        5. Craft  a prompt designed for an LLM to generate the final YARA rule along with an explanation and justification.
        6. Parse the LLM's JSON response to extract the YARA rule, its explanation, and its justification.

        """

        self.initialize()

        yara_rule: str = ""  #  Modify to store the generated YARA rule
        explanation: str = "" # Modify to store the explanation of the rule
        justification: str = "" # Modify to store the justification of the rule



        PROMPT = f"""
        To minimize false positives and enhance the precision of malware identification, a YARA rule will be generated based on specific n-grams extracted from malware samples. These n-grams have been meticulously selected for their uniqueness and specificity to the malware, prioritizing those without nearest neighbors to significantly reduce the likelihood of matching benign files.

        **Step 1: Selection of N-Grams**

        N-grams without nearest neighbors are prioritized for their distinctiveness to the malware samples. Following this, n-grams with the fewest nearest neighbors and unique BCC hashes are considered to ensure high specificity. The selection aims to cover a broad spectrum of malware characteristics while ensuring minimal overlap with benign software signatures.

        **Step 2: Rule Construction**

        Each n-gram is incorporated as a separate string within the rule, forming the basis of the malware identification pattern.

        **Step 3: Rule Refinement**
        Limit the number of strings in the rule to balance specificity and performance. 
        In particular , limit the number of strings to {max_yara_output_strings} to avoid excessive complexity.
        The final rule will be optimized to achieve the best possible detection accuracy with minimal false positives.

        **Output Specification**

        The output will be structured in JSON format, comprising three key components:

        1. `yara_rule`: The generated YARA rule, formatted as a string.
        2. `explanation`: A detailed explanation of the rule's purpose and structure.
        3. `justification`: The rationale behind the selection of specific conditions and strings, highlighting their importance in achieving the rule's objectives.
        
        **Do not use ```json or any other code block formatting. The output should be a plain JSON object without any additional text or formatting.**

        Here is the data:
        """

        ### START YOUR CODE HERE ###



        ### END YOUR CODE HERE ###

        logger.info(f"YARA Rule: {yara_rule}")
        logger.info(f"Explanation: {explanation}")
        logger.info(f"Justification: {justification}")


        return yara_rule

def main():
    parser = argparse.ArgumentParser(description='Yara Rule Generator')
    parser.add_argument('--bcc_file', type=str, default=TEST_BCC_FILE_PATH,
                        help='BCC file path.')

    yara_rule_generator = YaraRuleGenerator(TEST_MALWARE_BCC_FILE_PATH)

    yara_rule_generator.generate_rule()


if __name__ == '__main__':
    main()
