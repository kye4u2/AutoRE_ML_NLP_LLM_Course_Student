import argparse
import json
import logging
import math
import re

import backoff
import requests
import time
from typing import Optional
from logzero import logger

from lab_common.llm.client import llm_completion
from lab_common.llm.llm_common import get_llm_client_config, LLMContext, num_tokens_from_string, MAX_TOKEN_LENGTH

llm_client_config = get_llm_client_config()

# Sett logzero log level
logger.setLevel(logging.INFO)


def main() -> None:
    """
    Objective:
    To construct a YARA rule called 'detect_malware_samples' for accurately identifying Poseidon  malware family by utilizing
     unique n-gram patterns extracted from binary code. Formulate the prompt, incorporate data, and use  an LLM to
    generate the rule based on specific criteria to maximize detection and minimize false positives.

    Parameters:
    - None explicitly within the function block, but the function uses predefined prompts and data structures.

    Return Type:
    - None directly from this function, but it initiates the process to generate a YARA rule and handles the response from an LLM.

    Implementation Steps:
    1. Compose a detailed prompt that outlines the YARA rule generation objectives, specifying the focus on common n-grams
       without nearest neighbors and n-grams with minimal nearest neighbors for enhanced detection specificity.
    2. Integrate the provided data containing n-gram information from various malware samples into the prompt.
       This data should include unique and common n-grams that help identify specific and broad malware traits.
    3. The prompt should guide the LLM to formulate a YARA rule, an explanation of the rule's logic, and a justification
        for the n-gram selections made within the rule.
    4. Verify and ensure the YARA rule accurately represents the data and meets the detection criteria set forth in the prompt.

    """


    ### START CODE HERE ###

    yara_generation_prompt = """
    Objective:

    Create a YARA rule named detect_malware_samples to accurately detect malware by identifying unique and specific patterns, known as n-grams, within their binary code. The rule will use:

        Common N-Grams Without Nearest Neighbors: Patterns unique across all malware samples, essential for broad detection.
        N-Grams with Minimal Nearest Neighbors: Patterns unique to individual malware samples, critical for enhancing detection specificity.

    Procedure:

    Step 1: Identification of N-Grams

        Common N-Grams Without Nearest Neighbors: Find n-grams present in all malware samples that are unique, having no similar patterns nearby. These n-grams are foundational for the YARA rule and must be defined in the strings section.
        N-Grams with Minimal Nearest Neighbors: For each malware sample, identify n-grams that exhibit the least number of similar patterns, highlighting unique characteristics. These n-grams are key to the rule's precision.

    Step 2: Constructing YARA Rule Strings

        Inclusion of Common N-Grams: Directly include the common n-grams in the YARA strings. These n-grams establish the rule's detection baseline.
        Conditional Assembly for Specific N-Grams: Formulate a unique AND condition comprising the n-grams with minimal nearest neighbors for each malware sample. Use OR logic to concatenate these conditions, ensuring comprehensive coverage across various malware samples by including their unique identifiers.

    Step 3: YARA Rule Construction

        Strings Section: Define all n-grams identified in Step 1 as strings within the YARA rule. Ensure every n-gram mentioned in the condition section is previously declared to maintain rule integrity.
                         Each string should be named with the sample it is associated such as sample1_ngram1, sample1_ngram2, sample2_ngram1, etc.

        Condition Section: Utilize AND logic to combine the common n-gram with specific sets of n-grams for each malware sample. Then, integrate these sample-specific conditions with OR logic. This structure ensures the rule's broad applicability while accommodating distinct malware traits.

    This formulation guarantees a nuanced approach, enabling the rule to distinguish between different malware samples accurately.

    Verification: Ensure the yara rule represents each malware sample in the provided data.


    Conclusion:

    By clearly defining and structuring the YARA rule, we ensure that each malware sample is represented by its unique set of n-grams in the condition section, enhancing the rule's accuracy and specificity. This approach minimizes false positives and maximizes the detection rate, making detect_malware_samples an effective tool for identifying targeted malware families.

    Expected Response Format:
    {
      "yara_rule": "the yara rule",
      "explanation": "the explanation",
      "justification": "the justification"
    }

    For explanation, make sure you inline cite the specific n-grams that are associated with each malware sample that is in the yara rule.
 
    """

    ### END CODE HERE ###

    DATA = """
    
     
    Here is the data:        
     Common ngrams for all malware samples that have no nearest neighbor: {54 75 65 73 64}, {69 63 6B 43 6F}, {54 6C 73 46 72}, {74 0D 8B 16 89}, {00 00 00 00 68}, {E0 03 2B C8 FF}, {4F 45 4D 43 50}, {14 E8 B5 FE FF}, {17 46 47 49 75}, {67 65 64 20 76}, {4D 20 64 64 2C}, {1C FF 71 18 FF}, {75 73 68 46 69}, {00 00 00 8D A4}, {FF 83 C4 08 85}, {69 76 65 50 6F}, {55 52 50 51 51}, {6F 06 66 0F 7F}, {E9 02 88 47 01}, {E6 03 D1 72 0E}, {5D C3 8B 54 24}, {22 60 64 65 66}, {40 66 0F 6F 6E}, {EB B0 64 8F 05}, {C5 89 45 FC 8B}, {00 00 66 8B 45}, {50 72 6F 63 41}, {22 60 76 62 61}, {74 04 33 C0 EB}, {C8 C1 E0 08 03}
     [Malware sample ceaca3ae6f483342866c8ba4d6136e9812feed0ee2c978a21a799cb749d4be59] Selected ngram: {55 8B EC 83 EC} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {FD 89 4D F4 8A} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {28 FB 41 00 73} - File Hashes (that match on the n-gram): ['89518688079c340bf998dd98e1e1126f4d6e0f12004ec86e59ecbe03df1af648', '852b7a4831b702ec280c32fdcfd3d1cbfe3ab24884207d811933191d5162bf99']; Selected ngram: {6A 03 5B EB 03} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {24 08 2B C8 83} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652']; Selected ngram: {41 00 0C 02 88} - File Hashes (that match on the n-gram): ['45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '80d3d281ab93307f50f069426f8ddd133456a2d5b7ea8aa84e2fc63ffa96d630', '852b7a4831b702ec280c32fdcfd3d1cbfe3ab24884207d811933191d5162bf99']; Selected ngram: {33 F6 89 85 F0} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {74 56 6A 09 FF} - File Hashes (that match on the n-gram): ['2959c295d8505b6e8bdbc2fc82d998ddf32d3613c1a50cc3dfb2f43b2f440717', '6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd', 'ee9325de9533fe09be00f57b6148453422ca6486d7d133cde03a34e65068718e', '558a14d8e2ff665755afc5f80b4830b08f956e2eb01daa8568a9f9c714f3671c']; Selected ngram: {75 0E 57 FF 15} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {EA 41 00 8B 0D} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']
    [Malware sample 9eed477ddff14e826d8565b671eff5b3a00338adb88cb3138a19959ccbe70423] Selected ngram: {33 C0 C2 10 00} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {45 FC 03 C2 66} - File Hashes (that match on the n-gram): ['6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd']; Selected ngram: {00 56 53 6A 04} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 A1 3C 80 43} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {55 FC 03 D1 66} - File Hashes (that match on the n-gram): ['558a14d8e2ff665755afc5f80b4830b08f956e2eb01daa8568a9f9c714f3671c', '6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {C3 33 18 76 03} - File Hashes (that match on the n-gram): ['2959c295d8505b6e8bdbc2fc82d998ddf32d3613c1a50cc3dfb2f43b2f440717', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10', '80d3d281ab93307f50f069426f8ddd133456a2d5b7ea8aa84e2fc63ffa96d630']; Selected ngram: {07 0B 00 00 3B} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {06 83 7D 14 45} - File Hashes (that match on the n-gram): ['ee9325de9533fe09be00f57b6148453422ca6486d7d133cde03a34e65068718e', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e', '373d0778d083a04943032826a3571e5e8d450a9535f5a901db43c791200a774d']; Selected ngram: {14 A3 18 A0 43} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {BB 42 00 74 11} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']
    [Malware sample bbca47c78f5b5689c60a74442a749790d4092e6628b5bdd1a3290a6b0d27841d] Selected ngram: {55 8B EC 83 E4} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 12 42 00 8D} - File Hashes (that match on the n-gram): ['dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {11 42 00 8B E5} - File Hashes (that match on the n-gram): ['86f32f026eb0dd6a4d547179e5c0cb326c8c259bd791012e54a6f98455799887']; Selected ngram: {7A 42 00 2E 3D} - File Hashes (that match on the n-gram): ['ee9325de9533fe09be00f57b6148453422ca6486d7d133cde03a34e65068718e']; Selected ngram: {00 E8 BB EB 00} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 56 88 84 24} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {48 43 00 3B C3} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {13 42 00 85 C0} - File Hashes (that match on the n-gram): ['558a14d8e2ff665755afc5f80b4830b08f956e2eb01daa8568a9f9c714f3671c', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10', '86f32f026eb0dd6a4d547179e5c0cb326c8c259bd791012e54a6f98455799887']; Selected ngram: {D9 35 28 DE 02} - File Hashes (that match on the n-gram): ['ee9325de9533fe09be00f57b6148453422ca6486d7d133cde03a34e65068718e', '89518688079c340bf998dd98e1e1126f4d6e0f12004ec86e59ecbe03df1af648', '852b7a4831b702ec280c32fdcfd3d1cbfe3ab24884207d811933191d5162bf99']; Selected ngram: {33 0B 31 DD 92} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652']
    [Malware sample 341a17bb23972ed69002ae91677dc134aea820c916c2e88a1d5dbc7de8e4b181] Selected ngram: {56 8B F1 C7 06} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 83 C4 04 8B} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 56 8D 8C 24} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {20 FB 41 00 53} - File Hashes (that match on the n-gram): ['89518688079c340bf998dd98e1e1126f4d6e0f12004ec86e59ecbe03df1af648', '852b7a4831b702ec280c32fdcfd3d1cbfe3ab24884207d811933191d5162bf99']; Selected ngram: {00 E8 C2 3E 00} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {30 BD 42 00 8B} - File Hashes (that match on the n-gram): ['45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '2959c295d8505b6e8bdbc2fc82d998ddf32d3613c1a50cc3dfb2f43b2f440717', '558a14d8e2ff665755afc5f80b4830b08f956e2eb01daa8568a9f9c714f3671c']; Selected ngram: {33 0C 38 E8 91} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652']; Selected ngram: {24 BD 42 00 89} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {BD 42 00 57 33} - File Hashes (that match on the n-gram): ['45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd', '2959c295d8505b6e8bdbc2fc82d998ddf32d3613c1a50cc3dfb2f43b2f440717', '86f32f026eb0dd6a4d547179e5c0cb326c8c259bd791012e54a6f98455799887', 'ee9325de9533fe09be00f57b6148453422ca6486d7d133cde03a34e65068718e']; Selected ngram: {14 03 0D 28 BD} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']
    [Malware sample e6a811b7d3f7d4fd70ba27d4bbffbc1b740dac352c3a53b3a0bf6d0692a769c2] Selected ngram: {55 8B EC 83 E4} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {42 00 3B C3 0F} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 52 8D 94 24} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {34 15 00 00 E8} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {24 08 2B C8 83} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652']; Selected ngram: {00 A3 B0 60 42} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {00 89 84 24 24} - File Hashes (that match on the n-gram): ['2a1d8692f445791dc9dc9700e11f2ce68fce9ac5ad2abd56aa0f41f1047b38f1', '89518688079c340bf998dd98e1e1126f4d6e0f12004ec86e59ecbe03df1af648', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e', '852b7a4831b702ec280c32fdcfd3d1cbfe3ab24884207d811933191d5162bf99']; Selected ngram: {6A 01 53 FF 15} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {FF 15 CC 61 41} - File Hashes (that match on the n-gram): ['45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '558a14d8e2ff665755afc5f80b4830b08f956e2eb01daa8568a9f9c714f3671c', '6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd', '373d0778d083a04943032826a3571e5e8d450a9535f5a901db43c791200a774d', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {E8 58 0F 84 22} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', 'fbe5548641d632402786dd2b8435d7fd312ed101c17554e69759b9f52bf52be6', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']
    [Malware sample 06b1035b09478319831ed51f41f6d5f77a1c7642269aac806f8abaeed4ca1992] Selected ngram: {33 C0 C2 10 00} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {45 FC 03 C2 66} - File Hashes (that match on the n-gram): ['6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd']; Selected ngram: {43 00 3B C3 0F} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2']; Selected ngram: {00 56 89 85 24} - File Hashes (that match on the n-gram): ['8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {24 F6 42 00 68} - File Hashes (that match on the n-gram): ['89518688079c340bf998dd98e1e1126f4d6e0f12004ec86e59ecbe03df1af648', '852b7a4831b702ec280c32fdcfd3d1cbfe3ab24884207d811933191d5162bf99']; Selected ngram: {11 D4 41 00 11} - File Hashes (that match on the n-gram): ['ee9325de9533fe09be00f57b6148453422ca6486d7d133cde03a34e65068718e', '8cfb55087fa8e4c1e7bcc580d767cf2c884c1b8c890ad240c1e7009810af6736']; Selected ngram: {82 43 00 33 C5} - File Hashes (that match on the n-gram): ['7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '8e7eac5013699d34af9b001e61be38b410d98ef661ad11af37b356606eb63799', 'db345130e2cd90ea550affa0fc43a285f09c8f15c9382d35334a1bde48eaa92e']; Selected ngram: {55 FC 03 D1 66} - File Hashes (that match on the n-gram): ['558a14d8e2ff665755afc5f80b4830b08f956e2eb01daa8568a9f9c714f3671c', '6232e33d88d2cad0dbe0fc4324f63fb2b9ba833b0d14f7fd529ab52ed7c6dacd', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10']; Selected ngram: {00 68 D0 FE 42} - File Hashes (that match on the n-gram): ['8cfb55087fa8e4c1e7bcc580d767cf2c884c1b8c890ad240c1e7009810af6736', 'dd7a6d34e1a7b599d76ca9a895a1fd10c7dba227dd402a8782316b8f6a53aa10', '80d3d281ab93307f50f069426f8ddd133456a2d5b7ea8aa84e2fc63ffa96d630']; Selected ngram: {32 13 38 CB 73} - File Hashes (that match on the n-gram): ['6adaf93fda108331512c279e967f55379586948ebbc3dab4746e3d13d4736d97', '7b6a2c47f68bdabc73a7dc13ffc5d8c241419161c13f88f85de0e4ab5e5ca7f2', '45d82ff82b57f8403e6d0167f2e8c91f682f43dead77b74870a01a56a78a0652']
    """

    PROMPT = yara_generation_prompt + DATA

    llm_context = llm_completion(PROMPT)
    if llm_context is None:
        logger.info("No response from LLM.")
    else:
        logger.info(f"LLM Context: {llm_context}")


if __name__ == "__main__":
    main()
