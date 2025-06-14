#!/usr/bin/env python3
import argparse
import os
import json
import gzip
import re
from collections import Counter

from tqdm import tqdm
import wordninja

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH


def main():


    ### YOUR CODE HERE ###


    ### END OF YOUR CODE ###

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demunge function names using WordNinja")
    parser.add_argument("--use-pretrained-model", action="store_true", help="Use pretrained WordNinja model")
    args = parser.parse_args()

    if not  args.use_pretrained_model:
        main()

    else:

        def is_valid_function_name(name):
            return len(name) >= 6 and name.islower() and name.isalnum()


        function_names = []

        BCC_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab1")

        bcc_file_paths = [os.path.join(BCC_FOLDER_PATH, file) for file in os.listdir(BCC_FOLDER_PATH)]

        PRE_TRAINED_WORD_NINJA = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common/nlp/word_ninja_re_model.txt.gz")

        wordninja_model = wordninja.LanguageModel(PRE_TRAINED_WORD_NINJA)

        for bcc_file_path in tqdm(bcc_file_paths, desc="Extracting function names"):
            print(f"  → Processing {bcc_file_path}")
            vex_binary_context = VexBinaryContext.load_from_file(bcc_file_path)

            for function_context in vex_binary_context.function_context_dict.values():
                name = function_context.name
                if name and is_valid_function_name(name):
                    function_names.append(name)

        print(f"\nFound {len(function_names)} valid munged function names.\n")

        demunged_function_names = [" ".join(wordninja_model.split(name)) for name in function_names]


        for original, demunged in zip(function_names, demunged_function_names):
            print(f"Original: {original} → Demunged: {demunged}")
