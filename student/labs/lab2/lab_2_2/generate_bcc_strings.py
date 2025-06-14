import logging
import os
from tqdm import tqdm
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.nlp.tokenizer import Tokenizer


# Set log level for the binary context module to suppress unnecessary output
logging.getLogger("binarycontext").setLevel(logging.ERROR)

def main():

    # Define the path to the BCC folder
    bcc_folder_path = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets/lab2")

    ### YOUR CODE HERE ###



    ### END OF YOUR CODE ###


if __name__ == "__main__":
    main()
