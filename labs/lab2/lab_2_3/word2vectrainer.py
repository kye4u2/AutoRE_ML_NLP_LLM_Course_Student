import os
import logging
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from multiprocessing import cpu_count
import argparse
import logzero
from logzero import logger

logzero.loglevel(logging.INFO)

from lab_common.common import ROOT_PROJECT_FOLDER_PATH



def read_bcc_strings(file_path):
    with open(file_path, "r") as file:
        return [simple_preprocess(line) for line in file]

class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self._test_words = test_words
        self._epoch = 0

    def on_epoch_begin(self, model):
        self._epoch += 1
        logger.info(f"Epoch: {self._epoch}")

    def on_epoch_end(self, model):
        logger.info(f"\nModel loss: {model.get_latest_training_loss():,.2f}")
        for word in self._test_words:
            try:
                similar_words = model.wv.most_similar(word)
                logger.info(f"\nMost similar words to '{word}':")
                for similar_word, similarity in similar_words:
                    logger.info(f"  {similar_word:<20}: {similarity:.4f}")
            except KeyError:
                logger.info(f"\n'{word}' not found in the model vocabulary.")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a Word2Vec model with custom parameters.')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for context words (default: 5)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training (default: 5)')
    parser.add_argument('--min_count', type=int, default=5, help='Minimum word count threshold (default: 5)')
    parser.add_argument('--negative', type=int, default=5, help='Number of negative samples (default: 5)')
    parser.add_argument('--sg', type=int, default=1, choices=[0, 1], help='Training algorithm: 1 for skip-gram, 0 for CBOW (default: 0)')
    args = parser.parse_args()

    bcc_strings = read_bcc_strings(os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common", "nlp", "bcc_strings.txt"))

    # Monitor words
    monitor_words = [
        "sock", "port", "hash", "thread", "bin", "cache", "boot", "method", "object",
        "protocol", "algorithm", "root", "client", "cookie", "memory", "heap", "stack",
        "syscall", "fork", "payload", "malloc", "encrypt", "auth", "bind", "exploit",
        "daemon", "packet", "buffer", "xor"
    ]

    monitor = MonitorCallback(monitor_words)

    # Training the Word2Vec model
    word2vec_model = Word2Vec(sentences=bcc_strings,
                              min_count=args.min_count,
                              callbacks=[monitor],
                              vector_size=300,
                              window=args.window_size,
                              workers=cpu_count(),
                              epochs=args.epochs,
                              seed=20,
                              compute_loss=True,
                              negative=args.negative,
                              sg=args.sg  # Skip-gram or CBOW based on the input argument
                              )

    # Save the model
    word2vec_model.save("word2vec.model")

if __name__ == "__main__":
    main()
