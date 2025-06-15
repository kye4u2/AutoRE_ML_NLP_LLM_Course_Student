import argparse
import os.path


from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.nlp.tokenizer import Tokenizer
from logzero import logger, setup_logger
from pathlib import Path

from labs.lab7.sentence_longformermlm import SentenceLongformerMLM

logger = setup_logger(name=Path(__file__).stem, logfile=f"{Path(__file__).stem}.log", level="INFO")

def run_similarity_tests(model):
    tokenizer = Tokenizer()

    # Define test pairs and expected similarity values (set expected_similarity after first run)
    test_cases = [
        # Similar or related (actual similarity values from new model)
        ("security is important", "security is the highest priority", 0.7570),
        ("security is fascinating", "security is very interesting", 0.6281),
        ("initialize buffer", "setup memory buffer", 0.7525),
        ("connect socket", "open network socket", 0.9022),
        ("open file", "close file", 0.8568),
        ("encrypt data", "decrypt data", 0.8846),
        ("malloc memory", "free memory", 0.8125),

        # Somewhat similar
        ("security is boring", "security is very interesting", 0.5955),
        ("start thread", "begin execution", 0.6217),

        # Dissimilar or unrelated
        ("security is important", "play the record", 0.4871),
        ("encrypt data", "start thread", 0.5098),
        ("connect socket", "record the play", 0.4288),
        ("eat bytes", "open file", 0.5681),
        ("read registry", "encrypt message", 0.6701),
        ("render graphics", "start thread", 0.6108),
        ("decode image", "connect socket", 0.6283),
    ]

    tolerance = 0.01
    passed, failed = 0, 0

    logger.info("===== Starting Embedding Similarity Tests =====")

    for idx, (text1, text2, expected_similarity) in enumerate(test_cases, 1):
        logger.info(f"\n--- Test {idx} ---")
        logger.info(f"Text 1: {text1}")
        logger.info(f"Text 2: {text2}")

        similarity = model.compute_similarity(text1, text2)
        logger.info(f"Computed Cosine Similarity: {similarity:.4f}")

        if expected_similarity is None:
            logger.info("🔎 No expected value provided. Please update this test case after the first run.")
            result = "UNKNOWN"
        else:
            diff = abs(similarity - expected_similarity)
            if diff <= tolerance:
                result = "PASS"
                passed += 1
            else:
                result = "FAIL"
                failed += 1

        test_cases[idx - 1] = (text1, text2, expected_similarity, similarity, result)

    total_tests = len(test_cases)
    unknowns = sum(1 for _, _, exp, _, _ in test_cases if exp is None)


    logger.info("\n===== Detailed Results =====")
    for i, (t1, t2, expected, actual, result) in enumerate(test_cases, 1):
        expected_str = f"{expected:.4f}" if expected is not None else "N/A"
        actual_str = f"{actual:.4f}" if actual is not None else "N/A"
        logger.info(f"Test {i}: {result} | Sim: {actual_str} | Expected: {expected_str} | \"{t1}\" <-> \"{t2}\"")
    logger.info("\n===== End of Detailed Results =====")

    logger.info("\n===== Test Summary =====")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed:      {passed}")
    logger.info(f"Failed:      {failed}")
    logger.info(f"Unknown:     {unknowns} (need expected values)")
    logger.info("\n===== End of Embedding Similarity Tests =====")

def main():

    PRE_TRAINED_MODEL_CHECKPOINT_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                          "lab_datasets","lab7",
                                          "sentence_longformer_mlm",
                                          "small",
                                          "checkpoints",
                                          "longformer_mlm_2025-03-14_06-40-28")

    parser = argparse.ArgumentParser(description="Run CLS embedding similarity tests.")
    parser.add_argument("--checkpoint_dir", type=str, default=PRE_TRAINED_MODEL_CHECKPOINT_PATH, help="Path to model checkpoint directory.")
    args = parser.parse_args()

    model, _ = SentenceLongformerMLM.from_checkpoint(
        args.checkpoint_dir,
        model_filename="model_best_val_accuracy.pt",
    )

    run_similarity_tests(model)

if __name__ == "__main__":
    main()
