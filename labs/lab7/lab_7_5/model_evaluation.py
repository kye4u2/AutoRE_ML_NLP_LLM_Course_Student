import argparse
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from logzero import logger, setup_logger

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer

from labs.lab7.sentence_longformermlm import SentenceLongformerMLM

# Configure logging
logger = setup_logger(name=Path(__file__).stem, level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)


def cosine_similarity(tensor_a, tensor_b):
    """Compute cosine similarity between two vectors."""
    tensor_a = torch.tensor(tensor_a).view(-1)
    tensor_b = torch.tensor(tensor_b).view(-1)
    return F.cosine_similarity(tensor_a, tensor_b, dim=0).item()


def run_comparisons(summarizer, models_dict):
    """
    Run comparisons on various groups of test pairs.
    For each group, compute the similarity for each model on every test pair,
    then aggregate the scores to compute the average and median similarity for each model.
    At the end, print a summary of the statistics per model per test group.
    """

    # Define test groups
    test_pairs_default = [
        ("play the record", "record the play"),
        ("the dog chased the cat", "the cat chased the dog"),
        ("the doctor treated the patient", "the patient treated the doctor"),
        ("he went to the bank to deposit money", "he sat by the bank of the river"),
        ("she used the mouse to click", "the mouse ate the cheese"),
        ("the bat flew through the cave", "he swung the bat at the ball"),
        ("she gave the book to him", "he gave the book to her"),
        ("I shot the target with a drone", "I shot the drone with a target"),
        ("the key opened the lock", "the lock opened the key"),
        ("the teacher praised the student", "the student praised the teacher"),
        ("the attacker hit the defender", "the defender hit the attacker"),
        ("the client sent a request to the server", "the server sent a request to the client"),
        ("the program allocated memory", "the memory allocated the program"),
        ("record the failed login attempt", "log the record of events"),
        ("record the failed login attempt", "play back the record of the attack"),
        ("derive the session key from the handshake", "exchange the encryption key"),
        ("derive the session key from the handshake", "parse the key in the JSON object"),
        ("validate the authentication token", "renew the bearer token"),
        ("validate the authentication token", "tokenize the input string"),

        ("the attacker exploited a buffer overflow vulnerability",
         "a buffer overflow was used as an entry point by the attacker"),
        ("multi-factor authentication was required for login", "the user needed MFA to log in"),
        ("the phishing link tricked the employee", "an employee was deceived by a phishing URL"),
        ("the ransomware encrypted critical data files", "important data files were locked by ransomware"),
        ("the firewall denied access from the suspicious IP", "a suspicious IP was blocked by the firewall"),
        ("privilege escalation was observed on the system", "the system showed signs of increased user privileges"),
        ("the malware evaded antivirus detection", "AV tools failed to detect the malware"),
        ("the TLS certificate had expired", "an expired certificate broke the secure connection"),
    ]

    # Organize test groups into a dictionary
    test_groups = {
        "Generic Test Pairs": test_pairs_default,
    }

    # Dictionary to store summary statistics per group
    group_summary_stats = {}

    # Overall similarities (across all groups)
    overall_similarities = {"Word2Vec": []}
    for name in models_dict:
        overall_similarities[name] = []

    # Process each test group
    for group_title, pair_list in test_groups.items():
        logger.info("\n" + "#" * 80)
        logger.info(f"### Processing Group: {group_title}")
        logger.info("#" * 80)

        # Dictionary to collect similarity scores for this group
        group_similarities = {"Word2Vec": []}
        for name in models_dict:
            group_similarities[name] = []

        # Process each test pair in the group
        for idx, (s1, s2) in enumerate(pair_list, start=1):
            logger.info("\n" + "=" * 80)
            logger.info(f"Test Case {idx}:")
            logger.info(f"📝 Sentence A: \"{s1}\"")
            logger.info(f"📝 Sentence B: \"{s2}\"")

            # Compute Word2Vec similarity
            vec1_w2v = summarizer.summarize(s1)
            vec2_w2v = summarizer.summarize(s2)
            sim_w2v = cosine_similarity(vec1_w2v, vec2_w2v)
            group_similarities["Word2Vec"].append(sim_w2v)
            overall_similarities["Word2Vec"].append(sim_w2v)
            logger.info("-" * 80)
            logger.info(f"🔹 Word2Vec Similarity (context-unaware): {sim_w2v:.4f}")

            # Compute similarities for each context-aware model
            for name, model in models_dict.items():
                vec1 = model.generate_embedding(s1).view(-1)
                vec2 = model.generate_embedding(s2).view(-1)
                sim = F.cosine_similarity(vec1, vec2, dim=0).item()
                group_similarities[name].append(sim)
                overall_similarities[name].append(sim)
                logger.info(f"🔹 {name:25} Similarity (context-aware): {sim:.4f}")
            logger.info("=" * 80)


def main():
    pre_trained_small_model_checkpoint = os.path.join(
        ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab7",
        "sentence_longformer_mlm", "small" , "checkpoints", "longformer_mlm_2025-03-14_06-40-28"
    )
    student_trained_ckpt = os.path.join(
        ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab7",
        "sentence_longformer_mlm", "mini", "checkpoints", "longformer_mlm_2025-03-30_14-44-43"
    )

    parser = argparse.ArgumentParser(
        description="Compare Word2Vec and multiple Longformer models with detailed summary statistics per test group."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=pre_trained_small_model_checkpoint,
                        help="Path to small pre-trained Longformer checkpoint directory.")
    parser.add_argument("--student_trained_ckpt", type=str, default=student_trained_ckpt,
                        help="Path to student trained model checkpoint directory.")

    args = parser.parse_args()

    summarizer = SentenceSummarizer()

    small_model, _ = SentenceLongformerMLM.from_checkpoint(
        args.checkpoint_dir, model_filename="model_best_val_accuracy_epoch1.pt"
    )
    student_model, _ = SentenceLongformerMLM.from_checkpoint(
        args.student_trained_ckpt, model_filename="model_best_val_accuracy.pt"
    )

    models_dict = {
        "Longformer (small)": small_model,
        "Longformer (mini-student)": student_model
    }

    run_comparisons(summarizer, models_dict)


if __name__ == "__main__":
    main()
