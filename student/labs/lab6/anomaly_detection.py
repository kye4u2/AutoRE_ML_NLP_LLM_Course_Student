import argparse
import logging
import os
from typing import List

import faiss
import keras
import numpy as np

from lab_common.labs.lab6.base_generate_call_trace_sequence_dataset import CallTraceSequence
from lab_common.labs.lab6.base_lstm_anomaly_detection import BaseLSTMAnomalyDetector

CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache", os.path.splitext(os.path.basename(__file__))[0])



logging.getLogger("binarycaltracesimulator").setLevel(logging.WARN)


class LSTMAnomalyDetector(BaseLSTMAnomalyDetector):
    # use kwargs to pass in the parameters
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def detect_anomalies(self,
                         binary_call_traces: List[CallTraceSequence],
                         faiss_index: faiss.IndexFlatL2,
                         model: keras.Model,
                         import_name_embeddings: np.array,
                         call_trace_sequence_length: int,
                         k_top_candidates: int) -> int:

        """
            Detects anomalous binary call traces using an LSTM model and k-nearest neighbor search. This function predicts
            vector representations of library function names from binary call traces and differentiates normal from anomalous
            sequences, aiding in malware detection.

            Parameters:
            - binary_call_traces (List[CallTraceSequence]): List of binary call trace sequences to be analyzed.
            - faiss_index (faiss.IndexFlatL2): FAISS index for performing efficient nearest neighbor searches.
            - model (keras.Model): Trained LSTM model used for predicting embeddings.
            - import_name_embeddings (np.array): Array of embeddings for known import names.
            - call_trace_sequence_length (int): Length of the call trace sequences.
            - k_top_candidates (int): Number of top candidate embeddings to consider in the KNN search.

            Returns:
            - int: Number of anomalous binary call traces detected.

            Key Steps:
            1. Prepare the input data by converting binary call traces into a NumPy array, extracting the first N-1 embeddings
               from each call trace as input features.
            2. Use the LSTM model to predict the vector representations of the library function names for each input sequence.
            3. Perform a KNN search for each predicted embedding to find the k closest embeddings in the dataset.
            4. Compare each predicted embedding with the actual embedding of its respective call trace.
            5. Mark a binary call trace as anomalous if the actual embedding does not match any of the k nearest embeddings found.
            6. Return the total count of anomalous binary call traces identified.

            Notes:
            - Before using `faiss_index` for the k-nearest neighbor search (in Step 3), the `predicted_embedding` must be reshaped to match
              the index's expected input dimensions. For example, if the embedding size is 300, reshape the predicted embedding
              like so: `predicted_embedding.reshape(1, 300)`. This ensures compatibility with the FAISS index search method.

            - The `model.predict()` method expects input to be a numpy array. If you have a list of data points, convert it to
              a numpy array using `numpy.array()`. For example: `X_test = np.array(your_list)` where `your_list` contains the
              input data.

        """

        num_anomalous_traces = 0

        ### YOUR CODE HERE ###




        ### END YOUR CODE HERE ###

        return num_anomalous_traces


def main():
    """
    Main function to parse arguments, train, and optionally evaluate the LSTM anomaly detector.
    """

    parser = argparse.ArgumentParser(description="Train and evaluate LSTM Anomaly Detector.")
    parser.add_argument('-e','--run_evaluation', action='store_true',
                        help='Whether to run evaluation after training.')
    parser.add_argument('-k','--k_top_candidates', type=int, default=20,
                        help='Number of top anomaly candidates to consider.')
    parser.add_argument('-t','--anomalous_call_trace_threshold', type=float, default=0.50,
                        help='Threshold to classify a call trace as anomalous.')

    args = parser.parse_args()

    anomaly_detector = LSTMAnomalyDetector(
        cache_folder=CACHE_FOLDER,
        max_dataset_size=50000,
        malware_percentage=0.20,
        call_trace_sequence_length=5,
        patience=5,
        anomalous_call_trace_threshold=args.anomalous_call_trace_threshold,
        hidden_layer_size=256,
        dropout_rate=0.15,
        k_top_candidates=args.k_top_candidates,
        epochs=30,
        num_layers=2,
        batch_size=128
    )

    anomaly_detector.train()

    if args.run_evaluation:
        anomaly_detector.evaluate_model()

if __name__ == "__main__":
    main()

