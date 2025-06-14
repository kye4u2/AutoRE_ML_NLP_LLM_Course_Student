import logging
import os
import random
from typing import List, Optional, Dict

import faiss
import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Input, Dense
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from blackfyre.utils import setup_custom_logger
from lab_common.common import DEFAULT_KERAS_PATIENCE, RANDOM_SEED, Label, ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer
from lab_common.labs.lab6.base_generate_call_trace_sequence_dataset import CallTraceSequence, \
    GenerateCallTraceSequencesDataset, LAB_BENIGN_DATASET_DIR_PATH

logger = setup_custom_logger(os.path.splitext(os.path.basename(__file__))[0])
logging.getLogger("binarycontext").setLevel(logging.WARN)
logger.setLevel(logging.INFO)

MODEL_FILE_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                               "labs",
                               "lab6",
                               "lstm_anomaly_detector.model")


class BaseLSTMAnomalyDetector(object):

    def __init__(self,
                 call_trace_sequence_length: int = 5,
                 num_layers: int = 1,
                 hidden_layer_size=64,
                 max_dataset_size: int = 500,
                 malware_percentage: float = .20,
                 epochs: int = 100,
                 batch_size: int = 128,
                 k_top_candidates: int = 2,
                 dropout_rate=.15,
                 anomalous_call_trace_threshold=.5,
                 max_call_traces_per_binary=400,
                 min_call_traces_per_binary=200,
                 patience=DEFAULT_KERAS_PATIENCE,
                 cache_folder: str = None):
        self._initialized: bool = False

        self._is_model_trained = False

        self._call_trace_sequence_length = call_trace_sequence_length

        self._dataset: Optional[Dict[str, List[CallTraceSequence]]] = None

        self._cache_folder: str = cache_folder

        self._max_dataset_size: int = max_dataset_size

        self._num_layers: int = num_layers

        self._hidden_layer_size: int = hidden_layer_size

        self._epochs: int = epochs

        self._batch_size: int = batch_size

        self._patience = patience

        self._malware_percentage: float = malware_percentage

        self._dropout_rate: float = dropout_rate

        self._k_top_candidates = k_top_candidates

        self._anomalous_call_trace_threshold = anomalous_call_trace_threshold

        self._index: Optional[faiss.IndexFlatL2] = None

        self._import_name_embeddings: Optional[np.ndarray] = None

        self._sentence_summarize: Optional[SentenceSummarizer] = None

        self._test_benign_call_traces: Optional[List[CallTraceSequence]] = None

        self._max_call_traces_per_binary = max_call_traces_per_binary

        self._min_call_traces_per_binary = min_call_traces_per_binary

        self._model: Optional[keras.Model] = None

    def initialize(self):
        if self._initialized:
            return self._initialized

        # Set the seed for Python's random library
        random.seed(RANDOM_SEED)

        # Set the seed for NumPy
        np.random.seed(RANDOM_SEED)

        # Set the seed for TensorFlow
        tf.random.set_seed(RANDOM_SEED)

        gen_call_trace_seq_dataset = GenerateCallTraceSequencesDataset(cache_folder=self._cache_folder,
                                                                       max_dataset_size=self._max_dataset_size,
                                                                       malware_percentage=self._malware_percentage,
                                                                       call_trace_sequence_length=self._call_trace_sequence_length,
                                                                       max_call_traces_per_binary=self._max_call_traces_per_binary,
                                                                       min_call_traces_per_binary=self._min_call_traces_per_binary,
                                                                       benign_dataset_dir_path=LAB_BENIGN_DATASET_DIR_PATH)

        self._dataset: Dict[str, List[CallTraceSequence]] = gen_call_trace_seq_dataset.dataset

        num_traces = len([call_trace for call_trace in self._dataset['benign'] if
                          call_trace.binary_sha256 == "a455d32b8de771b493d3648bcbcac5edae2bb1f4062046e5d7ccd53e9d1e621e"])

        self._initialized = True

    def train(self):



        self.initialize()

        benign_call_traces: List[CallTraceSequence] = self._dataset['benign']

        # Split dataset by sha hash
        benign_binary_sha256s = list({benign_call_trace.binary_sha256 for benign_call_trace in benign_call_traces})
        benign_train_binary_sha256s, benign_test_binary_sha256s = train_test_split(benign_binary_sha256s, test_size=.20)

        self._test_benign_call_traces = [benign_call_trace for benign_call_trace in benign_call_traces
                                         if benign_call_trace.binary_sha256 in benign_test_binary_sha256s]

        if self.is_trained:
            logger.info("Model has already been trained")
            return

        benign_train_call_traces = [benign_call_trace for benign_call_trace in benign_call_traces
                                    if benign_call_trace.binary_sha256 in benign_train_binary_sha256s]

        X_benign_call_trace: np.ndarray = np.array(
            [np.array(call_trace.import_embedding_window[:self._call_trace_sequence_length - 1])
             for call_trace in benign_train_call_traces])

        num_zero_vectors = len([1 for call_trace in benign_train_call_traces
                                for embedding_element in call_trace.import_embedding_window
                                if np.all(embedding_element == 0)])

        # zero_z = np.all(X_benign_call_trace == 0, axis=2)
        # number_of_zero_vectors = len(np.where(zero_z))
        logger.info(f"Number of zero vectors {num_zero_vectors}")

        y_benign_call_trace: np.ndarray = np.array(
            [np.array(call_trace.import_embedding_window[self._call_trace_sequence_length - 1])
             for call_trace in benign_train_call_traces])

        call_trace_window_size: int = X_benign_call_trace.shape[1]
        import_name_embedding_size: int = X_benign_call_trace.shape[2]
        random.seed(RANDOM_SEED)

        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(call_trace_window_size, import_name_embedding_size)))

        if self._num_layers == 1:
            self._model.add(
                LSTM(self._hidden_layer_size, activation="relu", return_sequences=False, dropout=self._dropout_rate))
        else:
            for _ in range(self._num_layers - 1):
                self._model.add(
                    LSTM(self._hidden_layer_size, activation="relu", return_sequences=True, dropout=self._dropout_rate))
            self._model.add(
                LSTM(self._hidden_layer_size, activation="relu", return_sequences=False, dropout=self._dropout_rate))

        self._model.add(Dense(import_name_embedding_size))

        self._model.compile(loss=MeanSquaredError(),
                            optimizer=Adam(1e-3),
                            metrics=['accuracy'])

        self._model.summary()

        X_train, X_val, y_train, y_val = train_test_split(X_benign_call_trace, y_benign_call_trace, test_size=0.2)

        early_stopping = EarlyStopping(monitor="val_accuracy", patience=self._patience)

        self._model.fit(X_train, y_train, batch_size=self._batch_size, epochs=self._epochs,
                        validation_data=(X_val, y_val), callbacks=[early_stopping])

        self._model.save(MODEL_FILE_PATH)

        self._is_model_trained = True

    def _initialize_index(self):

        logger.info("Initializing the index")
        benign_call_traces = self._dataset['benign']
        import_names = {import_name for call_trace in benign_call_traces
                        for import_name in call_trace.import_name_window}

        logger.info(f"Number of unique import names: {len(import_names)}")

        self._sentence_summarizer = SentenceSummarizer()
        self._sentence_summarizer.initialize()

        self._import_name_embeddings = np.array([self._sentence_summarizer.summarize(import_name)
                                                 for import_name in import_names])

        d = self._import_name_embeddings[0].shape[0]

        self._index = faiss.IndexFlatL2(d)
        self._index.add(self._import_name_embeddings)

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
        """

        raise NotImplementedError("Subclasses must implement this method")



    def evaluate_model(self):

        if not self._is_model_trained:
            self.train()
        # Initialize the index
        self._initialize_index()

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        actual_malware_count = 0
        actual_benign_count = 0

        test_benign_call_traces = self._test_benign_call_traces
        test_malware_call_traces = self._dataset['malware']

        num_binaries = len(
            set(call_trace.binary_sha256 for call_traces in [test_benign_call_traces, test_malware_call_traces]
                for call_trace in call_traces))

        num_binaries_evaluated = 0
        for call_trace_dataset in [test_malware_call_traces, test_benign_call_traces]:

            logger.info(f"Evaluating {call_trace_dataset[0].label.name} call traces")

            if call_trace_dataset[0].label == Label.BENIGN:
                logger.info("Evaulating benign")
                pass

            binary_names = {call_trace.binary_name for call_trace in call_trace_dataset}

            for index, binary_name in enumerate(binary_names):
                binary_call_traces = [call_trace for call_trace in call_trace_dataset
                                      if call_trace.binary_name == binary_name]

                binary_label: Label = binary_call_traces[0].label

                num_traces = len(binary_call_traces)
                num_anomalies = self.detect_anomalies(binary_call_traces,
                                                      self._index,
                                                      self._model,
                                                      self._import_name_embeddings,
                                                      self._call_trace_sequence_length,
                                                      self._k_top_candidates)
                percentage_anomalies = float(num_anomalies / num_traces)

                is_malware: bool = False

                prediction: Label = Label.BENIGN

                if percentage_anomalies > self._anomalous_call_trace_threshold:
                    is_malware = True
                    prediction = Label.MALWARE

                if is_malware and binary_label == Label.MALWARE:
                    true_positives += 1
                elif is_malware and binary_label == Label.BENIGN:
                    false_positives += 1
                elif not is_malware and binary_label == Label.MALWARE:
                    false_negatives += 1
                elif not is_malware and binary_label == Label.BENIGN:
                    true_negatives += 1
                else:
                    raise Exception("We should not get here")

                num_binaries_evaluated += 1

                precision = 0 if true_positives + false_positives == 0 else true_positives / (
                        true_positives + false_positives)

                recall = 0 if true_positives + false_negatives == 0 else true_positives / (
                        true_positives + false_positives)

                correct_predictions = true_positives + true_negatives
                total_predictions = true_positives + true_negatives + false_positives + false_negatives

                accuracy = float(correct_predictions / total_predictions)

                f1_score = 0 if precision + recall == 0 else (2 * (precision * recall)) / (precision + recall)

                msg = f"\nBinary Name: {binary_name} [{num_binaries_evaluated}/{num_binaries}]\n"
                msg += f"Prediction: {prediction.name}  Actual: {binary_label.name}\n"
                msg += f"Percentage of Anomalous call traces: {num_anomalies}/{num_traces} ({percentage_anomalies}) "
                msg += f"Threshold to classify as Malware: {self._anomalous_call_trace_threshold}"
                logger.info(msg)

                msg = f"Precision: {precision:.3f}  Recall: {recall:.3f}   F1-score: {f1_score:.3f} "
                msg += f"Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.3f})\n"
                logger.info(msg)

                if binary_label == Label.MALWARE:
                    actual_malware_count += 1
                else:
                    actual_benign_count += 1

                logger.info(f"Actual Malware Count: {actual_malware_count}  "
                            f"Actual Benign Count : {actual_benign_count}")

    @property
    def is_trained(self):
        if self._is_model_trained:
            return True
        else:
            # Check if the model file exists
            if os.path.exists(MODEL_FILE_PATH):
                self._model = self._model = keras.models.load_model(MODEL_FILE_PATH)
                self._is_model_trained = True
                self.initialize()
                return True
            else:
                return False
