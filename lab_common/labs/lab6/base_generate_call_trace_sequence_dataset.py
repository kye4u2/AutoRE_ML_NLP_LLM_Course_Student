import logging
import multiprocessing
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
import random
from typing import List, Dict, Optional, Union

import numpy as np
from tqdm import tqdm

from blackfyre.common import PICKLE_EXT, BINARY_CONTEXT_CONTAINER_EXT
from blackfyre.utils import mkdir_p
from lab_common.calltracesimulator.binarycaltracesimulator import BinaryCallTraceSimulator
from lab_common.calltracesimulator.functioncalltracesimulator import IRSBContext
from lab_common.common import Label, ROOT_PROJECT_FOLDER_PATH, DEFAULT_MALWARE_PERCENTAGE, NUM_CPUS, RANDOM_SEED
from lab_common.labs.lab5.sentencesummarizer import SentenceSummarizer

DEFAULT_CALL_TRACE_SEQ_LEN = 5

DEFAULT_MAX_CALL_TRACES_PER_BINARY = 200

DEFAULT_MIN_CALL_TRACES_PER_BINARY = 100

DEFAULT_DATASET_SIZE = 100

from logzero import logger

logging.getLogger("binarycontext").setLevel(logging.WARN)
logging.getLogger("pyvex.lifting.gym.arm_spotter").setLevel(logging.CRITICAL)
logging.getLogger("pyvex.lifting.gym.x86_spotter").setLevel(logging.CRITICAL)
logging.getLogger("pyvex.lifting.libvex").setLevel(logging.CRITICAL)
logging.getLogger("binarycalltracesimulator").setLevel(logging.WARN)

logger.setLevel(logging.INFO)

LAB_MALWARE_DATASET_DIR_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab6", "malware")
LAB_BENIGN_DATASET_DIR_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab6", "benign")

sentence_summarizer: Optional[SentenceSummarizer] = None


@dataclass
class CallTraceSequence:
    label: Label
    binary_name: str
    binary_sha256: str
    import_name_window: List[str]
    import_embedding_window: List[np.array]


def generate_call_trace_sequences(job_request: Dict[str, Union[int, str, Label]]):
    global sentence_summarizer

    call_trace_sequences: List[CallTraceSequence] = []

    bcc_file_path: str = job_request['bcc_file_path']
    label: Label = job_request['label']
    call_trace_sequence_length: int = job_request['call_trace_sequence_length']
    max_call_traces_per_binary: int = job_request['max_call_traces_per_binary']
    min_call_traces_per_binary: int = job_request['min_call_traces_per_binary']

    if sentence_summarizer is None:
        sentence_summarizer = SentenceSummarizer()
        sentence_summarizer.initialize()

    min_import_funcs_in_trace = DEFAULT_CALL_TRACE_SEQ_LEN \
        if call_trace_sequence_length <= DEFAULT_CALL_TRACE_SEQ_LEN \
        else call_trace_sequence_length

    try:
        binary_call_trace_sim = BinaryCallTraceSimulator(binary_context_file_path=bcc_file_path,
                                                         min_import_funcs_in_trace=min_import_funcs_in_trace)

        call_trace_irsb_contexts: List[IRSBContext] = binary_call_trace_sim.call_traces

    except Exception as ex:
        logger.exception("Problem encountered running BinaryCallTraceSimulator on {bcc_file_path}...skipping")
        result = {'bcc_file_path': bcc_file_path,
                  'call_trace_sequences': []}
        return result

    call_trace_irsb_context: IRSBContext
    for call_trace_irsb_context in call_trace_irsb_contexts:
        import_names = [call_trace.name for call_trace in call_trace_irsb_context.call_trace_list
                        if call_trace.is_import]

        # Filter out import names that have an embedding of the zero vector
        import_names = [import_name for import_name in import_names
                        if not np.all(sentence_summarizer.summarize(import_name) == 0)]

        # Create call trace windows based on the specified `call_trace_sequence_length`
        N = call_trace_sequence_length
        import_windows = [import_names[i:i + N] for i in range(len(import_names) - N + 1)]

        for import_window in import_windows:
            import_embedding_window = [sentence_summarizer.summarize(import_name) for import_name in import_window]
            call_trace_sequence = CallTraceSequence(label=label,
                                                    binary_name=binary_call_trace_sim.binary_context.name,
                                                    binary_sha256=binary_call_trace_sim.binary_context.sha256_hash,
                                                    import_embedding_window=import_embedding_window,
                                                    import_name_window=import_window)

            call_trace_sequences.append(call_trace_sequence)

    if len(call_trace_sequences) < min_call_traces_per_binary:
        call_trace_sequences = []

    result = {'bcc_file_path': bcc_file_path,
              'call_trace_sequences': call_trace_sequences[:max_call_traces_per_binary]}

    return result


class GenerateCallTraceSequencesDataset(object):

    def __init__(self,
                 malware_dataset_dir_path: str = LAB_MALWARE_DATASET_DIR_PATH,
                 benign_dataset_dir_path: str = LAB_BENIGN_DATASET_DIR_PATH,
                 max_dataset_size: int = DEFAULT_DATASET_SIZE,
                 call_trace_sequence_length: int = DEFAULT_CALL_TRACE_SEQ_LEN,
                 malware_percentage: float = DEFAULT_MALWARE_PERCENTAGE,
                 max_call_traces_per_binary=DEFAULT_MAX_CALL_TRACES_PER_BINARY,
                 min_call_traces_per_binary=DEFAULT_MIN_CALL_TRACES_PER_BINARY,
                 num_cpus: int = NUM_CPUS,
                 cache_folder: str = None):

        self._call_trace_sequence_length: int = call_trace_sequence_length

        self._max_dataset_size: int = max_dataset_size

        self._benign_dataset_dir_path: str = benign_dataset_dir_path

        self._malware_dataset_dir_path: str = malware_dataset_dir_path

        self._initialized: bool = False

        self._malware_percentage: float = malware_percentage

        self._cache_folder: str = cache_folder

        self._num_cpus = num_cpus

        self._max_call_traces_per_binary = max_call_traces_per_binary

        self._min_call_traces_per_binary = min_call_traces_per_binary

        self._call_trace_sequence_dict: Optional[Dict[str, List[CallTraceSequence]]] = None

    def initialize(self):
        if self._initialized:
            return

        # Check if we should load from cache (if available)
        pickle_file_path = None
        if self._cache_folder is not None:

            pickle_file_path = os.path.join(self._cache_folder,
                                            f"call_trace_sequence_dataset_{int(self._malware_percentage * 100)}"
                                            f"_{self._max_dataset_size}_{self._call_trace_sequence_length}"
                                            f"_{self._min_call_traces_per_binary}_{self._max_call_traces_per_binary}."
                                            f"{PICKLE_EXT}")

            if os.path.exists(pickle_file_path):
                logger.info(f"Found cache pickled call trace sequence: '{pickle_file_path}'")
                self._call_trace_sequence_dict = pickle.load(open(pickle_file_path, "rb"))

        if self._call_trace_sequence_dict is None:

            self._call_trace_sequence_dict = {}

            self._generate_dataset()

            # cache the dataset
            if self._cache_folder:
                mkdir_p(self._cache_folder)

                pickle.dump(self._call_trace_sequence_dict, open(pickle_file_path, "wb"))
                logger.info(f"Cached call trace sequence: '{pickle_file_path}'")

        self._initialized = True

        return

    def _generate_dataset(self):

        random.seed(RANDOM_SEED)

        malware_bcc_file_paths = self._get_binary_context_files(self._malware_dataset_dir_path)
        benign_bcc_file_paths = self._get_binary_context_files(self._benign_dataset_dir_path)

        random.shuffle(malware_bcc_file_paths)
        random.shuffle(benign_bcc_file_paths)

        max_malware_call_trace_sequences: int = ceil(float(self._max_dataset_size * self._malware_percentage))
        max_benign_call_trace_sequences: int = ceil(float(self._max_dataset_size * (1 - self._malware_percentage)))

        # ****** Malware ******
        logger.info("****Creating the malware call trace sequences****")
        with multiprocessing.Pool(processes=self._num_cpus) as pool:

            malware_call_trace_sequences: List[CallTraceSequence] = []

            malware_call_trace_requests = [{'bcc_file_path': bcc_file_path,
                                            'call_trace_sequence_length': self._call_trace_sequence_length,
                                            'max_call_traces_per_binary': self._max_call_traces_per_binary,
                                            'min_call_traces_per_binary': self._min_call_traces_per_binary,
                                            'label': Label.MALWARE}
                                           for bcc_file_path in malware_bcc_file_paths]
            for result in pool.imap_unordered(generate_call_trace_sequences, malware_call_trace_requests):
                bcc_file_path = result['bcc_file_path']
                call_trace_sequences = result['call_trace_sequences']

                malware_call_trace_sequences += call_trace_sequences

                logger.info(
                    f"[{min(len(malware_call_trace_sequences), max_malware_call_trace_sequences)}/{max_malware_call_trace_sequences}] "
                    f"Created {len(call_trace_sequences)} malware call trace sequences for binary context: "
                    f"{bcc_file_path}")

                if len(malware_call_trace_sequences) >= max_malware_call_trace_sequences:
                    malware_call_trace_sequences = malware_call_trace_sequences[:max_malware_call_trace_sequences]
                    break
            self._call_trace_sequence_dict["malware"] = malware_call_trace_sequences

        # ****** Benign ******
        logger.info("****Creating the benign call trace sequences****")
        with multiprocessing.Pool(processes=self._num_cpus) as pool:

            benign_call_trace_sequences: List[CallTraceSequence] = []

            benign_call_trace_requests = [{'bcc_file_path': bcc_file_path,
                                           'call_trace_sequence_length': self._call_trace_sequence_length,
                                           'max_call_traces_per_binary': self._max_call_traces_per_binary,
                                           'min_call_traces_per_binary': self._min_call_traces_per_binary,
                                           'label': Label.BENIGN}
                                          for bcc_file_path in sorted(benign_bcc_file_paths)]

            for result in pool.imap_unordered(generate_call_trace_sequences, benign_call_trace_requests):
                bcc_file_path = result['bcc_file_path']
                call_trace_sequences = result['call_trace_sequences']

                benign_call_trace_sequences += call_trace_sequences

                logger.info(
                    f"[{min(len(benign_call_trace_sequences), max_benign_call_trace_sequences)}/{max_benign_call_trace_sequences}] "
                    f"Created {len(call_trace_sequences)} benign call trace sequences for binary context: "
                    f"{bcc_file_path}")

                if len(benign_call_trace_sequences) >= max_benign_call_trace_sequences:
                    benign_call_trace_sequences = benign_call_trace_sequences[:max_benign_call_trace_sequences]
                    break

            self._call_trace_sequence_dict["benign"] = benign_call_trace_sequences

    @staticmethod
    def _get_binary_context_files(binary_context_folder_path: str):
        binary_context_files = sorted([str(path) for index, path in
                                       enumerate(
                                           Path(binary_context_folder_path).rglob(
                                               f'*.{BINARY_CONTEXT_CONTAINER_EXT}'))])

        return binary_context_files

    @property
    def dataset(self) -> Dict[str, List[CallTraceSequence]]:
        self.initialize()
        return self._call_trace_sequence_dict
